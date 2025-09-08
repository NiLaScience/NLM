import argparse
import os

# Reduce CUDA memory fragmentation on long runs; safe no-op on CPU/MPS
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Optional

import numpy as np
import torch
import time
from datetime import datetime
import csv
from pathlib import Path
import random
import math

# Matplotlib is imported lazily in _maybe_plot to avoid hard dependency

from ..transformer import (
    TransformerLM,
    AdamW,
    cross_entropy_loss,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The two lines below are not strictly necessary but can help ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_memmap(path: str, dtype: str = "int64") -> np.ndarray:
    """
    Load a token array with memory mapping so it doesn't have to fit in RAM.
    Assumes the file was saved with numpy (e.g., np.save) and can be loaded
    via np.load with mmap_mode='r'.
    """
    return np.load(path, mmap_mode="r").astype(dtype, copy=False)


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, token_array: np.ndarray, batch_size: int, context_length: int, device: torch.device, num_batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(token_array, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _maybe_plot(log_csv_path: str, plot_dir: str) -> None:
    """
    Read the CSV log and write updated PNG plots to plot_dir.
    Generates:
      - loss_vs_iter.png (train/val losses vs iteration)
      - loss_vs_time.png (val loss vs elapsed seconds)
      - speed_vs_iter.png (tokens/sec vs iteration)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not os.path.exists(log_csv_path):
        return

    iters, lrs, train_losses, val_losses, elapsed_s, tokens_seen, tok_per_s = [], [], [], [], [], [], []
    with open(log_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iters.append(int(row["iter"]))
                lrs.append(float(row["lr"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))
                elapsed_s.append(float(row["elapsed_s"]))
                tokens_seen.append(float(row["tokens_seen"]))
                tok_per_s.append(float(row["tokens_per_s"]))
            except Exception:
                continue

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    if len(iters) == 0:
        return

    # Loss vs iter
    plt.figure(figsize=(7, 4))
    plt.plot(iters, train_losses, label="train_loss")
    plt.plot(iters, val_losses, label="val_loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_vs_iter.png"))
    plt.close()

    # Val loss vs time
    plt.figure(figsize=(7, 4))
    plt.plot(elapsed_s, val_losses, label="val_loss")
    plt.xlabel("elapsed seconds")
    plt.ylabel("val loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_vs_time.png"))
    plt.close()

    # Speed vs iter
    plt.figure(figsize=(7, 4))
    plt.plot(iters, tok_per_s, label="tokens/sec")
    plt.xlabel("iteration")
    plt.ylabel("tokens/sec")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "speed_vs_iter.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer LM from scratch")

    # Data
    parser.add_argument("--train_tokens", type=str, required=True, help="Path to npy tokens file for training (use np.memmap via np.load mmap_mode='r')")
    parser.add_argument("--val_tokens", type=str, required=True, help="Path to npy tokens file for validation (use np.memmap via np.load mmap_mode='r')")
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)

    # Model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=None, help="If None, model uses its own default (e.g., 4*d_model or rounded variant)")
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    # Ablations
    parser.add_argument("--disable_rmsnorm", action="store_true", help="Disable RMSNorm everywhere (ablates layer norm)")
    parser.add_argument("--post_norm", action="store_true", help="Use post-norm transformer blocks instead of pre-norm")
    parser.add_argument("--no_rope", action="store_true", help="Disable RoPE (No position embeddings)")
    parser.add_argument("--silu_ffn", action="store_true", help="Use SiLU FFN (d_ff=4*d_model) instead of SwiGLU")
    parser.add_argument("--checkpoint_activations", action="store_true", help="Enable activation checkpointing to reduce memory")

    # Optimizer & schedule
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=100000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Run control
    parser.add_argument("--max_iters", type=int, default=200000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save checkpoints (e.g., checkpoints/ckpt.pt)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if --checkpoint_path exists")

    # Device
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | mps. If None, auto-detect")

    # Logging
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="transformer-from-scratch")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_csv", type=str, default=None, help="Optional path to write CSV metrics log")
    parser.add_argument("--plot_dir", type=str, default=None, help="If set, write PNG plots of training curves here")
    parser.add_argument("--plot_every", type=int, default=0, help="If >0, re-generate plots every this many evals")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Device selection
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Precision settings for CUDA to reduce memory and enable flash attention
    model_dtype = None
    if device.type == "cuda":
        # Prefer bfloat16 on H100; enables memory-efficient SDPA/Flash
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        # Allow TF32 for matmuls/convs for additional speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Optional wandb
    if args.log_to_wandb:
        try:
            import wandb  # type: ignore
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as e:
            print(f"[warn] wandb not available or failed to init: {e}")
            args.log_to_wandb = False

    # Data (memory-mapped)
    train_tokens = load_memmap(args.train_tokens)
    val_tokens = load_memmap(args.val_tokens)

    # Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff if args.d_ff is not None else 4 * args.d_model,
        rope_theta=args.rope_theta,
        device=device,
        dtype=model_dtype,
        # Ablations
        use_rope=not args.no_rope,
        disable_rmsnorm=args.disable_rmsnorm,
        use_swiglu=not args.silu_ffn,
        post_norm=args.post_norm,
        use_checkpoint=args.checkpoint_activations,
    )
    model.to(device)

    # Calculate and print the initial loss before training
    print("\n" + "="*80)
    print("[info] Calculating initial loss on validation set before training starts...")
    with torch.no_grad():
        # Use a higher batch count for a more stable estimate of initial loss
        initial_val_loss = evaluate_loss(model, val_tokens, args.batch_size, args.context_length, device, num_batches=40)
    print(f"[info] Initial validation loss (average over 40 batches): {initial_val_loss:.4f}")
    print(f"[info] Theoretical random-guess loss should be ~{math.log(args.vocab_size):.4f}")
    print("="*80 + "\n")


    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr_max, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # Resume if requested
    start_iter = 0
    if args.resume and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        try:
            start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
            print(f"[info] Resumed from {args.checkpoint_path} at iter={start_iter}")
        except Exception as e:
            print(f"[warn] Failed to load checkpoint: {e}")

    # Training loop
    model.train()
    start_time = time.time()
    csv_writer = None
    csv_file = None
    if args.log_csv is not None:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        csv_file = open(args.log_csv, mode="a", newline="")
        csv_writer = csv.writer(csv_file)
        # Write header if file is empty
        if csv_file.tell() == 0:
            csv_writer.writerow(["timestamp", "iter", "lr", "train_loss", "val_loss", "elapsed_s", "tokens_seen", "tokens_per_s"])
    for it in range(start_iter, args.max_iters):
        # LR schedule
        lr_t = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.lr_max,
            min_learning_rate=args.lr_min,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_t

        # Batch
        xb, yb = get_batch(train_tokens, args.batch_size, args.context_length, device)

        # Forward + loss
        logits = model(xb)
        # Keep logits dtype (bf16 on CUDA) and use fused CE to avoid large temporaries
        loss = cross_entropy_loss(logits, yb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging
        if (it + 1) % args.eval_interval == 0:
            with torch.no_grad():
                train_loss = float(loss.item())
                val_loss = evaluate_loss(model, val_tokens, args.batch_size, args.context_length, device, num_batches=20)
            elapsed_s = time.time() - start_time
            tokens_seen = (it + 1) * args.batch_size * args.context_length
            tok_per_s = tokens_seen / max(elapsed_s, 1e-9)
            ts = datetime.utcnow().isoformat()
            print(
                f"iter {it+1} | lr {lr_t:.3e} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"elapsed_s {elapsed_s:.1f} | tok/s {tok_per_s:.1f}"
            )
            if csv_writer is not None:
                csv_writer.writerow([ts, it + 1, lr_t, train_loss, val_loss, elapsed_s, tokens_seen, tok_per_s])
                csv_file.flush()
            # Optional auto-plot
            if args.plot_dir is not None and (args.plot_every <= 0 or ((it + 1) // args.eval_interval) % max(args.plot_every, 1) == 0):
                _maybe_plot(args.log_csv, args.plot_dir)
            if args.log_to_wandb:
                try:
                    import wandb  # type: ignore
                    wandb.log({
                        "iter": it + 1,
                        "lr": lr_t,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "time/elapsed_s": elapsed_s,
                        "data/tokens_seen": tokens_seen,
                        "speed/tok_per_s": tok_per_s,
                    })
                except Exception:
                    pass

        # Checkpointing
        if args.checkpoint_path and (it + 1) % args.save_interval == 0:
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            save_checkpoint(model, optimizer, it + 1, args.checkpoint_path)
            print(f"[info] Saved checkpoint to {args.checkpoint_path} at iter={it+1}")

    # Final checkpoint
    if args.checkpoint_path:
        os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
        save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)
        print(f"[info] Saved final checkpoint to {args.checkpoint_path}")

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()