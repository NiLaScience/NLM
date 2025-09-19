#!/usr/bin/env python3
"""
One-command end-to-end pipeline for NLM.

Steps (configurable via flags):
1) Download TinyStories train/valid
2) Optionally download and prepare Ancient texts; optionally mix with TinyStories
3) Train BPE tokenizer (or reuse existing)
4) Tokenize datasets to NumPy arrays
5) Train Transformer model with detailed logs
6) Generate sample outputs with diverse settings

Examples:
  # TinyStories only
  python -m NLM.run_end_to_end

  # Include ancients with 40% share and cap total words ~200M words
  python -m NLM.run_end_to_end --with_ancients --ancient_share 0.4 --target_total_words 200000000

  # Force retraining tokenizer and restart training from scratch
  python -m NLM.run_end_to_end --overwrite_tokenizer --overwrite_train

"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from min_lm.tokenization.bpe import BPETokenizer
from min_lm.transformer import (
    TransformerLM, AdamW, cross_entropy_loss, get_batch,
    get_lr_cosine_schedule, gradient_clipping, save_checkpoint,
    load_checkpoint, generate,
)


EOT = "<|endoftext|>"

# Default paths
DATA_DIR = Path("data")
ANC_DIR = Path("ancients")
TOKENIZER_DIR = Path("tokenizers")
TOKENIZED_DIR = Path("tokenized_data")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")

TINYSTORIES_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


def download_file(url: str, dest_path: Path, overwrite: bool = False) -> None:
    import urllib.request
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        print(f"[download] exists: {dest_path}")
        return
    print(f"[download] {url} -> {dest_path}")
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100.0, 100.0 * downloaded / max(total_size, 1))
        sys.stdout.write(f"\r  {percent:6.2f}%")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, dest_path, reporthook=hook)
    print()


def ensure_dirs():
    for p in [DATA_DIR, ANC_DIR, TOKENIZER_DIR, TOKENIZED_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def step_download_tinystories(overwrite: bool = False):
    print("\n== Step 1: Download TinyStories ==")
    train_txt = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
    valid_txt = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
    download_file(TINYSTORIES_TRAIN_URL, train_txt, overwrite)
    download_file(TINYSTORIES_VALID_URL, valid_txt, overwrite)
    return train_txt, valid_txt


def step_optionally_prepare_ancients(with_ancients: bool, request_delay: float, overwrite: bool = False) -> Path | None:
    if not with_ancients:
        return None
    
    print("\n== Step 2: Download Ancient texts via Gutendex ==")
    dataset_path = ANC_DIR / "ancient_texts_dataset.txt"
    
    # Check if final dataset exists
    if dataset_path.exists() and not overwrite:
        print(f"[ancients] using existing: {dataset_path}")
        return dataset_path
    
    raw_dir = ANC_DIR / "raw"
    manifest = raw_dir / "manifest.csv"
    
    # Check if raw downloads exist
    if manifest.exists() and not overwrite:
        print(f"[ancients] raw texts already downloaded: {manifest}")
    else:
        # Use bundled downloader
        from ancients.download_ancient_gutenberg import main as dl_main  # type: ignore
        sys.argv = ["download_ancient_gutenberg.py", "--out_dir", str(raw_dir), "--manifest", str(manifest)]
        dl_main()

    print("Cleaning Gutenberg texts...")
    cleaned_dir = ANC_DIR / "clean"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    from ancients.clean_gutenberg_texts import main as clean_main  # type: ignore
    sys.argv = ["clean_gutenberg_texts.py", "--in_dir", str(raw_dir), "--out_dir", str(cleaned_dir)]
    clean_main()

    print("Building ancient dataset (chunked with EOT separators)...")
    from ancients.build_ancient_dataset import main as build_main  # type: ignore
    sys.argv = [
        "build_ancient_dataset.py",
        "--in_dir", str(cleaned_dir),
        "--out_path", str(dataset_path),
        "--min_words", "500",
        "--max_words", "1000",
    ]
    build_main()
    return dataset_path


def step_train_tokenizer(vocab_size: int, train_corpus: Path, vocab_path: Path, merges_path: Path, overwrite: bool = False) -> None:
    if vocab_path.exists() and merges_path.exists() and not overwrite:
        print(f"[tokenizer] using existing: {vocab_path}, {merges_path}")
        return
    print("\n== Step 3: Train tokenizer ==")
    tok = BPETokenizer.train(input_path=str(train_corpus), vocab_size=vocab_size, special_tokens=[EOT])
    serializable_vocab = {str(k): v.decode('latin-1') for k, v in tok.vocab.items()}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, indent=2, ensure_ascii=False)
    with open(merges_path, 'w', encoding='utf-8') as f:
        for t1, t2 in tok.merges:
            f.write(f"{t1.decode('latin-1')} {t2.decode('latin-1')}\n")
    print(f"[tokenizer] saved: {vocab_path}, {merges_path}")


def _tokenize_to_npy(input_txt: Path, tokenizer: BPETokenizer, output_npy: Path) -> int:
    print(f"[tokenize] {input_txt} -> {output_npy}")
    with open(input_txt, 'r', encoding='utf-8') as f:
        token_ids = list(tokenizer.encode_iterable(f))
    arr = np.array(token_ids, dtype=np.uint16)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, arr)
    return int(arr.shape[0])


def step_tokenize_datasets(train_txt: Path, valid_txt: Path, vocab_path: Path, merges_path: Path, overwrite: bool = False) -> tuple[Path, Path]:
    print("\n== Step 4: Tokenize datasets ==")
    tokenizer = BPETokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=[EOT])
    
    # Handle special case for mixed dataset (backward compatibility)
    if train_txt.stem == "mixed_dataset":
        train_out = TOKENIZED_DIR / "mixed_tokens.npy"
    else:
        train_out = TOKENIZED_DIR / (train_txt.stem + ".npy")
    valid_out = TOKENIZED_DIR / (valid_txt.stem + ".npy")
    
    if not train_out.exists() or overwrite:
        ntr = _tokenize_to_npy(train_txt, tokenizer, train_out)
        print(f"  train tokens: {ntr:,}")
    else:
        print(f"  exists: {train_out}")
        
    if not valid_out.exists() or overwrite:
        nva = _tokenize_to_npy(valid_txt, tokenizer, valid_out)
        print(f"  valid tokens: {nva:,}")
    else:
        print(f"  exists: {valid_out}")
        
    return train_out, valid_out


def step_maybe_mix_datasets(with_ancients: bool, ancient_share: float, target_total_words: int | None, 
                          ancient_dataset: Path | None, ts_train: Path, ts_valid: Path, overwrite: bool = False) -> tuple[Path, Path]:
    if not with_ancients or ancient_dataset is None:
        return ts_train, ts_valid
    print("\n== Step 2b: Build mixed datasets (train & valid) ==")
    
    # For backward compatibility, check if old mixed_dataset.txt exists
    old_mixed = ANC_DIR / "mixed_dataset.txt"
    out_train = ANC_DIR / "mixed_train.txt"
    out_valid = ANC_DIR / "mixed_valid.txt"
    
    # If new files exist and not overwriting, use them
    if out_train.exists() and out_valid.exists() and not overwrite:
        print(f"[mixed] using existing: {out_train}, {out_valid}")
        return out_train, out_valid
    
    # If old mixed file exists but new ones don't, use old approach with TS valid
    if old_mixed.exists() and not out_train.exists() and not overwrite:
        print(f"[mixed] using legacy: {old_mixed} (with TinyStories validation)")
        return old_mixed, ts_valid
        
    # Build new split datasets
    ancient_valid = ANC_DIR / "ancients_valid.txt"
    from ancients.build_mixed_dataset_v2 import build_mixed_splits  # type: ignore
    build_mixed_splits(
        ancient_dataset, 
        ancient_valid,
        ts_train, 
        ts_valid, 
        out_train,
        out_valid,
        ancient_share, 
        seed=123, 
        target_total_words=target_total_words
    )
    return out_train, out_valid


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


def step_train_model(
    train_tokens_path: Path,
    valid_tokens_path: Path,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    batch_size: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    weight_decay: float,
    max_grad_norm: float,
    total_tokens: int,
    flash_attention: bool,
    checkpoint_path: Path,
    overwrite: bool = False,
):
    print("\n== Step 5: Train model ==")
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # Adjust batch size for device lightly
    if device.type == "cpu":
        batch_size = min(batch_size, 16)

    # Iters
    max_iters = max(1, total_tokens // (batch_size * context_length))
    cosine_cycle_iters = max_iters
    print(f"[train] iters={max_iters:,} batch={batch_size} ctx={context_length} tokens={total_tokens:,}")

    # Data (memmap)
    train_tokens = np.load(train_tokens_path, mmap_mode='r')
    val_tokens = np.load(valid_tokens_path, mmap_mode='r')

    # Dtype preference for CUDA
    model_dtype = None
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=model_dtype,
        use_flash_attention=flash_attention,
    ).to(device)

    # Optim
    optimizer = AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=weight_decay)
    
    # Check for existing checkpoint
    start_iter = 0
    if checkpoint_path.exists() and not overwrite:
        print(f"[checkpoint] loading from: {checkpoint_path}")
        try:
            start_iter = load_checkpoint(str(checkpoint_path), model, optimizer)
            print(f"[checkpoint] resumed from iter {start_iter}")
        except Exception as e:
            print(f"[checkpoint] failed to load, starting fresh: {e}")
            start_iter = 0
    
    # Skip training if already completed
    if start_iter >= max_iters:
        print(f"[train] already completed {start_iter} iterations, skipping training")
        return model, device

    # Initial val loss (only if starting fresh)
    if start_iter == 0:
        print("[eval] measuring initial validation loss...")
        init_val = evaluate_loss(model, val_tokens, batch_size, context_length, device, num_batches=20)
        print(f"  initial val loss: {init_val:.4f}")

    # Train loop
    start_time = time.time()
    for it in range(start_iter, max_iters):
        lr_t = get_lr_cosine_schedule(it, lr_max, lr_min, warmup_iters, cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr_t

        xb, yb = get_batch(train_tokens, batch_size, context_length, device)
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)
        optimizer.step()

        if (it + 1) % max(1, max_iters // 200) == 0:
            with torch.no_grad():
                train_loss = float(loss.item())
                val_loss = evaluate_loss(model, val_tokens, batch_size, context_length, device, num_batches=20)
            elapsed = time.time() - start_time
            tokens_seen = (it + 1 - start_iter) * batch_size * context_length
            tok_per_s = tokens_seen / max(elapsed, 1e-9)
            print(
                f"iter {it+1:>7d}/{max_iters} | lr {lr_t:.3e} | train {train_loss:.4f} | val {val_loss:.4f} | tok/s {tok_per_s:,.0f}"
            )

        if (it + 1) % max(1, max_iters // 10) == 0:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, it + 1, str(checkpoint_path))
            print(f"  [saved checkpoint at iter {it+1}]")

    save_checkpoint(model, optimizer, max_iters, str(checkpoint_path))
    print(f"[done] final checkpoint: {checkpoint_path}")
    return model, device


def step_generate_samples(model, device, vocab_path: Path, merges_path: Path, out_path: Path) -> None:
    print("\n== Step 6: Generate samples ==")
    tokenizer = BPETokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=[EOT])
    eos_id = tokenizer.special_tokens.get(EOT, None)

    prompts = [
        "The",
        "Once upon a time",
        "In the ancient city,",
        "A curious child",
        "When the council met,",
    ]
    settings = [
        {"temperature": 0.7, "top_p": 0.9},
        {"temperature": 0.9, "top_p": 0.95},
        {"temperature": 1.0, "top_p": None},
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            for s in settings:
                ids = tokenizer.encode(prompt)
                out_ids = generate(
                    model,
                    ids,
                    max_new_tokens=256,
                    eos_token_id=eos_id,
                    temperature=s["temperature"],
                    top_p=s["top_p"],
                    device=device,
                )
                text = tokenizer.decode(out_ids.tolist())
                f.write(f"\n--- prompt: {prompt} | T={s['temperature']} top_p={s['top_p']} ---\n")
                f.write(text + "\n")
    print(f"[samples] wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Run NLM end-to-end pipeline")
    # Data options
    p.add_argument("--with_ancients", action="store_true", help="Include Ancient texts and optionally mix with TinyStories")
    p.add_argument("--ancient_share", type=float, default=0.4, help="Share of Ancient words in mixed dataset (0..1)")
    p.add_argument("--target_total_words", type=int, default=None, help="Approximate total words for mixed dataset (controls size)")
    p.add_argument("--request_delay", type=float, default=0.3, help="Delay between Gutendex requests (seconds)")

    # Overwrite flags
    p.add_argument("--overwrite_download", action="store_true", help="Force re-download of TinyStories even if exists")
    p.add_argument("--overwrite_ancient", action="store_true", help="Force re-download and rebuild of ancient texts")
    p.add_argument("--overwrite_mixed", action="store_true", help="Force rebuild of mixed dataset")
    p.add_argument("--overwrite_tokenizer", action="store_true", help="Force retrain tokenizer even if exists")
    p.add_argument("--overwrite_tokenize", action="store_true", help="Force re-tokenization even if .npy files exist")
    p.add_argument("--overwrite_train", action="store_true", help="Start training from scratch, ignore existing checkpoints")
    p.add_argument("--overwrite_all", action="store_true", help="Overwrite everything (implies all other overwrite flags)")

    # Tokenizer
    p.add_argument("--vocab_size", type=int, default=10000)

    # Model
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--flash_attention", action="store_true", help="Enable Flash Attention (requires CUDA and PyTorch >= 2.0)")

    # Train
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--total_tokens", type=int, default=163_840_000)
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    args = p.parse_args()

    # Handle overwrite_all
    if args.overwrite_all:
        args.overwrite_download = True
        args.overwrite_ancient = True
        args.overwrite_mixed = True
        args.overwrite_tokenizer = True
        args.overwrite_tokenize = True
        args.overwrite_train = True

    ensure_dirs()

    # 1) TinyStories download
    ts_train_txt, ts_valid_txt = step_download_tinystories(args.overwrite_download)

    # 2) Ancients (optional)
    ancient_dataset = step_optionally_prepare_ancients(args.with_ancients, args.request_delay, args.overwrite_ancient)

    # 2b) Mix (optional)
    train_corpus = ts_train_txt
    valid_corpus = ts_valid_txt
    if args.with_ancients and ancient_dataset is not None:
        train_corpus, valid_corpus = step_maybe_mix_datasets(True, args.ancient_share, args.target_total_words, ancient_dataset, 
                                      ts_train_txt, ts_valid_txt, args.overwrite_mixed)

    # 3) Tokenizer
    vocab_path = TOKENIZER_DIR / "tinystories_vocab.json"
    merges_path = TOKENIZER_DIR / "tinystories_merges.txt"
    step_train_tokenizer(args.vocab_size, train_corpus, vocab_path, merges_path, args.overwrite_tokenizer)

    # 4) Tokenize
    train_npy, valid_npy = step_tokenize_datasets(train_corpus, valid_corpus, vocab_path, merges_path, args.overwrite_tokenize)

    # 5) Train
    ckpt_path = CHECKPOINT_DIR / ("mixed_model.pt" if args.with_ancients else "tinystories_model.pt")
    model, device = step_train_model(
        train_npy,
        valid_npy,
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        args.batch_size,
        args.lr_max,
        args.lr_min,
        args.warmup_iters,
        args.weight_decay,
        args.max_grad_norm,
        args.total_tokens,
        args.flash_attention,
        ckpt_path,
        args.overwrite_train,
    )

    # 6) Generate
    samples_out = OUTPUT_DIR / "generated_samples.txt"
    step_generate_samples(model, device, vocab_path, merges_path, samples_out)


if __name__ == "__main__":
    main()

