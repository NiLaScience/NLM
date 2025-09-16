#!/usr/bin/env python3
"""
End-to-end training script for TinyStories dataset with NLM.
This script:
1. Downloads the TinyStories dataset
2. Trains a BPE tokenizer with vocab_size=10000
3. Tokenizes the dataset
4. Trains a Transformer model with configs from CS336 instructions
5. Generates sample outputs
"""

import os
import time
import urllib.request
import numpy as np
import torch
import json
from pathlib import Path

from NLM.min_lm.tokenization.bpe import BPETokenizer
from NLM.min_lm.transformer import (
    TransformerLM, AdamW, cross_entropy_loss, get_batch,
    get_lr_cosine_schedule, gradient_clipping, save_checkpoint,
    generate
)

# Configuration from instructions.md
CONFIG = {
    # Tokenizer
    "vocab_size": 10000,
    
    # Model architecture
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,  # roughly 8/3 * d_model while being a multiple of 64
    "rope_theta": 10000.0,
    "num_layers": 4,
    "num_heads": 16,
    
    # Training
    "total_tokens": 327_680_000,
    "batch_size": 64,  # Will adjust based on device
    "lr_max": 3e-4,    # To be tuned
    "lr_min": 3e-5,
    "warmup_iters": 1000,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "eval_interval": 100,
    "save_interval": 1000,
    
    # Generation
    "temperature": 0.8,
    "top_p": 0.9,
    "max_new_tokens": 256,
}

# Paths
DATA_DIR = Path("NLM/data")
TOKENIZER_DIR = Path("NLM/tokenizers")
TOKENIZED_DIR = Path("NLM/tokenized_data")
CHECKPOINT_DIR = Path("NLM/checkpoints")
OUTPUT_DIR = Path("NLM/outputs")

# Dataset URLs
TINYSTORIES_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f"Downloading {url} to {dest_path}...")
    
    def download_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
    
    urllib.request.urlretrieve(url, dest_path, reporthook=download_hook)
    print()  # New line after download


def step1_download_dataset():
    """Download TinyStories dataset if not already present."""
    print("\n" + "="*80)
    print("STEP 1: Downloading TinyStories Dataset")
    print("="*80)
    
    DATA_DIR.mkdir(exist_ok=True)
    
    train_txt = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
    valid_txt = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
    
    # Download training data
    if not train_txt.exists():
        download_file(TINYSTORIES_TRAIN_URL, train_txt)
    else:
        print(f"Training data already exists at {train_txt}")
    
    # Download validation data
    if not valid_txt.exists():
        download_file(TINYSTORIES_VALID_URL, valid_txt)
    else:
        print(f"Validation data already exists at {valid_txt}")
    
    # Print dataset info
    if train_txt.exists():
        train_size = train_txt.stat().st_size / (1024 * 1024)  # MB
        print(f"\nTraining data size: {train_size:.1f} MB")
    if valid_txt.exists():
        valid_size = valid_txt.stat().st_size / (1024 * 1024)  # MB
        print(f"Validation data size: {valid_size:.1f} MB")


def step2_train_tokenizer():
    """Train BPE tokenizer with vocab_size=10000."""
    print("\n" + "="*80)
    print("STEP 2: Training BPE Tokenizer")
    print("="*80)
    
    TOKENIZER_DIR.mkdir(exist_ok=True)
    vocab_path = TOKENIZER_DIR / "tinystories_vocab.json"
    merges_path = TOKENIZER_DIR / "tinystories_merges.txt"
    
    if vocab_path.exists() and merges_path.exists():
        print(f"Tokenizer already exists at {vocab_path} and {merges_path}")
        return
    
    print(f"Training BPE tokenizer with vocab_size={CONFIG['vocab_size']}...")
    start_time = time.time()
    
    # Train tokenizer
    tokenizer = BPETokenizer.train(
        input_path=str(DATA_DIR / "TinyStoriesV2-GPT4-train.txt"),
        vocab_size=CONFIG["vocab_size"],
        special_tokens=["<|endoftext|>"],
    )
    
    # Save vocab and merges
    vocab, merges = tokenizer.vocab, tokenizer.merges
    
    # Save vocab
    serializable_vocab = {str(k): v.decode('latin-1') for k, v in vocab.items()}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, indent=2, ensure_ascii=False)
    
    # Save merges
    with open(merges_path, 'w', encoding='utf-8') as f:
        for token1, token2 in merges:
            f.write(f"{token1.decode('latin-1')} {token2.decode('latin-1')}\n")
    
    duration = time.time() - start_time
    print(f"\nTokenizer training completed in {duration:.2f} seconds")
    print(f"Vocabulary saved to {vocab_path}")
    print(f"Merges saved to {merges_path}")
    
    # Analysis
    longest_token = max(vocab.values(), key=len)
    print(f"\nLongest token has length {len(longest_token)}")
    try:
        print(f"Longest token: '{longest_token.decode('utf-8')}'")
    except UnicodeDecodeError:
        print(f"Longest token (raw bytes): {longest_token}")


def step3_tokenize_dataset():
    """Tokenize the TinyStories dataset."""
    print("\n" + "="*80)
    print("STEP 3: Tokenizing Dataset")
    print("="*80)
    
    TOKENIZED_DIR.mkdir(exist_ok=True)
    
    vocab_path = TOKENIZER_DIR / "tinystories_vocab.json"
    merges_path = TOKENIZER_DIR / "tinystories_merges.txt"
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(
        str(vocab_path), 
        str(merges_path), 
        special_tokens=["<|endoftext|>"]
    )
    
    # Tokenize train and validation sets
    for split in ["train", "valid"]:
        input_path = DATA_DIR / f"TinyStoriesV2-GPT4-{split}.txt"
        output_path = TOKENIZED_DIR / f"TinyStoriesV2-GPT4-{split}.npy"
        
        if output_path.exists():
            print(f"\n{split} tokens already exist at {output_path}")
            token_array = np.load(output_path)
            print(f"Number of tokens: {len(token_array):,}")
            continue
        
        print(f"\nTokenizing {split} split...")
        start_time = time.time()
        
        with open(input_path, 'r', encoding='utf-8') as f:
            token_ids = list(tokenizer.encode_iterable(f))
        
        # Convert to numpy array with uint16
        token_array = np.array(token_ids, dtype=np.uint16)
        
        # Save
        np.save(output_path, token_array)
        
        duration = time.time() - start_time
        print(f"Tokenized {len(token_array):,} tokens in {duration:.2f} seconds")
        print(f"Saved to {output_path}")


def evaluate_loss(model, val_tokens, batch_size, context_length, device, num_batches=20):
    """Evaluate validation loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(val_tokens, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def step4_train_model():
    """Train the Transformer model."""
    print("\n" + "="*80)
    print("STEP 4: Training Transformer Model")
    print("="*80)
    
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Adjust batch size for device
    if device.type == "cpu":
        CONFIG["batch_size"] = 16
        CONFIG["total_tokens"] = 40_000_000  # Reduced for CPU
        print(f"Adjusted batch_size to {CONFIG['batch_size']} for CPU")
        print(f"Adjusted total_tokens to {CONFIG['total_tokens']:,} for CPU")
    elif device.type == "mps":
        CONFIG["batch_size"] = 32
        print(f"Adjusted batch_size to {CONFIG['batch_size']} for MPS")
    
    # Calculate iterations
    max_iters = CONFIG["total_tokens"] // (CONFIG["batch_size"] * CONFIG["context_length"])
    CONFIG["cosine_cycle_iters"] = max_iters  # Full cosine cycle
    
    print(f"\nTraining configuration:")
    print(f"  Total tokens: {CONFIG['total_tokens']:,}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Context length: {CONFIG['context_length']}")
    print(f"  Max iterations: {max_iters:,}")
    
    # Load data
    print("\nLoading tokenized data...")
    train_tokens = np.load(TOKENIZED_DIR / "TinyStoriesV2-GPT4-train.npy", mmap_mode='r')
    val_tokens = np.load(TOKENIZED_DIR / "TinyStoriesV2-GPT4-valid.npy", mmap_mode='r')
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Create model
    print("\nCreating model...")
    model = TransformerLM(
        vocab_size=CONFIG["vocab_size"],
        context_length=CONFIG["context_length"],
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        rope_theta=CONFIG["rope_theta"],
        device=device,
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["lr_max"],
        betas=(0.9, 0.95),
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Training loop
    print("\n" + "-"*60)
    print("Starting training...")
    print("-"*60)
    
    start_time = time.time()
    checkpoint_path = CHECKPOINT_DIR / "tinystories_model.pt"
    
    for it in range(max_iters):
        # Learning rate schedule
        lr_t = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=CONFIG["lr_max"],
            min_learning_rate=CONFIG["lr_min"],
            warmup_iters=CONFIG["warmup_iters"],
            cosine_cycle_iters=CONFIG["cosine_cycle_iters"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_t
        
        # Get batch
        xb, yb = get_batch(train_tokens, CONFIG["batch_size"], CONFIG["context_length"], device)
        
        # Forward pass
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), CONFIG["max_grad_norm"])
        optimizer.step()
        
        # Logging
        if (it + 1) % CONFIG["eval_interval"] == 0:
            train_loss = loss.item()
            val_loss = evaluate_loss(model, val_tokens, CONFIG["batch_size"], CONFIG["context_length"], device)
            elapsed = time.time() - start_time
            tokens_seen = (it + 1) * CONFIG["batch_size"] * CONFIG["context_length"]
            tok_per_sec = tokens_seen / max(elapsed, 1e-9)
            
            print(f"iter {it+1:5d} | lr {lr_t:.3e} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | elapsed {elapsed:6.1f}s | tok/s {tok_per_sec:6.0f}")
        
        # Checkpointing
        if (it + 1) % CONFIG["save_interval"] == 0:
            save_checkpoint(model, optimizer, it + 1, checkpoint_path)
            print(f"  [Saved checkpoint at iter {it+1}]")
    
    # Final checkpoint
    save_checkpoint(model, optimizer, max_iters, checkpoint_path)
    print(f"\n[Training complete! Final checkpoint saved to {checkpoint_path}]")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.1f} minutes")
    
    return model, checkpoint_path


def step5_generate_samples(model=None, checkpoint_path=None):
    """Generate sample outputs from the trained model."""
    print("\n" + "="*80)
    print("STEP 5: Generating Sample Outputs")
    print("="*80)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load model if not provided
    if model is None:
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "tinystories_model.pt"
        
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = TransformerLM(
            vocab_size=CONFIG["vocab_size"],
            context_length=CONFIG["context_length"],
            d_model=CONFIG["d_model"],
            num_layers=CONFIG["num_layers"],
            num_heads=CONFIG["num_heads"],
            d_ff=CONFIG["d_ff"],
            rope_theta=CONFIG["rope_theta"],
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    
    model.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(
        str(TOKENIZER_DIR / "tinystories_vocab.json"),
        str(TOKENIZER_DIR / "tinystories_merges.txt"),
        special_tokens=["<|endoftext|>"]
    )
    eos_id = tokenizer.special_tokens.get("<|endoftext|>", None)
    
    # Prompts to test
    prompts = [
        "The",  # Start with simple word instead of empty
        "Once upon a time",
        "The little girl",
        "One day, a cat",
        "In the forest,",
    ]
    
    # Generate samples
    output_path = OUTPUT_DIR / "generated_samples.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: '{prompt}'")
            f.write(f"\n--- Sample {i+1} ---\n")
            f.write(f"Prompt: '{prompt}'\n")
            f.write("-" * 40 + "\n")
            
            # Encode prompt
            prompt_ids = tokenizer.encode(prompt)
            
            # Generate
            output_ids = generate(
                model,
                prompt_ids,
                max_new_tokens=CONFIG["max_new_tokens"],
                eos_token_id=eos_id,
                temperature=CONFIG["temperature"],
                top_p=CONFIG["top_p"],
                device=device,
            )
            
            # Decode
            generated_text = tokenizer.decode(output_ids.tolist())
            print(generated_text)
            print()
            
            f.write(generated_text + "\n\n")
    
    print(f"\nGenerated samples saved to {output_path}")


def main():
    """Run the complete end-to-end pipeline for TinyStories only."""
    print("="*80)
    print("NLM TinyStories End-to-End Training Pipeline")
    print("="*80)
    
    # Step 1: Download dataset
    step1_download_dataset()
    
    # Step 2: Train tokenizer
    step2_train_tokenizer()
    
    # Step 3: Tokenize dataset
    step3_tokenize_dataset()
    
    # Step 4: Train model
    model, checkpoint_path = step4_train_model()
    
    # Step 5: Generate samples
    step5_generate_samples(model, checkpoint_path)
    
    print("\n" + "="*80)
    print("Pipeline complete!")
    print("="*80)
    print(f"\nArtifacts created:")
    print(f"  - Tokenizer: {TOKENIZER_DIR}/")
    print(f"  - Tokenized data: {TOKENIZED_DIR}/")
    print(f"  - Model checkpoint: {CHECKPOINT_DIR}/")
    print(f"  - Generated samples: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()


