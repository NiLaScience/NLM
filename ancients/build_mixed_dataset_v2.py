#!/usr/bin/env python3
"""
Build mixed datasets with proper train/validation splits maintaining the same distribution.
"""
import argparse
import os
import random
import re
from pathlib import Path

EOT = "<|endoftext|>"


def read_chunks(path: Path) -> list[str]:
    data = path.read_text(encoding="utf-8")
    return [c.strip() for c in data.split(EOT) if c.strip()]


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def downsample_chunks(chunks: list[str], target_words: int) -> list[str]:
    """Downsample chunks to approximately match target word count."""
    random.shuffle(chunks)
    out = []
    total_words = 0
    for c in chunks:
        w = count_words(c)
        if total_words + w > target_words:
            break
        out.append(c)
        total_words += w
    return out if out else chunks[:1]  # at least one chunk


def build_mixed_splits(
    ancient_train_path: Path,
    ancient_valid_path: Path, 
    tinystories_train_path: Path,
    tinystories_valid_path: Path,
    out_train_path: Path,
    out_valid_path: Path,
    ancient_share: float,
    seed: int = 123,
    target_total_words: int | None = None,
    valid_ratio: float = 0.05  # Approximate validation set size
) -> None:
    """Build mixed train and validation sets with consistent distribution."""
    random.seed(seed)
    
    # Read all data
    anc_train_chunks = read_chunks(ancient_train_path)
    anc_valid_chunks = read_chunks(ancient_valid_path) if ancient_valid_path.exists() else []
    ts_train_chunks = read_chunks(tinystories_train_path)
    ts_valid_chunks = read_chunks(tinystories_valid_path)
    
    # Calculate word counts
    anc_train_words = sum(count_words(c) for c in anc_train_chunks)
    anc_valid_words = sum(count_words(c) for c in anc_valid_chunks)
    ts_train_words = sum(count_words(c) for c in ts_train_chunks)
    ts_valid_words = sum(count_words(c) for c in ts_valid_chunks)
    
    total_train_words = anc_train_words + ts_train_words
    total_valid_words = anc_valid_words + ts_valid_words
    
    print(f"[Dataset Stats]")
    print(f"  Ancient train: {anc_train_words:,} words in {len(anc_train_chunks)} chunks")
    print(f"  Ancient valid: {anc_valid_words:,} words in {len(anc_valid_chunks)} chunks")
    print(f"  TinyStories train: {ts_train_words:,} words in {len(ts_train_chunks)} chunks")
    print(f"  TinyStories valid: {ts_valid_words:,} words in {len(ts_valid_chunks)} chunks")
    
    # Apply target size if specified
    if target_total_words is not None:
        target_train_words = int(target_total_words * (1 - valid_ratio))
        target_valid_words = int(target_total_words * valid_ratio)
    else:
        target_train_words = total_train_words
        target_valid_words = total_valid_words
    
    # Calculate target words for each dataset in train/valid
    target_anc_train = int(ancient_share * target_train_words)
    target_ts_train = target_train_words - target_anc_train
    target_anc_valid = int(ancient_share * target_valid_words)
    target_ts_valid = target_valid_words - target_anc_valid
    
    print(f"\n[Target Distribution]")
    print(f"  Train: {target_train_words:,} words ({ancient_share*100:.1f}% ancient)")
    print(f"  Valid: {target_valid_words:,} words ({ancient_share*100:.1f}% ancient)")
    
    # Process training set
    if anc_train_words < target_anc_train:
        # Oversample ancients
        needed = (target_anc_train // max(anc_train_words, 1)) + 1
        anc_train_use = anc_train_chunks * needed
        random.shuffle(anc_train_use)
        anc_train_use = downsample_chunks(anc_train_use, target_anc_train)
    else:
        anc_train_use = downsample_chunks(anc_train_chunks, target_anc_train)
    
    ts_train_use = downsample_chunks(ts_train_chunks, target_ts_train)
    
    # Process validation set
    if anc_valid_chunks:
        if anc_valid_words < target_anc_valid:
            needed = (target_anc_valid // max(anc_valid_words, 1)) + 1
            anc_valid_use = anc_valid_chunks * needed
            random.shuffle(anc_valid_use)
            anc_valid_use = downsample_chunks(anc_valid_use, target_anc_valid)
        else:
            anc_valid_use = downsample_chunks(anc_valid_chunks, target_anc_valid)
    else:
        # If no ancient validation set, sample from train (but don't use same chunks)
        anc_valid_use = random.sample(anc_train_chunks, min(len(anc_train_chunks)//10, 100))
    
    ts_valid_use = downsample_chunks(ts_valid_chunks, target_ts_valid)
    
    # Shuffle for better mixing
    random.shuffle(anc_train_use)
    random.shuffle(ts_train_use)
    random.shuffle(anc_valid_use) 
    random.shuffle(ts_valid_use)
    
    # Write training set with interleaving
    out_train_path.parent.mkdir(parents=True, exist_ok=True)
    with out_train_path.open("w", encoding="utf-8") as f:
        # Interleave for diversity
        max_len = max(len(anc_train_use), len(ts_train_use))
        for i in range(max_len):
            if i < len(anc_train_use):
                f.write(anc_train_use[i] + "\n" + EOT + "\n")
            if i < len(ts_train_use):
                f.write(ts_train_use[i] + "\n" + EOT + "\n")
    
    # Write validation set with interleaving
    with out_valid_path.open("w", encoding="utf-8") as f:
        max_len = max(len(anc_valid_use), len(ts_valid_use))
        for i in range(max_len):
            if i < len(anc_valid_use):
                f.write(anc_valid_use[i] + "\n" + EOT + "\n")
            if i < len(ts_valid_use):
                f.write(ts_valid_use[i] + "\n" + EOT + "\n")
    
    # Report final stats
    final_train_words = sum(count_words(c) for c in anc_train_use + ts_train_use)
    final_valid_words = sum(count_words(c) for c in anc_valid_use + ts_valid_use)
    anc_train_final = sum(count_words(c) for c in anc_train_use)
    anc_valid_final = sum(count_words(c) for c in anc_valid_use)
    
    print(f"\n[Final Mixed Dataset]")
    print(f"  Train: {final_train_words:,} words ({anc_train_final/final_train_words*100:.1f}% ancient)")
    print(f"  Valid: {final_valid_words:,} words ({anc_valid_final/final_valid_words*100:.1f}% ancient)")
    print(f"  Files: {out_train_path}, {out_valid_path}")


def main():
    p = argparse.ArgumentParser(description="Build mixed train/valid datasets with consistent distribution")
    p.add_argument("--ancient_train", type=str, default="ancients/ancient_texts_dataset.txt")
    p.add_argument("--ancient_valid", type=str, default="ancients/ancients_valid.txt")
    p.add_argument("--ts_train", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    p.add_argument("--ts_valid", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    p.add_argument("--out_train", type=str, default="ancients/mixed_train.txt")
    p.add_argument("--out_valid", type=str, default="ancients/mixed_valid.txt")
    p.add_argument("--ancient_share", type=float, default=0.5)
    p.add_argument("--target_total_words", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    
    build_mixed_splits(
        Path(args.ancient_train),
        Path(args.ancient_valid),
        Path(args.ts_train),
        Path(args.ts_valid),
        Path(args.out_train),
        Path(args.out_valid),
        args.ancient_share,
        args.seed,
        args.target_total_words
    )


if __name__ == "__main__":
    main()
