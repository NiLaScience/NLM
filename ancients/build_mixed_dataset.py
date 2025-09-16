#!/usr/bin/env python3
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


def downsample_chunks(chunks: list[str], target_tokens: int, avg_w: int = 180) -> list[str]:
    # Rough downsample by estimating ~tokens from words
    random.shuffle(chunks)
    out = []
    tot = 0
    for c in chunks:
        w = count_words(c)
        est_toks = int(w * 0.9)  # rough
        if tot + est_toks > target_tokens:
            break
        out.append(c)
        tot += est_toks
    return out if out else chunks


def build_mixed(ancient_path: Path, tinystories_train: Path, tinystories_valid: Path, out_mixed: Path, ancient_share: float, seed: int = 123, target_total_words: int | None = None) -> None:
    random.seed(seed)
    anc_chunks = read_chunks(ancient_path)
    ts_train = tinystories_train.read_text(encoding="utf-8")
    ts_valid = tinystories_valid.read_text(encoding="utf-8")
    ts_all = ts_train + ("\n" + EOT + "\n") + ts_valid
    ts_chunks = [c.strip() for c in ts_all.split(EOT) if c.strip()]

    # Estimate token volumes from word counts
    anc_words = sum(count_words(c) for c in anc_chunks)
    ts_words = sum(count_words(c) for c in ts_chunks)
    total_words = anc_words + ts_words

    # Optionally cap the total words for mixed dataset size control
    if target_total_words is not None and total_words > target_total_words:
        target_anc_words = int(ancient_share * target_total_words)
        target_ts_words = target_total_words - target_anc_words
    else:
        target_anc_words = int(ancient_share * total_words)
        target_ts_words = total_words - target_anc_words
    # target ancient share by words ~ tokens
    # Oversample ancients or downsample TinyStories
    if anc_words < target_anc_words:
        # Replicate ancients to reach target
        needed = target_anc_words // max(anc_words, 1)
        anc_pool = anc_chunks * max(1, needed)
        # add a tail
        while sum(count_words(c) for c in anc_pool) < target_anc_words:
            anc_pool += anc_chunks
        anc_use = anc_pool
        ts_use = ts_chunks
    else:
        anc_use = anc_chunks
        ts_use = downsample_chunks(ts_chunks, target_ts_words)

    random.shuffle(anc_use)
    random.shuffle(ts_use)
    out_mixed.parent.mkdir(parents=True, exist_ok=True)
    with out_mixed.open("w", encoding="utf-8") as f:
        # Interleave blocks 1:1 for diversity
        for a, t in zip(anc_use, ts_use):
            f.write(a + "\n" + EOT + "\n")
            f.write(t + "\n" + EOT + "\n")
        # Write any leftovers
        for a in anc_use[len(ts_use):]:
            f.write(a + "\n" + EOT + "\n")
        for t in ts_use[len(anc_use):]:
            f.write(t + "\n" + EOT + "\n")


def main():
    p = argparse.ArgumentParser(description="Build mixed dataset with oversampled ancients")
    p.add_argument("--ancient_path", type=str, default=str(Path("NLM/ancients/ancient_texts_dataset.txt")))
    p.add_argument("--ts_train", type=str, default=str(Path("NLM/data/TinyStoriesV2-GPT4-train.txt")))
    p.add_argument("--ts_valid", type=str, default=str(Path("NLM/data/TinyStoriesV2-GPT4-valid.txt")))
    p.add_argument("--out_path", type=str, default=str(Path("NLM/ancients/mixed_dataset.txt")))
    p.add_argument("--ancient_share", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    build_mixed(Path(args.ancient_path), Path(args.ts_train), Path(args.ts_valid), Path(args.out_path), args.ancient_share, args.seed)
    print(f"Wrote mixed dataset to {args.out_path}")


if __name__ == "__main__":
    main()



