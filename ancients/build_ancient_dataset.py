#!/usr/bin/env python3
import argparse
import os
import random
import re
from typing import List

EOT = "<|endoftext|>"


def count_words(text: str) -> int:
    return len([w for w in re.findall(r"\b\w+\b", text)])


def chunk_file(path: str, min_words: int, max_words: int) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # Split on double newlines to get paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", content) if p.strip()]
    chunks: List[str] = []
    buffer: List[str] = []
    words = 0
    for p in paragraphs:
        p_words = count_words(p)
        # If a single paragraph is too large, split it into ~max_words windows
        if p_words > max_words and len(p) > 0:
            sentences = re.split(r"(?<=[.?!])\s+", p)
            cur: List[str] = []
            cur_words = 0
            for s in sentences:
                sw = count_words(s)
                if cur_words + sw > max_words and cur:
                    chunks.append(" ".join(cur).strip())
                    cur = []
                    cur_words = 0
                cur.append(s)
                cur_words += sw
            if cur:
                chunks.append(" ".join(cur).strip())
            continue
        # Otherwise, aggregate paragraphs until we reach the band
        if words + p_words <= max_words:
            buffer.append(p)
            words += p_words
        else:
            if words >= min_words and buffer:
                chunks.append("\n\n".join(buffer).strip())
                buffer = [p]
                words = p_words
            else:
                # If current buffer is too small, but adding p would exceed max, flush buffer anyway
                if buffer:
                    chunks.append("\n\n".join(buffer).strip())
                buffer = [p]
                words = p_words
    if buffer:
        chunks.append("\n\n".join(buffer).strip())
    # Drop any tiny chunks (< min_words) except if it's the only one
    final = []
    for i, ch in enumerate(chunks):
        if count_words(ch) < min_words and len(chunks) > 1:
            # attempt to merge forward
            if i + 1 < len(chunks):
                merged = ch + "\n\n" + chunks[i + 1]
                chunks[i + 1] = merged
            else:
                if final:
                    final[-1] = final[-1] + "\n\n" + ch
        else:
            final.append(ch)
    return [c for c in final if c.strip()]


def build_dataset(in_dir: str, out_path: str, min_words: int, max_words: int, seed: int = 42) -> None:
    files = []
    for root, _, fs in os.walk(in_dir):
        for f in fs:
            if f.lower().endswith(".txt"):
                files.append(os.path.join(root, f))
    random.seed(seed)
    random.shuffle(files)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        total_chunks = 0
        for path in files:
            chunks = chunk_file(path, min_words, max_words)
            for ch in chunks:
                out_f.write(ch.strip() + "\n" + EOT + "\n")
                total_chunks += 1
    print(f"Wrote dataset to {out_path} with {total_chunks} chunks (each ended with {EOT}).")


def main():
    parser = argparse.ArgumentParser(description="Build dataset from cleaned texts with paragraph-aware chunks and <|endoftext|> separators.")
    parser.add_argument("--in_dir", required=True, help="Directory of cleaned .txt files")
    parser.add_argument("--out_path", required=True, help="Output dataset .txt path")
    parser.add_argument("--min_words", type=int, default=500, help="Minimum words per chunk")
    parser.add_argument("--max_words", type=int, default=1000, help="Maximum words per chunk")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for file order")
    args = parser.parse_args()

    build_dataset(args.in_dir, args.out_path, args.min_words, args.max_words, args.seed)


if __name__ == "__main__":
    main()
