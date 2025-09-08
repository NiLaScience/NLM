import argparse
import time
import json
import cProfile
import pstats
import io
import os
from ..tokenization.bpe import BPETokenizer

def save_vocab(vocab, file_path):
    """Saves the vocabulary to a JSON file."""
    # JSON keys must be strings, and bytes values need to be serializable.
    # We'll represent bytes as strings using 'latin-1' encoding, which
    # can represent any byte value.
    serializable_vocab = {str(k): v.decode('latin-1') for k, v in vocab.items()}
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, indent=2, ensure_ascii=False)

def save_merges(merges, file_path):
    """Saves the merges to a text file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for token1, token2 in merges:
            # Represent bytes as strings for saving
            f.write(f"{token1.decode('latin-1')} {token2.decode('latin-1')}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on the TinyStories dataset."
    )
    parser.add_argument(
        "input_path", type=str, help="Path to the TinyStories dataset file."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=10000, help="Maximum vocabulary size."
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save the trained tokenizer."
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling."
    )
    args = parser.parse_args()

    print(f"Starting BPE training with vocab size {args.vocab_size}...")
    print(f"Input file: {args.input_path}")

    # --- Training ---
    special_tokens = ["<|endoftext|>"]
    
    start_time = time.time()
    
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    tokenizer = BPETokenizer.train(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    vocab, merges = tokenizer.vocab, tokenizer.merges

    if args.profile:
        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(30) # Print top 30 functions
        print("\n--- Profiling Results ---")
        print(s.getvalue())
        print("-------------------------\n")


    end_time = time.time()
    duration = end_time - start_time

    print(f"\nTraining completed in {duration:.2f} seconds.")

    # --- Save Artifacts ---
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.basename(args.input_path).split('.')[0]
    vocab_path = os.path.join(args.output_dir, f"{base_filename}_vocab.json")
    merges_path = os.path.join(args.output_dir, f"{base_filename}_merges.txt")
    
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)
    print(f"Vocabulary saved to {vocab_path}")
    print(f"Merges saved to {merges_path}")

    # --- Analysis ---
    longest_token = b""
    for token_bytes in vocab.values():
        if len(token_bytes) > len(longest_token):
            longest_token = token_bytes
    
    print(f"\nLongest token in vocabulary has length {len(longest_token)}.")
    # Try to print it, but it might be unreadable bytes
    try:
        print(f"Longest token: '{longest_token.decode('utf-8')}'")
    except UnicodeDecodeError:
        print(f"Longest token (raw bytes): {longest_token}")


if __name__ == "__main__":
    main()
