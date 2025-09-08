import argparse
import os
import numpy as np
from ..tokenization.bpe import BPETokenizer
import time

def load_tokenizer(vocab_path, merges_path, special_tokens=None):
    """Loads a tokenizer from vocab and merges files."""
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    return BPETokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)

def tokenize_and_serialize(dataset_path, tokenizer, output_dir, output_dtype=np.uint16):
    """Tokenizes a dataset and saves it as a NumPy array."""
    start_time = time.time()
    print(f"Tokenizing {dataset_path}...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        # Using encode_iterable for memory efficiency with large files
        token_ids = list(tokenizer.encode_iterable(f))
    
    # Convert to numpy array with uint16
    token_array = np.array(token_ids, dtype=output_dtype)
    
    # Save to file
    filename = os.path.basename(dataset_path)
    output_filename = os.path.splitext(filename)[0] + ".npy"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, token_array)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Finished tokenizing {dataset_path}.")
    print(f"Saved {len(token_array)} tokens to {output_path}.")
    print(f"Time taken: {duration:.2f} seconds.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize one or more text files with a specified BPE tokenizer and serialize as NumPy arrays."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more input .txt files to tokenize.",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to tokenizer vocab JSON.",
    )
    parser.add_argument(
        "--merges_path",
        type=str,
        required=True,
        help="Path to tokenizer merges TXT.",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens to reserve. Default: <|endoftext|>.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenized_data",
        help="Directory to write .npy files.",
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        choices=["uint16", "int32", "int64"],
        default="uint16",
        help="NumPy dtype for serialized tokens.",
    )
    args = parser.parse_args()

    # Load tokenizer once and reuse for all inputs
    tokenizer = load_tokenizer(args.vocab_path, args.merges_path, args.special_tokens)
    dtype_map = {"uint16": np.uint16, "int32": np.int32, "int64": np.int64}
    out_dtype = dtype_map[args.output_dtype]

    print("--- Starting tokenization ---\n")
    for input_path in args.inputs:
        tokenize_and_serialize(input_path, tokenizer, args.output_dir, out_dtype)
    print("--- All inputs have been tokenized and serialized. ---")

if __name__ == "__main__":
    main()
