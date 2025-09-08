import os
from typing import BinaryIO, Iterable, Iterator
from multiprocessing import Pool
from collections import Counter
import regex as re
import time
import heapq
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# --- Parallel Pre-tokenization & Frequency Counting (Stateless Utilities) ---

def _process_chunk(args: tuple[str, int, int, str, list[str]]) -> Counter:
    """
    Process a single chunk of the file for pre-tokenization and frequency counting.
    This function is designed to be called by a multiprocessing pool.
    """
    input_path, start, end, PAT, special_tokens = args
    tokenized_words = Counter()
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # Split on special tokens to prevent merging across boundaries
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_segments = re.split(special_pattern, chunk)
        
        # Run pre-tokenization and count frequencies on each segment separately
        for segment in text_segments:
            if segment:
                # re.finditer avoids loading all matches into memory at once
                for match in re.finditer(PAT, segment):
                    word = match.group()
                    tokens = tuple(word.encode("utf-8"))
                    tokenized_words[tokens] += 1
    return tokenized_words

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size


    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def parallel_pretokenize_and_count(
    input_path: str, PAT: str, special_tokens: list[str]
) -> tuple[list[list[int]], list[int]]:
    """
    Reads a file, pre-tokenizes it in parallel, and counts word frequencies.
    This is a memory-efficient implementation that processes the file in chunks
    and aggregates frequency counts, avoiding loading the entire file's tokenized
    representation into memory.
    """
    num_processes = os.cpu_count() or 4
    
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    
    print(f"Found {len(chunk_ranges)} chunks")
    
    args_list = [
        (input_path, start, end, PAT, special_tokens)
        for start, end in chunk_ranges
    ]
    
    merged_counts = Counter()
    with Pool(processes=len(chunk_ranges)) as pool:
        print(f"Using {len(chunk_ranges)} processes")
        # Use imap_unordered for better memory usage as results are processed as they complete
        results = pool.imap_unordered(_process_chunk, args_list)
        for chunk_counts in results:
            merged_counts.update(chunk_counts)
    
    # Convert to list format for merging, which is what the next step expects
    list_of_list_of_ints = []
    word_frequencies = []
    for word_tokens, freq in merged_counts.items():
        list_of_list_of_ints.append(list(word_tokens))
        word_frequencies.append(freq)
    
    return list_of_list_of_ints, word_frequencies


# A wrapper class to invert comparison, allowing a min-heap to function as a max-heap.
class MaxHeapComparable:
    def __init__(self, val):
        self.val = val
    def __lt__(self, other):
        return self.val > other.val
    def __eq__(self, other):
        return self.val == other.val

# --- BPE Tokenizer Class ---

class BPETokenizer:
    """
    A didactic implementation of the Byte Pair Encoding (BPE) tokenizer.
    
    This class can be used to train a new tokenizer from a corpus or to load a
    pre-trained tokenizer for encoding and decoding text.
    
    The implementation follows the standard BPE algorithm:
    1. Start with a vocabulary of all bytes (256 tokens)
    2. Iteratively merge the most frequent adjacent token pairs
    3. Apply learned merges during encoding, in order
    
    Args:
        vocab: Dictionary mapping token IDs to byte sequences
        merges: List of (bytes, bytes) tuples representing merge operations
        special_tokens: List of special token strings (e.g., '<|endoftext|>')
    """
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        self.vocab = vocab or {}
        self.merges = merges or []
        
        # Add special tokens to the vocabulary if they are provided.
        if special_tokens:
            for token in special_tokens:
                if token.encode("utf-8") not in self.vocab.values():
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token.encode("utf-8")

        # Create efficient lookup structures for encoding.
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        # A dictionary mapping a pair of token IDs to its merge rank.
        self.merge_ranks = {
            (self.byte_to_id[p1], self.byte_to_id[p2]): i
            for i, (p1, p2) in enumerate(self.merges)
        }
        # Add special tokens to a separate mapping for easy lookup during encoding.
        self.special_tokens = {
            token: self.byte_to_id[token.encode("utf-8")] for token in sorted(special_tokens or [], key=len, reverse=True)
        }

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        """
        Load a pre-trained tokenizer from vocabulary and merges files.
        Constructs a Tokenizer from a serialized vocabulary and merges file.
        """
        # Load the vocabulary from the JSON file.
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            serializable_vocab = json.load(f)
        vocab = {int(k): v.encode('latin-1') for k, v in serializable_vocab.items()}
        
        # Load the merges from the text file.
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_text = f.read().split('\n')
        
        byte_to_id = {v: k for k, v in vocab.items()}
        merges = []
        for merge_str in merges_text:
            parts = merge_str.split(' ')
            if len(parts) == 2:
                p1_str, p2_str = parts
                p1_bytes, p2_bytes = p1_str.encode('latin-1'), p2_str.encode('latin-1')
                if p1_bytes in byte_to_id and p2_bytes in byte_to_id:
                    merges.append((p1_bytes, p2_bytes))
            
        return cls(vocab, merges, special_tokens)

    @classmethod
    def train(cls, input_path: str, vocab_size: int, special_tokens: list[str]):
        """
        Train a new BPE tokenizer on a text corpus.
        Trains a new tokenizer from a text file and returns a new instance.
        """
        # Step 1: Initialize a temporary vocabulary to get base tokens and special tokens.
        temp_vocab = {}
        for i in range(256):
            temp_vocab[i] = bytes([i])
        for i, token in enumerate(special_tokens):
            temp_vocab[256 + i] = token.encode("utf-8")
        
        # Step 2: Pre-tokenize the text and count initial word frequencies.
        print("PRETOKENIZING")
        word_token_ids, frequencies = parallel_pretokenize_and_count(
            input_path, PAT, special_tokens
        )
        
        # Step 3: Iteratively merge the most frequent token pairs.
        print("MERGING")
        vocab, merges = cls._compute_merges(
            word_token_ids, frequencies, temp_vocab, vocab_size, special_tokens
        )
        
        # Step 4: Return a new, fully initialized tokenizer instance.
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _compute_merges(
        word_token_ids: list[list[int]],
        frequencies: list[int],
        initial_vocab: dict[int, bytes],
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        The core of the BPE algorithm (static method).
        Iteratively finds the most frequent pair of tokens and merges them.
        """
        vocab = initial_vocab.copy()
        merges = []
        
        # --- Step 1: Pre-calculate initial pair counts and locations (run once) ---
        id_pair_counts = Counter()
        # This index maps a pair to a set of indices of words where it occurs.
        # Using a set is important for efficient add/remove operations.
        pair_to_word_indices = {}

        for i, word in enumerate(word_token_ids):
            # Iterate through all adjacent pairs in the word
            for p1, p2 in zip(word, word[1:]):
                pair = (p1, p2)
                # Increment the pair's frequency by the word's frequency
                id_pair_counts[pair] += frequencies[i]
                if pair not in pair_to_word_indices:
                    # Use a set for the list of indices for efficient removal
                    pair_to_word_indices[pair] = set()
                pair_to_word_indices[pair].add(i)

        # --- Step 2: Build the priority queue ---
        # We use a min-heap, so we store negative frequencies to simulate a max-heap.
        # For tie-breaking, we wrap the byte-pair tuple in our custom class
        # to invert the lexicographical comparison.
        pq = [
            (-freq, MaxHeapComparable((vocab[p1], vocab[p2])), (p1, p2))
            for (p1, p2), freq in id_pair_counts.items()
        ]
        heapq.heapify(pq)

        # The number of merges is the target vocab size minus the initial tokens.
        num_merges = vocab_size - len(vocab)
        
        for merge_idx in range(num_merges):
            tic = time.time()
            # --- Step 3: Get the best pair from the priority queue ---
            # This loop implements "lazy deletion". We might pop stale pairs
            # from the heap that have been updated. We keep popping until we find
            # a pair whose frequency in the heap matches its current frequency.
            best_pair = None
            while pq:
                neg_freq, _, pair = heapq.heappop(pq)
                
                # Check for staleness: if the frequency in the heap doesn't match
                # the current frequency in our counts dict, it's an old entry.
                if id_pair_counts.get(pair) == -neg_freq:
                    best_pair = pair
                    break
            
            if best_pair is None:
                break # No valid pairs left in the heap.
            
            # 3. Create a new token for the best pair.
            new_token_id = len(vocab)
            vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
            
            # 4. Perform targeted updates for affected words.
            indices_to_update = list(pair_to_word_indices[best_pair])
            for i in indices_to_update:
                word = word_token_ids[i]
                freq = frequencies[i]
                
                # Get the pairs in this word *before* the merge.
                old_pairs_in_word = Counter(zip(word, word[1:]))

                # Perform the merge operation on the word.
                j = 0
                while j < len(word) - 1:
                    if word[j] == best_pair[0] and word[j+1] == best_pair[1]:
                        word[j:j+2] = [new_token_id]
                    else:
                        j += 1
                
                # Get the pairs in this word *after* the merge.
                new_pairs_in_word = Counter(zip(word, word[1:]))

                # Update the counts and indices for the pairs that changed.
                all_affected_pairs = old_pairs_in_word.keys() | new_pairs_in_word.keys()
                
                for pair in all_affected_pairs:
                    # Calculate the change in frequency for this pair in this specific word.
                    delta_freq = (new_pairs_in_word.get(pair, 0) - old_pairs_in_word.get(pair, 0)) * freq
                    
                    if delta_freq != 0:
                        id_pair_counts[pair] += delta_freq
                        
                        # --- Step 5: Push updated pairs back to the heap ---
                        p1, p2 = pair
                        heapq.heappush(pq, (-id_pair_counts[pair], MaxHeapComparable((vocab[p1], vocab[p2])), pair))

                    # Update the location index if the pair's presence in the word changed.
                    if pair not in new_pairs_in_word and i in pair_to_word_indices.get(pair, set()):
                        pair_to_word_indices[pair].remove(i)
                    if pair not in old_pairs_in_word and pair in new_pairs_in_word:
                        if pair not in pair_to_word_indices:
                            pair_to_word_indices[pair] = set()
                        pair_to_word_indices[pair].add(i)

            # Clean up the merged pair from our cache.
            del id_pair_counts[best_pair]
            del pair_to_word_indices[best_pair]
            
            if merge_idx % 100 == 0:
                toc = time.time()
                print(f"Time taken: {toc - tic} seconds")
                tic = time.time()
            
        return vocab, merges

    def encode(self, text: str) -> list[int]:
        """
        Encode text into a sequence of token IDs.
        Encodes a string into a sequence of token IDs.
        """
        # --- Step 1: Handle Special Tokens ---
        # First, split the text by the special token pattern. This ensures that
        # special tokens are not merged with other text.
        if self.special_tokens:
            special_pattern = "|".join(re.escape(token) for token in self.special_tokens.keys())
            # The regex split will produce segments of text and the special tokens
            # that separated them.
            text_segments = re.split(f"({special_pattern})", text)
        else:
            text_segments = [text]

        all_token_ids = []
        for segment in text_segments:
            if not segment:
                continue
            
            # If the segment is a special token, encode it directly.
            if segment in self.special_tokens:
                all_token_ids.append(self.special_tokens[segment])
                continue

            # --- Step 2: Pre-tokenize and Encode Regular Text ---
            # Use the standard pattern to pre-tokenize the segment.
            for pre_token in re.findall(PAT, segment):
                # Convert pre-token to a list of byte values (initial token IDs).
                byte_sequence = [self.byte_to_id[bytes([b])] for b in pre_token.encode("utf-8")]

                # --- Step 3: Iteratively Apply Merges ---
                while len(byte_sequence) > 1:
                    # Find the next best merge in the current sequence.
                    # The `merge_ranks` gives us the priority of each possible merge.
                    pairs = {
                        (byte_sequence[i], byte_sequence[i+1]): self.merge_ranks.get((byte_sequence[i], byte_sequence[i+1]), float("inf"))
                        for i in range(len(byte_sequence) - 1)
                    }
                    
                    best_pair_to_merge = min(pairs, key=pairs.get)
                    
                    # If the best pair has no rank, it means no more merges are possible.
                    if best_pair_to_merge not in self.merge_ranks:
                        break
                    
                    # Merge the best pair.
                    p1, p2 = best_pair_to_merge
                    new_token_id = self.byte_to_id[self.vocab[p1] + self.vocab[p2]]
                    
                    i = 0
                    new_byte_sequence = []
                    while i < len(byte_sequence):
                        if i < len(byte_sequence) - 1 and (byte_sequence[i], byte_sequence[i+1]) == best_pair_to_merge:
                            new_byte_sequence.append(new_token_id)
                            i += 2
                        else:
                            new_byte_sequence.append(byte_sequence[i])
                            i += 1
                    byte_sequence = new_byte_sequence
                
                all_token_ids.extend(byte_sequence)

        return all_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficiently encode an iterable of strings.
        Lazily encodes an iterable of strings, yielding token IDs.
        This is memory-efficient for large files.
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            
            # Split the buffer by special tokens first.
            if self.special_tokens:
                special_pattern = "|".join(re.escape(token) for token in self.special_tokens.keys())
                segments = re.split(f"({special_pattern})", buffer)
            else:
                segments = [buffer]

            # The last segment might be incomplete. Hold it back.
            buffer = segments.pop(-1)
            
            for segment in segments:
                if not segment:
                    continue
                
                # Yield the special token ID directly if it's a special token.
                if segment in self.special_tokens:
                    yield self.special_tokens[segment]
                    continue
                
                # For regular text, use the standard pre-tokenization and merging.
                pre_tokens = re.findall(PAT, segment)
                for pre_token in pre_tokens:
                    byte_sequence = [self.byte_to_id[bytes([b])] for b in pre_token.encode("utf-8")]

                    while len(byte_sequence) > 1:
                        pairs = {
                            (byte_sequence[i], byte_sequence[i+1]): self.merge_ranks.get((byte_sequence[i], byte_sequence[i+1]), float("inf"))
                            for i in range(len(byte_sequence) - 1)
                        }
                        
                        best_pair_to_merge = min(pairs, key=pairs.get)
                        
                        if best_pair_to_merge not in self.merge_ranks:
                            break
                        
                        p1, p2 = best_pair_to_merge
                        new_token_id = self.byte_to_id[self.vocab[p1] + self.vocab[p2]]
                        
                        i = 0
                        new_byte_sequence = []
                        while i < len(byte_sequence):
                            if i < len(byte_sequence) - 1 and (byte_sequence[i], byte_sequence[i+1]) == best_pair_to_merge:
                                new_byte_sequence.append(new_token_id)
                                i += 2
                            else:
                                new_byte_sequence.append(byte_sequence[i])
                                i += 1
                        byte_sequence = new_byte_sequence
                    
                    for token_id in byte_sequence:
                        yield token_id
        
        # If there's anything left in the buffer after the loop, encode it.
        if buffer:
            final_ids = self.encode(buffer)
            for token_id in final_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back to text.
        Decodes a sequence of token IDs into a string.
        """
        # Look up the byte sequence for each token ID.
        all_bytes = b"".join(self.vocab.get(id, b'') for id in ids)
        # Decode the concatenated bytes into a string, replacing errors.
        return all_bytes.decode("utf-8", errors="replace")
    
