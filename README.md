# Minimal LM from Scratch

A clean, minimal implementation of a GPT-style language model with BPE tokenization.

## Structure

```
NLM/min_lm/
├── tokenization/       # BPE tokenizer implementation
│   ├── bpe.py         # BPETokenizer class
│   └── __init__.py
├── transformer/        # Model components
│   ├── modules.py     # Basic layers (Linear, Embedding, RMSNorm, SwiGLU)
│   ├── attention.py   # Attention mechanisms (RoPE, SDPA, MultiHeadAttention)
│   ├── model.py       # TransformerBlock, TransformerLM
│   ├── optim.py       # AdamW optimizer
│   ├── utils.py       # Training utilities (loss, scheduling, batching)
│   ├── sampling.py    # Text generation
│   └── __init__.py    # Convenient re-exports
├── scripts/           # CLI tools
│   ├── train_tokenizer.py  # Train BPE tokenizer
│   ├── tokenize.py        # Tokenize text files to .npy
│   ├── train.py           # Train transformer model
│   ├── generate.py        # Generate text from model
│   └── __init__.py
└── __init__.py
```

Additional top-level items:
- `NLM/run_end_to_end.py` (exposed as `nlm-run`)
- `NLM/examples/` for example scripts
- `NLM/ancients/` for dataset preparation utilities
- `NLM/docs/` for documentation

## Setup

```bash
# Install (editable mode recommended during development)
pip install -e .[viz]
```

## Quick Start

### 1. Train a tokenizer

```bash
python -m NLM.min_lm.scripts.train_tokenizer \
  data/train.txt \
  --vocab_size 32000 \
  --output_dir tokenizers/
```

### 2. Tokenize your data

```bash
python -m NLM.min_lm.scripts.tokenize \
  --inputs data/train.txt data/valid.txt \
  --vocab_path tokenizers/train_vocab.json \
  --merges_path tokenizers/train_merges.txt \
  --output_dir data/tokenized/
```

### 3. Train the model (one command end-to-end)

```bash
# TinyStories only (download, tokenize, train, generate)
nlm-run

# With Ancients mixed in (40% share), and flash attention if available
nlm-run --with_ancients --ancient_share 0.4 --flash_attention

# Control dataset size (approximate words) for the mixed dataset
nlm-run --with_ancients --ancient_share 0.5 --target_total_words 200000000
```

### 4. Generate text

```bash
python -m NLM.min_lm.scripts.generate \
  --checkpoint checkpoints/model.pt \
  --vocab_path tokenizers/train_vocab.json \
  --merges_path tokenizers/train_merges.txt \
  --prompt "Once upon a time" \
  --max_tokens 100 \
  --temperature 0.8
```

## CLI Overview

### `nlm-run` - Complete End-to-End Pipeline

The main entry point that runs the entire pipeline from data download to trained model:

```bash
nlm-run [options]
```

#### Data Options
- `--with_ancients`: Include ancient texts from Project Gutenberg
- `--ancient_share FLOAT`: Share of ancient words in mixed dataset (default: 0.4, range: 0.0-1.0)
- `--target_total_words INT`: Approximate total words for mixed dataset size control
- `--request_delay FLOAT`: Delay between Gutendex API requests in seconds (default: 0.3)

#### Overwrite Flags
- `--overwrite_download`: Force re-download of TinyStories dataset
- `--overwrite_ancient`: Force re-download and rebuild of ancient texts
- `--overwrite_mixed`: Force rebuild of mixed dataset
- `--overwrite_tokenizer`: Force retrain tokenizer
- `--overwrite_tokenize`: Force re-tokenization of datasets
- `--overwrite_train`: Start training from scratch, ignore checkpoints
- `--overwrite_all`: Overwrite everything (implies all other overwrite flags)

#### Tokenizer Options
- `--vocab_size INT`: Vocabulary size (default: 10000)

#### Model Architecture
- `--context_length INT`: Maximum sequence length (default: 256)
- `--d_model INT`: Model dimension (default: 512)
- `--num_layers INT`: Number of transformer layers (default: 4)
- `--num_heads INT`: Number of attention heads (default: 16)
- `--d_ff INT`: Feed-forward dimension (default: 1344)
- `--rope_theta FLOAT`: RoPE theta parameter (default: 10000.0)
- `--flash_attention`: Enable Flash Attention (requires CUDA and PyTorch >= 2.0)

#### Training Options
- `--batch_size INT`: Training batch size (default: 64)
- `--total_tokens INT`: Total tokens to train on (default: 163,840,000)
- `--lr_max FLOAT`: Maximum learning rate (default: 3e-4)
- `--lr_min FLOAT`: Minimum learning rate (default: 3e-5)
- `--warmup_iters INT`: Number of warmup iterations (default: 1000)
- `--weight_decay FLOAT`: Weight decay coefficient (default: 0.1)
- `--max_grad_norm FLOAT`: Gradient clipping threshold (default: 1.0)

### Low-Level Scripts

For custom workflows, individual components are available:

#### `nlm-train-tokenizer` - Train BPE Tokenizer
```bash
nlm-train-tokenizer input_file [options]
```
- `input_file`: Path to training corpus
- `--vocab_size INT`: Maximum vocabulary size (default: 10000)
- `--output_dir PATH`: Directory to save tokenizer files (default: current directory)
- `--profile`: Enable profiling

#### `nlm-tokenize` - Tokenize Text Files
```bash
nlm-tokenize --inputs file1.txt file2.txt --vocab_path vocab.json --merges_path merges.txt [options]
```
- `--inputs`: One or more text files to tokenize
- `--vocab_path`: Path to vocabulary JSON file
- `--merges_path`: Path to merges text file
- `--special_tokens`: Special tokens to reserve (default: ["<|endoftext|>"])
- `--output_dir`: Directory for output .npy files (default: "tokenized_data")
- `--output_dtype`: NumPy dtype for tokens (choices: uint16, int32, int64; default: uint16)

#### `nlm-train` - Train Transformer Model
```bash
nlm-train --train_tokens train.npy --val_tokens val.npy --vocab_size 10000 [options]
```
- `--train_tokens`: Path to training tokens (.npy file)
- `--val_tokens`: Path to validation tokens (.npy file)
- `--context_length`: Sequence length (default: 1024)
- `--batch_size`: Batch size (default: 32)
- Model architecture flags (same as nlm-run)
- Training hyperparameter flags (same as nlm-run)
- `--checkpoint_path`: Path to save/load checkpoints
- `--resume`: Resume from checkpoint if exists
- `--device`: Device to use (cpu/cuda/mps, auto-detect if not specified)
- `--log_to_wandb`: Enable Weights & Biases logging
- `--log_csv`: Path to write CSV metrics log
- `--plot_dir`: Directory to save training curve plots
- `--seed`: Random seed for reproducibility

#### `nlm-generate` - Generate Text
```bash
nlm-generate --checkpoint model.pt --vocab_path vocab.json --merges_path merges.txt [options]
```
- `--checkpoint`: Path to model checkpoint
- `--vocab_path`: Path to vocabulary JSON
- `--merges_path`: Path to merges text file
- `--prompt`: Prompt text to continue from (default: empty)
- `--max_tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature, 0 for greedy (default: 1.0)
- `--top_p`: Nucleus sampling threshold (optional)
- `--num_samples`: Number of samples to generate (default: 1)
- `--device`: Device to use (cpu/cuda/mps, auto-detect if not specified)

## Library Usage

```python
from NLM.min_lm.tokenization import BPETokenizer
from NLM.min_lm.transformer import TransformerLM, generate

# Load tokenizer
tokenizer = BPETokenizer.from_files("vocab.json", "merges.txt")

# Create model
model = TransformerLM(
    vocab_size=32000,
    context_length=1024,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    rope_theta=10000.0,
)

# Generate text
prompt_ids = tokenizer.encode("Hello world")
output_ids = generate(model, prompt_ids, max_new_tokens=50)
text = tokenizer.decode(output_ids.tolist())
```

## Model Architecture

- Pre-norm transformer blocks with RMSNorm
- Rotary Position Embeddings (RoPE)
- SwiGLU activation in FFN
- No bias terms
- Supports gradient checkpointing for memory efficiency