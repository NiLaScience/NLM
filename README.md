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

- `nlm-run`: end-to-end pipeline with flags for ancients mixing and flash attention
  - `--with_ancients`: include Ancient texts (download/clean/mix)
  - `--ancient_share 0.4`: approximate share of ancients in mixed dataset
  - `--target_total_words N`: cap mixed dataset size (approximate)
  - Training flags: `--batch_size`, `--total_tokens`, `--d_model`, `--num_layers`, `--num_heads`, `--d_ff`, `--flash_attention`, etc.

- Low-level scripts remain available under `min_lm.scripts.*` for custom workflows.

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