# NLM GitHub Setup Guide

This repository contains a minimal implementation of a GPT-style language model. Due to file size limitations on GitHub, some files are not included and need to be downloaded separately.

## 🚀 Quick Start

1. Clone and install
```bash
git clone https://github.com/yourusername/NLM.git
cd NLM
pip install -e .[viz]
```

2. One-command pipeline
```bash
nlm-run
# or include ancients
nlm-run --with_ancients --ancient_share 0.4 --flash_attention
```

## 📁 Repository Structure

```
NLM/
├── min_lm/                 # Core library code
│   ├── tokenization/       # BPE tokenizer
│   └── transformer/        # Model components
├── ancients/               # Ancient texts preparation utilities
├── examples/               # Example scripts
├── docs/                   # Documentation
├── run_end_to_end.py       # End-to-end orchestrator (exposed as nlm-run)
└── README.md               # Main documentation
```

## 🗄️ Large Files (Not in Git)

Excluded by .gitignore:

- `data/` - Raw text datasets
- `tokenized_data/` - Preprocessed token arrays
- `checkpoints/` - Model checkpoints
- `outputs/` - Generated text and logs

## 🎯 Low-level Usage Examples

```bash
# Train tokenizer
python -m NLM.min_lm.scripts.train_tokenizer data/train.txt --vocab_size 10000

# Tokenize
python -m NLM.min_lm.scripts.tokenize --inputs data/train.txt data/valid.txt \
  --vocab_path tokenizers/train_vocab.json --merges_path tokenizers/train_merges.txt

# Train
python -m NLM.min_lm.scripts.train --train_tokens tokenized_data/train.npy \
  --val_tokens tokenized_data/valid.npy --vocab_size 10000
```

## 🔄 Reproducing TinyStories

```bash
nlm-run
```

Expected training time varies by hardware.

## 📊 Pre-trained Resources

The repo includes a pre-trained TinyStories tokenizer:
- Vocabulary size: 10,000 tokens
- Special tokens: `<|endoftext|>`

## 🤝 Contributing

Issues and PRs welcome.

## 📝 License

MIT



