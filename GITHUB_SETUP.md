# NLM GitHub Setup Guide

This repository contains a minimal implementation of a GPT-style language model. Due to file size limitations on GitHub, some files are not included and need to be downloaded separately.

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/NLM.git
   cd NLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or for development:
   pip install -e .
   ```

3. **Download the data** (optional - only if you want to train from scratch):
   ```bash
   # Create data directory
   mkdir -p data
   
   # Download TinyStories dataset
   wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -P data/
   wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt -P data/
   ```

## 📁 Repository Structure

```
NLM/
├── min_lm/                 # Core library code
│   ├── tokenization/       # BPE tokenizer
│   ├── transformer/        # Model components
│   └── scripts/           # CLI tools
├── tokenizers/            # Pre-trained tokenizer files (included)
│   ├── tinystories_vocab.json
│   └── tinystories_merges.txt
├── train_tinystories_end2end.py  # End-to-end training example
└── README.md              # Main documentation
```

## 🗄️ Large Files (Not in Git)

The following directories/files are excluded from version control:

- `data/` - Raw text datasets (~2.1GB)
- `tokenized_data/` - Preprocessed token arrays (~2.2GB)
- `checkpoints/` - Model checkpoints (~260MB each)
- `outputs/` - Generated text and logs

## 🎯 Usage Examples

### Train a model from scratch:
```bash
python train_tinystories_end2end.py
```

### Use the pre-trained tokenizer:
```python
from min_lm.tokenization import BPETokenizer

tokenizer = BPETokenizer.from_files(
    "tokenizers/tinystories_vocab.json",
    "tokenizers/tinystories_merges.txt"
)
text = "Hello world!"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
```

### Train your own model:
```bash
# 1. Train tokenizer
python -m min_lm.scripts.train_tokenizer data/train.txt --vocab_size 10000

# 2. Tokenize data
python -m min_lm.scripts.tokenize --inputs data/train.txt data/valid.txt

# 3. Train model
python -m min_lm.scripts.train --train_tokens tokenized_data/train.npy
```

## 🔄 Reproducing Results

To reproduce the TinyStories training:

1. Download the data (see Quick Start)
2. Run `python train_tinystories_end2end.py`
3. The script will:
   - Train a BPE tokenizer (if needed)
   - Tokenize the dataset
   - Train a 4-layer transformer
   - Generate sample outputs

Expected training time:
- GPU (A100): ~30 minutes
- MPS (Apple Silicon): ~3.5 hours
- CPU: ~10+ hours

## 📊 Pre-trained Resources

The repository includes a pre-trained BPE tokenizer for TinyStories:
- Vocabulary size: 10,000 tokens
- Trained on TinyStories V2 dataset
- Special tokens: `<|endoftext|>`

## 🤝 Contributing

Feel free to open issues or submit PRs! The codebase is designed to be educational and easy to understand.

## 📝 License

[Your chosen license]
