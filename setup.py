"""Setup script for NLM (Natural Language Model) package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nlm-min",
    version="0.1.0",
    author="NLM Contributors",
    description="A minimal implementation of GPT-style language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlm",  # Update with actual URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "einops>=0.6.0",
        "regex>=2022.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "wandb",
        ],
    },
    entry_points={
        "console_scripts": [
            "nlm-train-tokenizer=min_lm.scripts.train_tokenizer:main",
            "nlm-tokenize=min_lm.scripts.tokenize:main",
            "nlm-train=min_lm.scripts.train:main",
            "nlm-generate=min_lm.scripts.generate:main",
        ],
    },
)
