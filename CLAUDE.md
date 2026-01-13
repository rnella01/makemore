# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Makemore is an educational character-level language model toolkit by Andrej Karpathy. It takes a text file (one item per line, typically names) and generates new similar items. The project demonstrates neural language models from bigrams to Transformers.

## Development Commands

```bash
# Setup environment
uv sync                    # Install core dependencies
uv sync --group dev        # Install with dev tools (ruff, pytest, ipykernel)

# Run the main trainer
uv run makemore.py -i names.txt -o out --type transformer

# Linting
.venv/bin/ruff check .     # Check all files
.venv/bin/ruff check . --fix  # Auto-fix issues

# Run educational scripts
uv run python makemore_part1_bigrams.py
uv run python makemore_part2_mlp.py
uv run python makemore_part3_bn.py
```

## Architecture

### Model Implementations

`makemore.py` implements 6 interchangeable model architectures, all inheriting from `nn.Module` with a consistent interface:
- `forward(idx, targets=None)` → returns `(logits, loss)`
- `get_block_size()` → context window size

| Model | Description |
|-------|-------------|
| `bigram` | Simple lookup table |
| `mlp` | Embedding + hidden layer (Bengio 2003) |
| `rnn` | Vanilla RNN with RNNCell |
| `gru` | Gated recurrent unit |
| `bow` | Bag of Words with causal masking |
| `transformer` | Multi-head self-attention (GPT-2 style) |

### Key Components

- `ModelConfig` - Dataclass with all hyperparameters
- `CharDataset` - PyTorch Dataset for character sequences
- `CausalSelfAttention` / `Block` - Transformer building blocks
- `create_datasets()` - Loads data with 90/10 train/test split
- `generate()` - Sampling with temperature and top-k support

### Educational Files

Standalone implementations from Karpathy's "Neural Networks: Zero to Hero" video series:

- `makemore_part1_bigrams.py` - Bigram language model
- `makemore_part2_mlp.py` - MLP with character embeddings (Bengio et al. 2003)
- `makemore_part3_bn.py` - Deep MLP with Batch Normalization (Ioffe & Szegedy 2015), includes custom layer classes (Linear, BatchNorm1d, Tanh)

## Ruff Configuration

- Line length: 120
- Ignores: E501 (line length), E741 (ambiguous variable names - common in ML code like `l`, `I`)
- Quote style: double quotes

## Training Output

Models save to `--work-dir` (default: `out/`):
- `model.pt` - Best model checkpoint (based on test loss)
- TensorBoard logs for training visualization
