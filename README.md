# OlfactoryNet

GNN-based molecular odor prediction system using PyTorch Geometric.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Train a model
python scripts/train.py

# Run predictions
python scripts/predict.py --smiles "CCO"

# Launch web interface
python scripts/app.py
```
