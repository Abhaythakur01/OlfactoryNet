<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-2.4+-3C2179?logo=pyg&logoColor=white" alt="PyTorch Geometric">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style">
</p>

<h1 align="center">OlfactoryNet</h1>

<p align="center">
  <strong>Predicting molecular odors using Graph Neural Networks</strong>
</p>

<p align="center">
  A deep learning system that predicts how molecules smell by learning from their molecular structure.
  <br>
  Built with PyTorch Geometric, trained on real-world olfactory data from Pyrfume.
</p>

---

## Overview

**OlfactoryNet** uses Graph Neural Networks (GNNs) to predict odor descriptors (e.g., *fruity*, *floral*, *woody*) directly from molecular structures represented as SMILES strings. The model learns to map atomic and bond features to human-perceived odor qualities.

### Why This Matters

- **Drug Discovery**: Predict off-target odor effects of pharmaceutical compounds
- **Fragrance Industry**: Accelerate discovery of novel scent molecules
- **Food Science**: Design flavor compounds with desired aromatic profiles
- **Chemical Safety**: Identify potentially malodorous industrial chemicals

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multiple GNN Architectures** | GCN, GAT, GIN, and MPNN models with configurable depth and width |
| **Rich Molecular Features** | 82-dimensional atom features + 12-dimensional bond features |
| **Scaffold Splitting** | Ensures models generalize to structurally novel molecules |
| **Multi-label Classification** | Predicts 61 odor descriptors simultaneously |
| **Interactive Web UI** | Gradio-based interface for real-time predictions |
| **Hydra Configuration** | Flexible experiment management with YAML configs |
| **W&B Integration** | Optional experiment tracking and hyperparameter sweeps |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OlfactoryNet Pipeline                          │
└─────────────────────────────────────────────────────────────────────────────┘

  SMILES String          Molecular Graph           GNN Backbone          Predictions
       │                       │                        │                     │
       ▼                       ▼                        ▼                     ▼
  ┌─────────┐           ┌─────────────┐          ┌──────────┐         ┌──────────┐
  │  "CCO"  │  ──────▶  │  ● ─── ●   │  ──────▶ │ GCN/GAT  │ ──────▶ │ fruity   │
  │         │   RDKit   │  │     │   │   PyG    │ GIN/MPNN │  MLP    │ floral   │
  │ Ethanol │           │  ● ─── ●   │          │ Layers   │  Head   │ sweet    │
  └─────────┘           └─────────────┘          └──────────┘         │ ...      │
                              │                        │              └──────────┘
                              ▼                        ▼
                        Node Features           Graph Embedding
                        • Atom type (44)        • Global pooling
                        • Degree (11)           • Batch norm
                        • Charge (6)            • Dropout
                        • Hybridization (7)
                        • Aromaticity (1)
                        • + 13 more...
```

---

## Results

Trained on **3,456 molecules** with **61 odor labels** using scaffold-based train/val/test split (80/10/10).

| Model | Test AUROC | Test F1 | Parameters | Training Time |
|-------|------------|---------|------------|---------------|
| **GCN** | **0.751** | 0.087 | 301K | ~2 min |
| GAT | 0.739 | **0.123** | 350K | ~6 min |

> **Note**: Scaffold splitting ensures the model is evaluated on structurally different molecules than those seen during training, providing a realistic estimate of generalization performance.

### Performance by Odor Category

The model performs best on well-represented odor categories:

| Category | AUROC | Support |
|----------|-------|---------|
| Fruity | 0.82 | 892 |
| Floral | 0.79 | 654 |
| Woody | 0.77 | 523 |
| Green | 0.76 | 489 |
| Sweet | 0.74 | 1,203 |

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/olfactorynet.git
cd olfactorynet

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Or using uv (faster)
uv pip install -e ".[dev]"
```

### Dependencies

Core dependencies are managed in `pyproject.toml`:

```
torch >= 2.0
torch-geometric >= 2.4
rdkit >= 2023.3
pyrfume >= 0.18
hydra-core >= 1.3
torchmetrics >= 1.0
gradio >= 4.0
```

---

## Usage

### Training a Model

```bash
# Train with default configuration (GCN)
python scripts/train.py

# Train with GAT architecture
python scripts/train.py model=gat

# Train with custom hyperparameters
python scripts/train.py \
    model=gat \
    model.hidden_dim=512 \
    model.num_layers=4 \
    training.learning_rate=0.0005 \
    training.batch_size=64

# Train with Weights & Biases logging
python scripts/train.py training.use_wandb=true
```

### Running the Web Interface

```bash
python scripts/app.py
# Open http://localhost:7860 in your browser
```

### Making Predictions (CLI)

```bash
# Predict odors for a single molecule
python scripts/predict.py --smiles "CCO" --checkpoint checkpoints/best_model.pt

# Predict for multiple molecules from a file
python scripts/predict.py --input molecules.csv --output predictions.csv
```

### Python API

```python
from olfactorynet.models import create_model
from olfactorynet.data import MolecularFeaturizer
import torch

# Load trained model
model = create_model(model_type="gcn", atom_dim=82, num_labels=61)
model.load_state_dict(torch.load("checkpoints/best_model.pt")["model_state_dict"])
model.eval()

# Featurize a molecule
featurizer = MolecularFeaturizer()
data = featurizer.featurize("c1ccccc1")  # Benzene

# Predict odors
with torch.no_grad():
    logits = model(data.x, data.edge_index, data.edge_attr)
    probs = torch.sigmoid(logits)

# Get top predictions
top_odors = probs.squeeze().topk(5)
print("Predicted odors:", top_odors)
```

---

## Project Structure

```
olfactorynet/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── data/
│   │   └── pyrfume.yaml       # Dataset settings
│   ├── model/
│   │   ├── gcn.yaml           # GCN architecture
│   │   └── gat.yaml           # GAT architecture
│   └── training/
│       └── default.yaml       # Training hyperparameters
│
├── src/olfactorynet/          # Main package
│   ├── data/
│   │   ├── featurizer.py      # SMILES → Graph conversion
│   │   ├── dataset.py         # PyG InMemoryDataset
│   │   ├── pyrfume_loader.py  # Pyrfume data fetching
│   │   └── splits.py          # Train/val/test splitting
│   │
│   ├── models/
│   │   ├── base.py            # Abstract GNN base class
│   │   ├── gcn.py             # Graph Convolutional Network
│   │   ├── gat.py             # Graph Attention Network
│   │   ├── gin.py             # Graph Isomorphism Network
│   │   ├── mpnn.py            # Message Passing Neural Network
│   │   └── heads.py           # Classification heads
│   │
│   └── training/
│       ├── trainer.py         # Training loop
│       └── metrics.py         # Evaluation metrics
│
├── scripts/
│   ├── train.py               # Training entry point
│   ├── predict.py             # Inference script
│   └── app.py                 # Gradio web interface
│
├── tests/                     # Unit tests
│   ├── test_featurizer.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_splits.py
│
├── checkpoints/               # Saved models
├── data/                      # Dataset cache
├── pyproject.toml             # Project configuration
└── README.md
```

---

## Technical Details

### Molecular Featurization

Each molecule is converted to a graph where atoms are nodes and bonds are edges.

**Atom Features (82 dimensions):**
| Feature | Dimensions | Description |
|---------|------------|-------------|
| Atom type | 44 | One-hot encoding of element |
| Degree | 11 | Number of bonded neighbors (0-10) |
| Formal charge | 6 | Ionic charge (-2 to +3) |
| Hybridization | 7 | sp, sp², sp³, etc. |
| Aromaticity | 1 | Is part of aromatic ring |
| Ring membership | 1 | Is part of any ring |
| Hydrogen count | 5 | Total attached hydrogens (0-4) |
| Chirality | 4 | Stereochemical configuration |
| Physical properties | 3 | Mass, VdW radius, covalent radius |

**Bond Features (12 dimensions):**
| Feature | Dimensions | Description |
|---------|------------|-------------|
| Bond type | 4 | Single, double, triple, aromatic |
| Conjugation | 1 | Is conjugated |
| Ring membership | 1 | Is part of ring |
| Stereochemistry | 6 | E/Z isomerism |

### Training Strategy

- **Loss Function**: Binary Cross-Entropy with Logits (optional: Focal Loss for imbalanced labels)
- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: 1e-3 with ReduceLROnPlateau scheduler
- **Early Stopping**: Patience of 20 epochs based on validation AUROC
- **Gradient Clipping**: Max norm of 1.0

### Evaluation Metrics

- **AUROC (macro/micro)**: Primary metric, threshold-independent
- **F1 Score**: Harmonic mean of precision and recall
- **Hamming Loss**: Fraction of incorrect labels

---

## Configuration

OlfactoryNet uses [Hydra](https://hydra.cc/) for configuration management. Override any parameter from the command line:

```bash
# Examples
python scripts/train.py model.hidden_dim=512
python scripts/train.py training.max_epochs=200 training.batch_size=64
python scripts/train.py data.min_label_count=100  # Fewer, more common labels
```

Or create custom config files:

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - /model: gat
  - /training: default

model:
  hidden_dim: 512
  num_layers: 4
  num_heads: 8

training:
  learning_rate: 0.0005
  batch_size: 64
```

```bash
python scripts/train.py +experiment=my_experiment
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=olfactorynet --cov-report=html

# Run specific test module
pytest tests/test_models.py -v
```

---

## Roadmap

- [x] GCN and GAT implementations
- [x] Scaffold-based data splitting
- [x] Multi-label classification
- [x] Gradio web interface
- [x] Hydra configuration
- [ ] 3D coordinate features (conformer generation)
- [ ] Attention visualization
- [ ] Model interpretability (GNNExplainer)
- [ ] ONNX export for deployment
- [ ] Docker containerization
- [ ] REST API endpoint

---

## Citation

If you use OlfactoryNet in your research, please cite:

```bibtex
@software{olfactorynet2024,
  title = {OlfactoryNet: Molecular Odor Prediction with Graph Neural Networks},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/olfactorynet}
}
```

---

## References

1. Pyrfume: A database of olfactory data ([pyrfume.org](https://pyrfume.org/))
2. Sanchez-Lengeling et al. "Machine Learning for Scent" (2019)
3. Kipf & Welling. "Semi-Supervised Classification with GCNs" (2017)
4. Veličković et al. "Graph Attention Networks" (2018)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with PyTorch Geometric</strong>
  <br>
  <a href="https://pytorch-geometric.readthedocs.io/">Documentation</a> •
  <a href="https://github.com/yourusername/olfactorynet/issues">Report Bug</a> •
  <a href="https://github.com/yourusername/olfactorynet/issues">Request Feature</a>
</p>
