"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)CC1=CC=C(C=C1)C(C)C",  # Isobutylbenzene
        "CCCCCCCC",  # Octane
        "O=C1CCCCC1",  # Cyclohexanone
    ]


@pytest.fixture
def sample_labels():
    """Sample multi-hot labels."""
    return np.array([
        [1, 0, 1, 0, 0],  # sweet, fruity
        [0, 1, 0, 1, 0],  # sour, pungent
        [0, 0, 0, 0, 1],  # aromatic
        [1, 0, 0, 0, 1],  # sweet, aromatic
        [0, 0, 0, 1, 0],  # pungent
        [1, 1, 1, 0, 0],  # sweet, sour, fruity
    ], dtype=np.float32)


@pytest.fixture
def label_names():
    """Sample label names."""
    return ["sweet", "sour", "fruity", "pungent", "aromatic"]


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")
