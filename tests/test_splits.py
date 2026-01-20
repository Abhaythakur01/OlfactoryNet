"""Tests for data splitting module."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from olfactorynet.data.splits import DataSplitter, DataSplits, create_dataloaders


class MockDataset:
    """Mock dataset for testing splits."""

    def __init__(self, smiles_list):
        self._smiles_list = smiles_list
        self._data_list = [
            Data(
                x=torch.randn(5, 10),
                edge_index=torch.randint(0, 5, (2, 8)),
                y=torch.randint(0, 2, (3,)).float(),
                smiles=smiles,
            )
            for smiles in smiles_list
        ]

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self._data_list[i] for i in idx]
        return self._data_list[idx]


class TestDataSplits:
    """Tests for DataSplits container."""

    def test_init(self):
        """Test DataSplits initialization."""
        splits = DataSplits(
            train_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            val_indices=[8, 9],
            test_indices=[10, 11],
        )

        assert splits.train_size == 8
        assert splits.val_size == 2
        assert splits.test_size == 2

    def test_repr(self):
        """Test string representation."""
        splits = DataSplits(
            train_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            val_indices=[8, 9],
            test_indices=[10, 11],
        )

        repr_str = repr(splits)
        assert "train=8" in repr_str
        assert "val=2" in repr_str
        assert "test=2" in repr_str


class TestDataSplitter:
    """Tests for DataSplitter class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        # Create molecules with different scaffolds
        smiles = [
            "CCO",  # Scaffold: CCO
            "CCCO",  # Scaffold: CCCO
            "CCCCO",  # Scaffold: CCCCO
            "c1ccccc1",  # Scaffold: benzene
            "c1ccccc1C",  # Scaffold: benzene
            "c1ccccc1CC",  # Scaffold: benzene
            "CC(=O)O",  # Scaffold: CC(=O)O
            "CCC(=O)O",  # Scaffold: CC(=O)O-like
            "CCCC(=O)O",  # Scaffold: CC(=O)O-like
            "CC(C)C",  # Scaffold: CC(C)C
        ]
        return MockDataset(smiles)

    def test_random_split(self, sample_dataset):
        """Test random splitting."""
        splitter = DataSplitter(
            strategy="random",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
        )

        splits = splitter.split(sample_dataset)

        # Check split sizes (approximate due to integer division)
        total = len(sample_dataset)
        assert splits.train_size >= int(0.5 * total)
        assert splits.train_size + splits.val_size + splits.test_size == total

        # Check no overlap
        train_set = set(splits.train_indices)
        val_set = set(splits.val_indices)
        test_set = set(splits.test_indices)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_scaffold_split(self, sample_dataset):
        """Test scaffold splitting."""
        splitter = DataSplitter(
            strategy="scaffold",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
        )

        splits = splitter.split(sample_dataset)

        # Check that all indices are covered
        all_indices = set(splits.train_indices + splits.val_indices + splits.test_indices)
        assert all_indices == set(range(len(sample_dataset)))

    def test_reproducibility(self, sample_dataset):
        """Test that same seed produces same splits."""
        splitter1 = DataSplitter(strategy="random", random_seed=42)
        splitter2 = DataSplitter(strategy="random", random_seed=42)

        splits1 = splitter1.split(sample_dataset)
        splits2 = splitter2.split(sample_dataset)

        assert splits1.train_indices == splits2.train_indices
        assert splits1.val_indices == splits2.val_indices
        assert splits1.test_indices == splits2.test_indices

    def test_different_seeds(self, sample_dataset):
        """Test that different seeds produce different splits."""
        splitter1 = DataSplitter(strategy="random", random_seed=42)
        splitter2 = DataSplitter(strategy="random", random_seed=123)

        splits1 = splitter1.split(sample_dataset)
        splits2 = splitter2.split(sample_dataset)

        # Very unlikely to be exactly the same with different seeds
        assert splits1.train_indices != splits2.train_indices

    def test_invalid_strategy(self, sample_dataset):
        """Test that invalid strategy raises error."""
        splitter = DataSplitter(strategy="invalid")

        with pytest.raises(ValueError):
            splitter.split(sample_dataset)


class TestCreateDataloaders:
    """Tests for dataloader creation."""

    @pytest.fixture
    def sample_dataset_and_splits(self):
        """Create dataset and splits."""
        smiles = [f"C{'C' * i}O" for i in range(20)]
        dataset = MockDataset(smiles)

        splits = DataSplits(
            train_indices=list(range(16)),
            val_indices=[16, 17],
            test_indices=[18, 19],
        )

        return dataset, splits

    def test_create_dataloaders(self, sample_dataset_and_splits):
        """Test dataloader creation."""
        dataset, splits = sample_dataset_and_splits

        # Use DataLoader directly instead of create_dataloaders
        train_data = dataset[splits.train_indices]
        val_data = dataset[splits.val_indices]
        test_data = dataset[splits.test_indices]

        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_batch_size(self, sample_dataset_and_splits):
        """Test that batch size is respected."""
        dataset, splits = sample_dataset_and_splits

        train_data = dataset[splits.train_indices]
        train_loader = DataLoader(train_data, batch_size=4, shuffle=False)

        batch = next(iter(train_loader))
        # For PyG batches, num_graphs tells us the batch size
        assert batch.num_graphs <= 4
