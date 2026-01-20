"""Tests for molecular featurizer."""

import pytest
import torch
import numpy as np

from olfactorynet.data.featurizer import MolecularFeaturizer


class TestMolecularFeaturizer:
    """Tests for MolecularFeaturizer class."""

    def test_init_default(self):
        """Test default initialization."""
        featurizer = MolecularFeaturizer()
        assert featurizer.atom_dim > 0
        assert featurizer.bond_dim > 0
        assert not featurizer.add_hydrogens
        assert not featurizer.use_3d_coords

    def test_feature_dimensions(self):
        """Test feature dimension calculations."""
        featurizer = MolecularFeaturizer()
        # Based on implementation: ~82 atom features, ~12 bond features
        assert featurizer.atom_dim == 82
        assert featurizer.bond_dim == 12

    def test_featurize_simple_molecule(self, sample_smiles):
        """Test featurization of simple molecules."""
        featurizer = MolecularFeaturizer()

        for smiles in sample_smiles:
            data = featurizer.featurize(smiles)

            assert data is not None
            assert data.x is not None
            assert data.edge_index is not None
            assert data.x.shape[1] == featurizer.atom_dim
            assert data.smiles == smiles

    def test_featurize_ethanol(self):
        """Test featurization of ethanol (CCO)."""
        featurizer = MolecularFeaturizer()
        data = featurizer.featurize("CCO")

        # Ethanol has 3 heavy atoms (2 C, 1 O)
        assert data.x.shape[0] == 3

        # Ethanol has 2 bonds, so 4 edges (bidirectional)
        assert data.edge_index.shape[1] == 4
        assert data.edge_attr.shape[0] == 4
        assert data.edge_attr.shape[1] == featurizer.bond_dim

    def test_featurize_with_labels(self, sample_smiles, sample_labels):
        """Test featurization with labels."""
        featurizer = MolecularFeaturizer()

        for smiles, labels in zip(sample_smiles, sample_labels):
            data = featurizer.featurize(smiles, y=labels)

            assert data is not None
            assert data.y is not None
            # y has shape (1, num_labels) for proper batching
            assert data.y.shape == (1, len(labels))
            assert torch.equal(data.y.squeeze(0), torch.tensor(labels, dtype=torch.float))

    def test_featurize_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        featurizer = MolecularFeaturizer()

        # Invalid SMILES
        data = featurizer.featurize("not_a_valid_smiles")
        assert data is None

        # Empty SMILES
        data = featurizer.featurize("")
        assert data is None

    def test_featurize_single_atom(self):
        """Test featurization of single atom molecule."""
        featurizer = MolecularFeaturizer()

        # Methane (represented as just C in simplified form)
        data = featurizer.featurize("[CH4]")

        if data is not None:
            # Single atom has no bonds
            assert data.edge_index.shape[1] == 0
            assert data.edge_attr.shape[0] == 0

    def test_featurize_aromatic(self):
        """Test featurization of aromatic molecule."""
        featurizer = MolecularFeaturizer()

        # Benzene
        data = featurizer.featurize("c1ccccc1")

        assert data is not None
        assert data.x.shape[0] == 6  # 6 carbons

        # Benzene has 6 bonds, so 12 edges
        assert data.edge_index.shape[1] == 12

    def test_featurize_with_hydrogens(self):
        """Test featurization with explicit hydrogens."""
        featurizer_no_h = MolecularFeaturizer(add_hydrogens=False)
        featurizer_h = MolecularFeaturizer(add_hydrogens=True)

        data_no_h = featurizer_no_h.featurize("CCO")
        data_h = featurizer_h.featurize("CCO")

        # With hydrogens should have more atoms
        assert data_h.x.shape[0] > data_no_h.x.shape[0]

    def test_feature_dims_property(self):
        """Test feature_dims property."""
        featurizer = MolecularFeaturizer()
        dims = featurizer.feature_dims

        assert dims.atom_dim == featurizer.atom_dim
        assert dims.bond_dim == featurizer.bond_dim

    def test_featurize_batch_consistency(self, sample_smiles):
        """Test that featurization is consistent across calls."""
        featurizer = MolecularFeaturizer()

        for smiles in sample_smiles:
            data1 = featurizer.featurize(smiles)
            data2 = featurizer.featurize(smiles)

            assert torch.equal(data1.x, data2.x)
            assert torch.equal(data1.edge_index, data2.edge_index)
            assert torch.equal(data1.edge_attr, data2.edge_attr)
