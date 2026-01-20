"""Tests for GNN models."""

import pytest
import torch
from torch_geometric.data import Data, Batch

from olfactorynet.models import GCNModel, GATModel, OdorPredictor
from olfactorynet.models.heads import create_model, MultiLabelHead
from olfactorynet.data.featurizer import MolecularFeaturizer


class TestGCNModel:
    """Tests for GCN model."""

    @pytest.fixture
    def gcn_model(self):
        """Create GCN model for testing."""
        return GCNModel(
            atom_dim=82,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
        )

    def test_init(self, gcn_model):
        """Test model initialization."""
        assert gcn_model.atom_dim == 82
        assert gcn_model.hidden_dim == 64
        assert gcn_model.num_layers == 2
        assert gcn_model.output_dim == 64

    def test_forward(self, gcn_model, sample_smiles):
        """Test forward pass."""
        featurizer = MolecularFeaturizer()
        data = featurizer.featurize(sample_smiles[0])

        output = gcn_model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )

        # Output should be (1, hidden_dim) for single graph
        assert output.shape == (1, 64)

    def test_forward_batch(self, gcn_model, sample_smiles):
        """Test forward pass with batch."""
        featurizer = MolecularFeaturizer()
        data_list = [featurizer.featurize(s) for s in sample_smiles[:3]]
        batch = Batch.from_data_list(data_list)

        output = gcn_model.forward_batch(batch)

        # Output should be (batch_size, hidden_dim)
        assert output.shape == (3, 64)


class TestGATModel:
    """Tests for GAT model."""

    @pytest.fixture
    def gat_model(self):
        """Create GAT model for testing."""
        return GATModel(
            atom_dim=82,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            num_heads=4,
            edge_dim=12,
        )

    def test_init(self, gat_model):
        """Test model initialization."""
        assert gat_model.atom_dim == 82
        assert gat_model.hidden_dim == 64
        assert gat_model.num_layers == 2
        assert gat_model._num_heads == 4
        assert gat_model._edge_dim == 12

    def test_forward_with_edge_features(self, gat_model, sample_smiles):
        """Test forward pass with edge features."""
        featurizer = MolecularFeaturizer()
        data = featurizer.featurize(sample_smiles[0])

        output = gat_model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )

        assert output.shape == (1, 64)

    def test_forward_without_edge_features(self, sample_smiles):
        """Test forward pass without edge features."""
        model = GATModel(
            atom_dim=82,
            hidden_dim=64,
            num_layers=2,
            edge_dim=None,  # No edge features
        )

        featurizer = MolecularFeaturizer()
        data = featurizer.featurize(sample_smiles[0])

        output = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=None,
        )

        assert output.shape == (1, 64)


class TestMultiLabelHead:
    """Tests for classification head."""

    def test_init(self):
        """Test head initialization."""
        head = MultiLabelHead(
            input_dim=64,
            num_labels=10,
            hidden_dim=32,
        )

        assert head.input_dim == 64
        assert head.num_labels == 10
        assert head.hidden_dim == 32

    def test_forward(self):
        """Test head forward pass."""
        head = MultiLabelHead(
            input_dim=64,
            num_labels=10,
        )

        x = torch.randn(4, 64)  # Batch of 4
        output = head(x)

        assert output.shape == (4, 10)


class TestOdorPredictor:
    """Tests for complete odor prediction model."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for testing."""
        backbone = GCNModel(
            atom_dim=82,
            hidden_dim=64,
            num_layers=2,
        )
        return OdorPredictor(
            backbone=backbone,
            num_labels=10,
            head_hidden_dim=32,
        )

    def test_forward(self, predictor, sample_smiles):
        """Test forward pass."""
        featurizer = MolecularFeaturizer()
        data = featurizer.featurize(sample_smiles[0])

        output = predictor(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )

        assert output.shape == (1, 10)

    def test_forward_batch(self, predictor, sample_smiles):
        """Test forward pass with batch."""
        featurizer = MolecularFeaturizer()
        data_list = [featurizer.featurize(s) for s in sample_smiles[:3]]
        batch = Batch.from_data_list(data_list)

        output = predictor.forward_batch(batch)

        assert output.shape == (3, 10)

    def test_predict_proba(self, predictor, sample_smiles):
        """Test probability predictions."""
        featurizer = MolecularFeaturizer()
        data_list = [featurizer.featurize(s) for s in sample_smiles[:2]]
        batch = Batch.from_data_list(data_list)

        probs = predictor.predict_proba(batch)

        assert probs.shape == (2, 10)
        # Probabilities should be between 0 and 1
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict(self, predictor, sample_smiles):
        """Test binary predictions."""
        featurizer = MolecularFeaturizer()
        data_list = [featurizer.featurize(s) for s in sample_smiles[:2]]
        batch = Batch.from_data_list(data_list)

        preds = predictor.predict(batch, threshold=0.5)

        assert preds.shape == (2, 10)
        # Predictions should be binary
        assert ((preds == 0) | (preds == 1)).all()


class TestCreateModel:
    """Tests for model factory function."""

    def test_create_gcn(self):
        """Test creating GCN model."""
        model = create_model(
            model_type="gcn",
            atom_dim=82,
            num_labels=10,
            hidden_dim=64,
        )

        assert isinstance(model, OdorPredictor)
        assert isinstance(model.backbone, GCNModel)
        assert model.num_labels == 10

    def test_create_gat(self):
        """Test creating GAT model."""
        model = create_model(
            model_type="gat",
            atom_dim=82,
            num_labels=10,
            hidden_dim=64,
            edge_dim=12,
            num_heads=4,
        )

        assert isinstance(model, OdorPredictor)
        assert isinstance(model.backbone, GATModel)

    def test_create_invalid_type(self):
        """Test creating model with invalid type."""
        with pytest.raises(ValueError):
            create_model(
                model_type="invalid",
                atom_dim=82,
                num_labels=10,
            )
