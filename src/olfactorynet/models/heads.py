"""Classification heads for multi-label prediction."""

from __future__ import annotations

from typing import Literal, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .base import BaseGNN
from .gat import GATModel
from .gcn import GCNModel
from .gin import GINModel
from .mpnn import MPNNModel


class MultiLabelHead(nn.Module):
    """Multi-label classification head with MLP.

    Takes graph-level embeddings and outputs logits for each label.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_hidden_layers: int = 1,
    ):
        """Initialize classification head.

        Args:
            input_dim: Input dimension (from GNN backbone)
            num_labels: Number of output labels
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            num_hidden_layers: Number of hidden layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, num_labels))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Graph embeddings (batch_size, input_dim)

        Returns:
            Logits (batch_size, num_labels)
        """
        return self.mlp(x)


class OdorPredictor(nn.Module):
    """Complete model combining GNN backbone with classification head.

    Outputs logits - apply sigmoid for probabilities.
    """

    def __init__(
        self,
        backbone: BaseGNN,
        num_labels: int,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
        head_num_layers: int = 1,
    ):
        """Initialize odor predictor.

        Args:
            backbone: GNN backbone (GCN, GAT, GIN, MPNN, etc.)
            num_labels: Number of odor labels
            head_hidden_dim: Hidden dimension for classification head
            head_dropout: Dropout for classification head
            head_num_layers: Number of hidden layers in head
        """
        super().__init__()

        self.backbone = backbone
        self.head = MultiLabelHead(
            input_dim=backbone.output_dim,
            num_labels=num_labels,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            num_hidden_layers=head_num_layers,
        )

    @property
    def num_labels(self) -> int:
        return self.head.num_labels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch assignment

        Returns:
            Logits (batch_size, num_labels)
        """
        # Get graph embeddings from backbone
        graph_embed = self.backbone(x, edge_index, edge_attr, batch)

        # Get predictions from head
        logits = self.head(graph_embed)

        return logits

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        """Forward pass with PyG Batch object.

        Args:
            batch: PyG Batch

        Returns:
            Logits (batch_size, num_labels)
        """
        return self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
            batch=batch.batch,
        )

    def predict_proba(self, batch: Batch) -> torch.Tensor:
        """Get probability predictions.

        Args:
            batch: PyG Batch

        Returns:
            Probabilities (batch_size, num_labels)
        """
        logits = self.forward_batch(batch)
        return torch.sigmoid(logits)

    def predict(self, batch: Batch, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions.

        Args:
            batch: PyG Batch
            threshold: Probability threshold for positive prediction

        Returns:
            Binary predictions (batch_size, num_labels)
        """
        probs = self.predict_proba(batch)
        return (probs >= threshold).float()


def create_model(
    model_type: Literal["gcn", "gat", "gin", "mpnn"],
    atom_dim: int,
    num_labels: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.2,
    pooling: Literal["mean", "max", "add", "set2set"] = "mean",
    edge_dim: Optional[int] = None,
    num_heads: int = 4,
    train_eps: bool = True,
    mlp_layers: int = 2,
    edge_hidden_dim: int = 64,
    set2set_steps: int = 3,
    head_hidden_dim: int = 256,
    head_dropout: float = 0.2,
    head_num_layers: int = 1,
) -> OdorPredictor:
    """Factory function to create odor prediction models.

    Args:
        model_type: Type of GNN backbone ('gcn', 'gat', 'gin', 'mpnn')
        atom_dim: Dimension of atom features
        num_labels: Number of odor labels
        hidden_dim: Hidden dimension for GNN
        num_layers: Number of GNN layers
        dropout: Dropout probability
        pooling: Global pooling strategy
        edge_dim: Edge feature dimension
        num_heads: Number of attention heads (GAT only)
        train_eps: Whether to train epsilon (GIN only)
        mlp_layers: Number of MLP layers (GIN only)
        edge_hidden_dim: Edge network hidden dim (MPNN only)
        set2set_steps: Set2Set iterations (MPNN only)
        head_hidden_dim: Hidden dimension for classification head
        head_dropout: Dropout for classification head
        head_num_layers: Number of hidden layers in head

    Returns:
        OdorPredictor model
    """
    if model_type == "gcn":
        backbone = GCNModel(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling if pooling != "set2set" else "mean",
        )
    elif model_type == "gat":
        backbone = GATModel(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling if pooling != "set2set" else "mean",
            num_heads=num_heads,
            edge_dim=edge_dim,
        )
    elif model_type == "gin":
        backbone = GINModel(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling if pooling != "set2set" else "mean",
            edge_dim=edge_dim,
            train_eps=train_eps,
            mlp_layers=mlp_layers,
        )
    elif model_type == "mpnn":
        if edge_dim is None:
            raise ValueError("MPNN requires edge_dim to be specified")
        backbone = MPNNModel(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            edge_dim=edge_dim,
            edge_hidden_dim=edge_hidden_dim,
            set2set_steps=set2set_steps,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return OdorPredictor(
        backbone=backbone,
        num_labels=num_labels,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        head_num_layers=head_num_layers,
    )
