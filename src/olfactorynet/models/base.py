"""Abstract base class for GNN models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


class BaseGNN(nn.Module, ABC):
    """Abstract base class for Graph Neural Network models.

    Provides common functionality for GNN-based molecular encoders:
    - Node embedding layer
    - Batch normalization and dropout
    - Global pooling
    """

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: Literal["mean", "max", "add"] = "mean",
        batch_norm: bool = True,
    ):
        """Initialize base GNN.

        Args:
            atom_dim: Dimension of input atom features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
            pooling: Global pooling strategy
            batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling_type = pooling
        self.use_batch_norm = batch_norm

        # Node embedding layer
        self.node_embedding = nn.Linear(atom_dim, hidden_dim)

        # Batch normalization layers
        if batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.batch_norms = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Pooling function
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

    @property
    def output_dim(self) -> int:
        """Output dimension of the GNN encoder."""
        return self.hidden_dim

    @abstractmethod
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build GNN convolutional layers.

        Must be implemented by subclasses.

        Returns:
            ModuleList of GNN layers
        """
        pass

    @abstractmethod
    def _conv_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Forward pass through a single GNN layer.

        Args:
            x: Node features (num_nodes, hidden_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)
            layer_idx: Index of the layer

        Returns:
            Updated node features
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the GNN.

        Args:
            x: Node features (num_nodes, atom_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)
            batch: Batch assignment for nodes

        Returns:
            Graph-level embeddings (batch_size, hidden_dim)
        """
        # Initial node embedding
        x = self.node_embedding(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # GNN layers
        for i in range(self.num_layers):
            # Message passing
            x_new = self._conv_forward(x, edge_index, edge_attr, i)

            # Batch normalization
            if self.batch_norms is not None:
                x_new = self.batch_norms[i](x_new)

            # Activation and dropout
            x_new = torch.relu(x_new)
            x_new = self.dropout(x_new)

            # Residual connection (if dimensions match)
            x = x + x_new

        # Global pooling
        if batch is None:
            # Single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.pool(x, batch)

        return x

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        """Forward pass with a PyG Batch object.

        Args:
            batch: PyG Batch object

        Returns:
            Graph-level embeddings
        """
        return self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
            batch=batch.batch,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"atom_dim={self.atom_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout_rate}, "
            f"pooling={self.pooling_type})"
        )
