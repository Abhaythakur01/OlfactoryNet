"""Graph Convolutional Network (GCN) model."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from .base import BaseGNN


class GCNModel(BaseGNN):
    """Graph Convolutional Network for molecular property prediction.

    Uses GCNConv layers for message passing. Note that GCN does not use
    edge features - use GAT for edge feature support.
    """

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: Literal["mean", "max", "add"] = "mean",
        batch_norm: bool = True,
        improved: bool = True,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        """Initialize GCN model.

        Args:
            atom_dim: Dimension of input atom features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
            pooling: Global pooling strategy
            batch_norm: Whether to use batch normalization
            improved: Whether to use improved GCN (adds self-loops differently)
            add_self_loops: Whether to add self-loops
            normalize: Whether to use symmetric normalization
        """
        # Store GCN-specific parameters before calling super().__init__
        self._improved = improved
        self._add_self_loops = add_self_loops
        self._normalize = normalize

        super().__init__(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            batch_norm=batch_norm,
        )

        # Build convolutional layers
        self.convs = self._build_conv_layers()

    def _build_conv_layers(self) -> nn.ModuleList:
        """Build GCN convolutional layers."""
        convs = nn.ModuleList()

        for _ in range(self.num_layers):
            conv = GCNConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                improved=self._improved,
                add_self_loops=self._add_self_loops,
                normalize=self._normalize,
            )
            convs.append(conv)

        return convs

    def _conv_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Forward pass through a GCN layer.

        Note: GCN ignores edge_attr by design.
        """
        return self.convs[layer_idx](x, edge_index)
