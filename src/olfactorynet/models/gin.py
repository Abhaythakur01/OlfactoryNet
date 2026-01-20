"""Graph Isomorphism Network (GIN) model."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv

from .base import BaseGNN


class GINModel(BaseGNN):
    """Graph Isomorphism Network for molecular property prediction.

    GIN is provably as powerful as the Weisfeiler-Lehman graph isomorphism test.
    Uses MLPs for message transformation, making it more expressive than GCN.
    """

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: Literal["mean", "max", "add"] = "mean",
        batch_norm: bool = True,
        edge_dim: Optional[int] = None,
        train_eps: bool = True,
        mlp_layers: int = 2,
    ):
        """Initialize GIN model.

        Args:
            atom_dim: Dimension of input atom features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
            pooling: Global pooling strategy
            batch_norm: Whether to use batch normalization
            edge_dim: Edge feature dimension (uses GINE if provided)
            train_eps: Whether to train epsilon parameter
            mlp_layers: Number of layers in the MLP
        """
        self._edge_dim = edge_dim
        self._train_eps = train_eps
        self._mlp_layers = mlp_layers

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

    def _build_mlp(self, in_dim: int, out_dim: int) -> nn.Sequential:
        """Build MLP for GIN layer."""
        layers = []

        for i in range(self._mlp_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                layers.append(nn.Linear(out_dim, out_dim))

            if i < self._mlp_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _build_conv_layers(self) -> nn.ModuleList:
        """Build GIN convolutional layers."""
        convs = nn.ModuleList()

        for _ in range(self.num_layers):
            mlp = self._build_mlp(self.hidden_dim, self.hidden_dim)

            if self._edge_dim is not None:
                # Use GINE (GIN with edge features)
                conv = GINEConv(mlp, train_eps=self._train_eps, edge_dim=self._edge_dim)
            else:
                conv = GINConv(mlp, train_eps=self._train_eps)

            convs.append(conv)

        return convs

    def _conv_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Forward pass through a GIN layer."""
        if self._edge_dim is not None and edge_attr is not None:
            return self.convs[layer_idx](x, edge_index, edge_attr)
        else:
            return self.convs[layer_idx](x, edge_index)

    def __repr__(self) -> str:
        return (
            f"GINModel("
            f"atom_dim={self.atom_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"edge_dim={self._edge_dim}, "
            f"dropout={self.dropout_rate}, "
            f"pooling={self.pooling_type})"
        )
