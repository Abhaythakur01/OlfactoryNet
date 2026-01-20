"""Graph Attention Network (GAT) model."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from .base import BaseGNN


class GATModel(BaseGNN):
    """Graph Attention Network for molecular property prediction.

    Uses GATConv layers with multi-head attention. Supports edge features
    through the edge_dim parameter.
    """

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: Literal["mean", "max", "add"] = "mean",
        batch_norm: bool = True,
        num_heads: int = 4,
        edge_dim: Optional[int] = None,
        concat_heads: bool = False,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
    ):
        """Initialize GAT model.

        Args:
            atom_dim: Dimension of input atom features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
            pooling: Global pooling strategy
            batch_norm: Whether to use batch normalization
            num_heads: Number of attention heads
            edge_dim: Dimension of edge features (None to ignore)
            concat_heads: Whether to concatenate attention heads (vs. average)
            negative_slope: Negative slope for LeakyReLU in attention
            add_self_loops: Whether to add self-loops
        """
        # Store GAT-specific parameters before calling super().__init__
        self._num_heads = num_heads
        self._edge_dim = edge_dim
        self._concat_heads = concat_heads
        self._negative_slope = negative_slope
        self._add_self_loops = add_self_loops

        # Adjust hidden_dim for head concatenation
        if concat_heads:
            # Ensure hidden_dim is divisible by num_heads when concatenating
            self._head_dim = hidden_dim // num_heads
            self._effective_hidden = self._head_dim * num_heads
        else:
            self._head_dim = hidden_dim
            self._effective_hidden = hidden_dim

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

        # Edge feature projection if using edge features
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, edge_dim)
        else:
            self.edge_proj = None

    def _build_conv_layers(self) -> nn.ModuleList:
        """Build GAT convolutional layers."""
        convs = nn.ModuleList()

        for i in range(self.num_layers):
            # Input dimension
            in_channels = self.hidden_dim

            # Output dimension per head
            if self._concat_heads:
                out_channels = self._head_dim
            else:
                out_channels = self.hidden_dim

            conv = GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=self._num_heads,
                concat=self._concat_heads,
                negative_slope=self._negative_slope,
                dropout=self.dropout_rate,
                add_self_loops=self._add_self_loops,
                edge_dim=self._edge_dim,
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
        """Forward pass through a GAT layer with edge features."""
        # Project edge features if available
        if edge_attr is not None and self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)

        return self.convs[layer_idx](x, edge_index, edge_attr=edge_attr)

    def __repr__(self) -> str:
        return (
            f"GATModel("
            f"atom_dim={self.atom_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self._num_heads}, "
            f"edge_dim={self._edge_dim}, "
            f"dropout={self.dropout_rate}, "
            f"pooling={self.pooling_type})"
        )
