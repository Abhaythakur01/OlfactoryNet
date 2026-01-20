"""Message Passing Neural Network (MPNN) model."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, Set2Set

from .base import BaseGNN


class MPNNModel(BaseGNN):
    """Message Passing Neural Network for molecular property prediction.

    MPNN uses edge networks to compute message functions, making it
    highly expressive for molecular graphs with rich bond features.
    Optionally uses Set2Set pooling for better graph-level readout.
    """

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: Literal["mean", "max", "add", "set2set"] = "mean",
        batch_norm: bool = True,
        edge_dim: int = 12,
        edge_hidden_dim: int = 64,
        set2set_steps: int = 3,
    ):
        """Initialize MPNN model.

        Args:
            atom_dim: Dimension of input atom features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout probability
            pooling: Global pooling strategy ('set2set' recommended for MPNN)
            batch_norm: Whether to use batch normalization
            edge_dim: Edge feature dimension
            edge_hidden_dim: Hidden dimension for edge network
            set2set_steps: Number of Set2Set iterations
        """
        self._edge_dim = edge_dim
        self._edge_hidden_dim = edge_hidden_dim
        self._set2set_steps = set2set_steps
        self._use_set2set = pooling == "set2set"

        # Override pooling for parent class if using set2set
        parent_pooling = "mean" if self._use_set2set else pooling

        super().__init__(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=parent_pooling,
            batch_norm=batch_norm,
        )

        # Build convolutional layers
        self.convs = self._build_conv_layers()
        self.edge_networks = self._build_edge_networks()

        # Set2Set pooling
        if self._use_set2set:
            self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)
            # Set2Set outputs 2 * hidden_dim
            self._output_dim = 2 * hidden_dim
        else:
            self._output_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        """Output dimension of the MPNN encoder."""
        return self._output_dim

    def _build_edge_network(self) -> nn.Sequential:
        """Build edge network for NNConv."""
        return nn.Sequential(
            nn.Linear(self._edge_dim, self._edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(self._edge_hidden_dim, self.hidden_dim * self.hidden_dim),
        )

    def _build_edge_networks(self) -> nn.ModuleList:
        """Build edge networks for all layers."""
        return nn.ModuleList([
            self._build_edge_network() for _ in range(self.num_layers)
        ])

    def _build_conv_layers(self) -> nn.ModuleList:
        """Build MPNN convolutional layers."""
        convs = nn.ModuleList()

        for i in range(self.num_layers):
            # NNConv uses an edge network to compute message weights
            # The edge network is built separately and passed during forward
            conv = NNConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                nn=self._build_edge_network(),
                aggr="mean",
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
        """Forward pass through an MPNN layer."""
        if edge_attr is None:
            raise ValueError("MPNN requires edge features")

        return self.convs[layer_idx](x, edge_index, edge_attr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the MPNN.

        Args:
            x: Node features (num_nodes, atom_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim) - REQUIRED
            batch: Batch assignment for nodes

        Returns:
            Graph-level embeddings (batch_size, output_dim)
        """
        if edge_attr is None:
            raise ValueError("MPNN requires edge features (edge_attr)")

        # Initial node embedding
        x = self.node_embedding(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # GNN layers
        for i in range(self.num_layers):
            x_new = self._conv_forward(x, edge_index, edge_attr, i)

            if self.batch_norms is not None:
                x_new = self.batch_norms[i](x_new)

            x_new = torch.relu(x_new)
            x_new = self.dropout(x_new)

            # Residual connection
            x = x + x_new

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self._use_set2set:
            x = self.set2set(x, batch)
        else:
            x = self.pool(x, batch)

        return x

    def __repr__(self) -> str:
        return (
            f"MPNNModel("
            f"atom_dim={self.atom_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"edge_dim={self._edge_dim}, "
            f"pooling={'set2set' if self._use_set2set else self.pooling_type}, "
            f"dropout={self.dropout_rate})"
        )
