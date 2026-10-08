"""Neural architectures used by the final TomTom traffic experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    """Aggregate directed neighbors, then project the feature vector."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("ij,btjf->btif", adjacency, x)
        return F.relu(self.projection(aggregated))


class A3TGCN(nn.Module):
    """Graph convolution, GRU temporal encoding, and temporal attention."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.graph_convolution = GraphConvolution(input_size, hidden_size)
        self.temporal_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.temporal_attention = nn.Linear(hidden_size, 1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, node_count, _ = x.shape
        spatial = self.graph_convolution(x, adjacency)
        sequence = spatial.permute(0, 2, 1, 3).reshape(
            batch_size * node_count, time_steps, self.hidden_size
        )
        temporal, _ = self.temporal_encoder(sequence)
        attention = torch.softmax(
            self.temporal_attention(temporal).squeeze(-1), dim=1
        )
        context = torch.sum(temporal * attention.unsqueeze(-1), dim=1)
        context = F.dropout(context, p=self.dropout, training=self.training)
        prediction = self.output(context).reshape(batch_size, node_count)
        attention = attention.reshape(batch_size, node_count, time_steps)
        return prediction, attention


class TGCN(nn.Module):
    """Graph convolution and GRU temporal encoding without attention."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.graph_convolution = GraphConvolution(input_size, hidden_size)
        self.temporal_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        batch_size, time_steps, node_count, _ = x.shape
        spatial = self.graph_convolution(x, adjacency)
        sequence = spatial.permute(0, 2, 1, 3).reshape(
            batch_size * node_count, time_steps, self.hidden_size
        )
        temporal, _ = self.temporal_encoder(sequence)
        context = F.dropout(temporal[:, -1], p=self.dropout, training=self.training)
        prediction = self.output(context).reshape(batch_size, node_count)
        return prediction, None


class SharedNodeLSTM(nn.Module):
    """Temporal baseline that processes every road segment independently."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.temporal_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        del adjacency
        batch_size, time_steps, node_count, feature_count = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(
            batch_size * node_count, time_steps, feature_count
        )
        temporal, _ = self.temporal_encoder(sequence)
        context = F.dropout(temporal[:, -1], p=self.dropout, training=self.training)
        prediction = self.output(context).reshape(batch_size, node_count)
        return prediction, None


def build_model(
    model_name: str, input_size: int, hidden_size: int, dropout: float
) -> nn.Module:
    if model_name == "a3tgcn":
        return A3TGCN(input_size, hidden_size, dropout)
    if model_name == "tgcn":
        return TGCN(input_size, hidden_size, dropout)
    if model_name == "lstm":
        return SharedNodeLSTM(input_size, hidden_size, dropout)
    raise ValueError(f"Unknown neural model: {model_name}")
