from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .activations import make_activation
from .config import AblationConfig, ModelConfig


class DenseGraphAttention(nn.Module):

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, steps, stocks, hidden = features.shape
        shape = (batch, steps, stocks, self.heads, self.head_dim)
        query = self.query(features).reshape(shape).permute(0, 1, 3, 2, 4)
        key = self.key(features).reshape(shape).permute(0, 1, 3, 2, 4)
        value = self.value(features).reshape(shape).permute(0, 1, 3, 2, 4)
        logits = torch.einsum("bthid,bthjd->bthij", query, key) / (self.head_dim**0.5)
        graph = adjacency.unsqueeze(2)
        logits = logits + torch.log(graph.clamp_min(1e-12))
        logits = logits.masked_fill(graph <= 0, torch.finfo(logits.dtype).min)
        attention = self.dropout(F.softmax(logits, dim=-1))
        output = torch.einsum("bthij,bthjd->bthid", attention, value)
        output = output.permute(0, 1, 3, 2, 4).reshape(batch, steps, stocks, hidden)
        return self.output(output)


class SpatialGraphEncoder(nn.Module):
    def __init__(self, config: ModelConfig, ablation: AblationConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseGraphAttention(config.hidden_dim, config.graph_heads, config.dropout)
                for _ in range(config.graph_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(config.graph_layers)]
        )
        self.activation = make_activation(ablation.use_sgfrelu)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        output = features
        for layer, norm in zip(self.layers, self.norms):
            update = self.dropout(self.activation(layer(output, adjacency)))
            output = norm(output + update)
        return output
