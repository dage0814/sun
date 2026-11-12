from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .activations import make_activation
from .config import AblationConfig, ModelConfig
from .revin import RevIN
from .spatial import SpatialGraphEncoder
from .temporal import TemporalConvNet, TransformerTemporalBranch
from .topology import DualTopology
from .history import PointInTimePrior


@dataclass
class ModelOutput:
    logits: torch.Tensor
    adjacency: torch.Tensor
    branches: Dict[str, torch.Tensor]


class AdaptiveSpatioTemporalGraphTransformer(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        ablation: Optional[AblationConfig] = None,
        prior_adjacency: Optional[torch.Tensor] = None,
        prior_graph: Optional[PointInTimePrior] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ablation = ablation or AblationConfig()
        self.revin = RevIN(config.num_input_features)
        self.input_projection = nn.Linear(config.num_input_features, config.hidden_dim)
        self.topology = DualTopology(config, self.ablation, prior_adjacency, prior_graph)
        self.spatial = SpatialGraphEncoder(config, self.ablation)
        self.transformer = TransformerTemporalBranch(config, self.ablation)
        self.tcn = TemporalConvNet(config, self.ablation)

        branch_count = int(self.ablation.use_transformer) + int(self.ablation.use_tcn)
        branch_count = max(branch_count, 1)
        self.fusion = nn.Sequential(
            nn.Linear(branch_count * config.hidden_dim, config.hidden_dim),
            make_activation(self.ablation.use_sgfrelu),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            make_activation(self.ablation.use_sgfrelu),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, features: torch.Tensor, window_cutoffs: Optional[torch.Tensor] = None) -> ModelOutput:
        if features.ndim != 4:
            raise ValueError("features must use shape [B,T,N,F]")
        if features.shape[2] != self.config.num_stocks:
            raise ValueError(
                f"Expected {self.config.num_stocks} stocks, got {features.shape[2]}"
            )
        if features.shape[-1] != self.config.num_input_features:
            raise ValueError("Unexpected backbone feature width")
        if features.shape[1] < self.config.pre_len:
            raise ValueError("pre_len cannot exceed the input sequence length")

        normalized = self.revin(features, "norm") if self.ablation.use_revin else features
        embedded = self.input_projection(normalized)
        adjacency = self.topology(embedded, window_cutoffs)
        spatial = self.spatial(embedded, adjacency) if self.ablation.use_gat else embedded

        branches: Dict[str, torch.Tensor] = {}
        if self.ablation.use_transformer:
            branches["transformer"] = self.transformer(spatial, self.config.pre_len)
        if self.ablation.use_tcn:
            branches["tcn"] = self.tcn(normalized, self.config.pre_len)
        if not branches:
            branches["spatial"] = spatial[:, -self.config.pre_len :]
        fused = self.fusion(torch.cat(list(branches.values()), dim=-1))
        return ModelOutput(self.classifier(fused), adjacency, branches)
