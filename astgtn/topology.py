from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import AblationConfig, ModelConfig
from .history import PointInTimePrior


class PriorGraphBuilder:


    @staticmethod
    def from_metadata(
        industry_ids: torch.Tensor,
        concept_membership: Optional[torch.Tensor] = None,
        *,
        industry_weight: float = 1.0,
        concept_weight: float = 1.0,
        self_loop_weight: float = 1.0,
    ) -> torch.Tensor:
        industry_ids = torch.as_tensor(industry_ids, dtype=torch.long)
        if industry_ids.ndim != 1:
            raise ValueError("industry_ids must use shape [N]")
        adjacency = (industry_ids[:, None] == industry_ids[None, :]).float() * industry_weight
        if concept_membership is not None:
            concepts = torch.as_tensor(concept_membership, dtype=torch.float)
            if concepts.ndim != 2 or concepts.shape[0] != industry_ids.numel():
                raise ValueError("concept_membership must use shape [N,C]")
            shared = concepts @ concepts.transpose(0, 1)
            adjacency = adjacency + (shared > 0).to(adjacency.dtype) * concept_weight
        adjacency.fill_diagonal_(self_loop_weight)
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class AdaptiveAdjacency(nn.Module):

    def __init__(self, config: ModelConfig, ablation: AblationConfig) -> None:
        super().__init__()
        self.config = config
        self.ablation = ablation
        self.node_source = nn.Parameter(
            torch.empty(config.num_stocks, config.node_embedding_dim)
        )
        self.node_target = nn.Parameter(
            torch.empty(config.num_stocks, config.node_embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_source)
        nn.init.xavier_uniform_(self.node_target)
        self.dynamic_query = nn.Linear(config.hidden_dim, config.node_embedding_dim, bias=False)
        self.dynamic_key = nn.Linear(config.hidden_dim, config.node_embedding_dim, bias=False)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("AdaptiveAdjacency expects [B,T,N,D]")
        static_logits = F.relu(self.node_source @ self.node_target.transpose(0, 1))
        static = self._sparsify(static_logits).unsqueeze(0).unsqueeze(0)
        static = static.expand(features.shape[0], features.shape[1], -1, -1)

        if self.ablation.use_dynamic_graph:
            query = self.dynamic_query(features)
            key = self.dynamic_key(features)
            dynamic_logits = torch.einsum("btid,btjd->btij", query, key)
            dynamic_logits = F.relu(dynamic_logits / (self.config.node_embedding_dim**0.5))
            dynamic = self._sparsify(dynamic_logits)
        else:
            dynamic = torch.zeros_like(static)
        return static, dynamic

    def _sparsify(self, logits: torch.Tensor) -> torch.Tensor:
        stocks = logits.shape[-1]
        top_k = min(max(1, self.config.graph_top_k), stocks)
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        sparse_logits.scatter_(-1, indices, values)
        identity = torch.eye(stocks, device=logits.device, dtype=torch.bool)
        while identity.ndim < sparse_logits.ndim:
            identity = identity.unsqueeze(0)
        sparse_logits = torch.where(identity, torch.maximum(sparse_logits, torch.zeros_like(sparse_logits)), sparse_logits)
        return F.softmax(sparse_logits, dim=-1)


class DualTopology(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        ablation: AblationConfig,
        prior_adjacency: Optional[torch.Tensor] = None,
        prior_graph: Optional[PointInTimePrior] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ablation = ablation
        if prior_adjacency is not None and prior_graph is not None:
            raise ValueError("Supply either prior_adjacency or prior_graph, not both")
        if prior_adjacency is None:
            prior_adjacency = torch.eye(config.num_stocks)
        prior_adjacency = torch.as_tensor(prior_adjacency, dtype=torch.float)
        if prior_adjacency.shape != (config.num_stocks, config.num_stocks):
            raise ValueError(
                f"prior_adjacency must be [{config.num_stocks},{config.num_stocks}]"
            )
        if not torch.equal(prior_adjacency, torch.eye(config.num_stocks, device=prior_adjacency.device)):
            raise ValueError(
                "Undated prior_adjacency is not allowed. Supply a PointInTimePrior "
                "with source-backed relation events and per-window cutoffs."
            )
        if prior_graph is not None and prior_graph.num_stocks != config.num_stocks:
            raise ValueError("Point-in-time graph stock count does not match the model")
        self.register_buffer("prior_adjacency", prior_adjacency)
        self.prior_graph = prior_graph
        self.adaptive = AdaptiveAdjacency(config, ablation)

    def forward(self, features: torch.Tensor, window_cutoffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, steps, stocks, _ = features.shape
        adjacency = torch.zeros(
            batch, steps, stocks, stocks, device=features.device, dtype=features.dtype
        )
        if self.ablation.use_prior_graph:
            if not torch.equal(self.prior_adjacency, torch.eye(stocks, device=self.prior_adjacency.device)):
                raise ValueError("Checkpoint contains an undated non-identity prior; rebuild dated events")
            prior = self.prior_adjacency
            if self.prior_graph is not None:
                if window_cutoffs is None or window_cutoffs.shape != (batch, steps):
                    raise ValueError("Dated prior graphs require window_cutoffs with shape [B,T]")
                prior = self.prior_graph(window_cutoffs).to(device=features.device, dtype=features.dtype)
            adjacency = adjacency + self.config.prior_graph_weight * prior
        if self.ablation.use_adaptive_graph:
            static, dynamic = self.adaptive(features)
            adjacency = adjacency + self.config.adaptive_graph_weight * static
            if self.ablation.use_dynamic_graph:
                adjacency = adjacency + self.config.dynamic_graph_weight * dynamic
        identity = torch.eye(stocks, device=features.device, dtype=features.dtype)
        adjacency = adjacency + identity
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
