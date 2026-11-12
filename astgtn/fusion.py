from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .config import AblationConfig


@dataclass
class FusionOutput:
    combined: torch.Tensor
    stock_news: torch.Tensor
    market_cap_weight: torch.Tensor
    attention: Dict[str, Optional[torch.Tensor]]


class QuantitativeFeatureTokenizer(nn.Module):

    def __init__(self, num_features: int, token_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, token_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, token_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"Expected {self.weight.shape[0]} quantitative features, got {values.shape[-1]}"
            )
        return values.unsqueeze(-1) * self.weight + self.bias


class MarketCapAdapter(nn.Module):

    def __init__(self, normalize_within_industry: bool = True, max_weight: float = 10.0) -> None:
        super().__init__()
        self.normalize_within_industry = normalize_within_industry
        self.max_weight = max_weight

    def forward(
        self,
        industry_features: torch.Tensor,
        stock_industry: torch.Tensor,
        market_caps: torch.Tensor,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if industry_features.ndim != 4 or market_caps.ndim != 3:
            raise ValueError("Expected industry_features [B,T,I,D] and market_caps [B,T,N]")
        if market_caps.shape[:2] != industry_features.shape[:2]:
            raise ValueError("Market caps and industry features are not aligned")
        if stock_industry.ndim != 1 or stock_industry.numel() != market_caps.shape[2]:
            raise ValueError("stock_industry must contain one id per stock")
        stock_industry = stock_industry.to(device=industry_features.device, dtype=torch.long)
        if int(stock_industry.min()) < 0 or int(stock_industry.max()) >= industry_features.shape[2]:
            raise ValueError("stock_industry contains an invalid industry id")
        stock_news = industry_features[:, :, stock_industry, :]
        if not enabled:
            return stock_news, torch.ones_like(market_caps)

        if not torch.isfinite(market_caps).all() or (market_caps <= 0).any():
            raise ValueError("Market caps must be finite and strictly positive")
        caps = market_caps.to(device=industry_features.device, dtype=industry_features.dtype)
        weight = torch.ones_like(caps)
        for industry in torch.unique(stock_industry).tolist():
            member_mask = stock_industry == industry

            if int(member_mask.sum()) == 1:
                continue
            member_caps = caps[:, :, member_mask]
            other_caps = member_caps.sum(dim=-1, keepdim=True) - member_caps
            member_weight = other_caps / (member_caps + 1e-8)
            if self.normalize_within_industry:
                member_weight = member_weight / member_weight.mean(dim=-1, keepdim=True).clamp_min(1e-8)
            weight[:, :, member_mask] = member_weight.clamp(max=self.max_weight)
        return stock_news * weight.unsqueeze(-1), weight


class DualPathCrossModalFusion(nn.Module):

    def __init__(
        self,
        num_quant_features: int,
        news_feature_dim: int,
        model_dim: int = 128,
        num_news_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        ablation: Optional[AblationConfig] = None,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.ablation = ablation or AblationConfig()
        self.model_dim = model_dim
        self.num_news_tokens = num_news_tokens
        self.quant_tokenizer = QuantitativeFeatureTokenizer(num_quant_features, model_dim)
        self.news_tokenizer = nn.Linear(news_feature_dim, num_news_tokens * model_dim)
        self.market_cap_adapter = MarketCapAdapter()
        self.ca1 = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.ca2 = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.ca3 = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.ca1_norm = nn.LayerNorm(model_dim)
        self.ca2_norm = nn.LayerNorm(model_dim)
        self.ca3_norm = nn.LayerNorm(model_dim)
        self.baseline_projection = nn.Linear(2 * model_dim, model_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        quantitative: torch.Tensor,
        industry_news: torch.Tensor,
        stock_industry: torch.Tensor,
        market_caps: torch.Tensor,
    ) -> FusionOutput:
        if quantitative.ndim != 4:
            raise ValueError("quantitative must use shape [B,T,N,Fq]")
        if market_caps.shape != quantitative.shape[:3]:
            raise ValueError("market_caps must align with quantitative [B,T,N]")
        if industry_news.shape[:2] != quantitative.shape[:2]:
            raise ValueError("News and quantitative batch/time dimensions are not aligned")
        stock_news, cap_weight = self.market_cap_adapter(
            industry_news,
            stock_industry,
            market_caps,
            enabled=self.ablation.use_market_cap_weight,
        )
        batch, steps, stocks, _ = quantitative.shape
        groups = batch * steps * stocks
        quant_tokens = self.quant_tokenizer(quantitative).reshape(
            groups, quantitative.shape[-1], self.model_dim
        )
        news_tokens = self.news_tokenizer(stock_news).reshape(
            groups, self.num_news_tokens, self.model_dim
        )

        attention: Dict[str, Optional[torch.Tensor]] = {"ca1": None, "ca2": None, "ca3": None}
        if self.ablation.use_cross_attention_ca1:
            ca1_delta, attention["ca1"] = self.ca1(
                quant_tokens, news_tokens, news_tokens, need_weights=True, average_attn_weights=False
            )
            stock_guided_news = self.ca1_norm(quant_tokens + ca1_delta)
        else:
            stock_guided_news = quant_tokens

        if self.ablation.use_cross_attention_ca2:
            ca2_delta, attention["ca2"] = self.ca2(
                news_tokens, quant_tokens, quant_tokens, need_weights=True, average_attn_weights=False
            )
            news_guided_stock = self.ca2_norm(news_tokens + ca2_delta)
        else:
            news_guided_stock = news_tokens

        if self.ablation.use_cross_attention_ca3:
            ca3_delta, attention["ca3"] = self.ca3(
                stock_guided_news,
                news_guided_stock,
                news_guided_stock,
                need_weights=True,
                average_attn_weights=False,
            )
            fused_tokens = self.ca3_norm(stock_guided_news + ca3_delta)
            pooled = fused_tokens.mean(dim=1)
        else:
            pooled = self.baseline_projection(
                torch.cat(
                    [stock_guided_news.mean(dim=1), news_guided_stock.mean(dim=1)], dim=-1
                )
            )
        combined = self.output_projection(pooled).reshape(batch, steps, stocks, self.model_dim)
        return FusionOutput(combined, stock_news, cap_weight, attention)
