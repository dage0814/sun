from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import nn

from .activations import make_activation
from .config import AblationConfig, ModelConfig, TextEncoderConfig
from .fusion import DualPathCrossModalFusion, FusionOutput
from .model import AdaptiveSpatioTemporalGraphTransformer, ModelOutput
from .text_encoder import FinFEBERTEncoder, TextEncoderOutput
from .history import PointInTimePrior


@dataclass
class EncodedPipelineOutput:
    features: torch.Tensor
    text: Optional[TextEncoderOutput]
    fusion: Optional[FusionOutput]


@dataclass
class MultiModalOutput:
    prediction: ModelOutput
    encoded: EncodedPipelineOutput


class MultiModalASTGTN(nn.Module):

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        text_config: TextEncoderConfig,
        ablation: AblationConfig,
        num_industries: int,
        num_quant_features: int,
        num_technical_features: int,
        stock_industry: torch.Tensor,
        prior_adjacency: Optional[torch.Tensor] = None,
        prior_graph: Optional[PointInTimePrior] = None,
        fusion_dim: int = 128,
        fusion_heads: int = 4,
        text_backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if prior_adjacency is not None and prior_graph is not None:
            raise ValueError("Supply either prior_adjacency or prior_graph, not both")
        if num_industries < 1 or num_quant_features < 1 or num_technical_features < 0:
            raise ValueError("Invalid industry or feature counts")
        if fusion_dim < 1 or fusion_heads < 1 or fusion_dim % fusion_heads:
            raise ValueError("fusion_dim must be positive and divisible by fusion_heads")
        self.ablation = ablation
        self.num_industries = num_industries
        self.num_quant_features = num_quant_features
        self.num_technical_features = num_technical_features
        stock_industry = torch.as_tensor(stock_industry)
        if stock_industry.shape != (model_config.num_stocks,):
            raise ValueError("stock_industry must contain one id per stock")
        if not torch.isfinite(stock_industry).all() or not torch.equal(
            stock_industry, stock_industry.long()
        ):
            raise ValueError("stock_industry must contain integer industry ids")
        stock_industry = stock_industry.long()
        if (stock_industry < 0).any() or (stock_industry >= num_industries).any():
            raise ValueError("stock_industry contains an invalid industry id")
        self.register_buffer("stock_industry", stock_industry)

        self.text_encoder: Optional[FinFEBERTEncoder]
        if ablation.use_text:
            self.text_encoder = FinFEBERTEncoder(
                num_industries, text_config, ablation, backbone=text_backbone
            )
        else:
            self.text_encoder = None
        self.cross_modal_fusion = DualPathCrossModalFusion(
            num_quant_features=num_quant_features,
            news_feature_dim=text_config.industry_feature_dim,
            model_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=model_config.dropout,
            ablation=ablation,
        )
        self.quantitative_only = nn.Sequential(
            nn.Linear(num_quant_features, fusion_dim),
            make_activation(ablation.use_sgfrelu),
        )
        backbone_config = replace(
            model_config,
            num_input_features=fusion_dim + num_technical_features,
        )
        self.backbone = AdaptiveSpatioTemporalGraphTransformer(
            backbone_config, ablation, prior_adjacency, prior_graph
        )

    def encode_features(
        self,
        *,
        quantitative: torch.Tensor,
        technical: torch.Tensor,
        market_caps: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        news_mask: Optional[torch.Tensor] = None,
    ) -> EncodedPipelineOutput:
        if quantitative.ndim != 4 or technical.ndim != 4:
            raise ValueError("quantitative and technical must use [B,T,N,F]")
        if quantitative.shape[:3] != technical.shape[:3]:
            raise ValueError("quantitative and technical tensors are not aligned")
        if quantitative.shape[2] != self.stock_industry.numel():
            raise ValueError("Unexpected number of stocks")
        if quantitative.shape[-1] != self.num_quant_features:
            raise ValueError("Unexpected number of quantitative features")
        if technical.shape[-1] != self.num_technical_features:
            raise ValueError("Unexpected number of technical features")

        if self.ablation.use_text:
            if input_ids is None or attention_mask is None:
                raise ValueError("Text-enabled runs require input_ids and attention_mask")
            assert self.text_encoder is not None
            text_output = self.text_encoder(input_ids, attention_mask, news_mask)
            if text_output.industry_features.shape[:2] != quantitative.shape[:2]:
                raise ValueError("News and quantitative batch/time dimensions are not aligned")
            fusion_output = self.cross_modal_fusion(
                quantitative,
                text_output.industry_features,
                self.stock_industry,
                market_caps,
            )
            combined = fusion_output.combined
        else:
            text_output = None
            fusion_output = None
            combined = self.quantitative_only(quantitative)

        return EncodedPipelineOutput(
            torch.cat([combined, technical], dim=-1), text_output, fusion_output
        )

    def forward(self, *, window_cutoffs: Optional[torch.Tensor] = None, **inputs) -> MultiModalOutput:
        encoded = self.encode_features(**inputs)
        return MultiModalOutput(self.backbone(encoded.features, window_cutoffs), encoded)
