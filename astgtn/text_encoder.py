from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import AblationConfig, TextEncoderConfig


@dataclass
class TextEncoderOutput:
    industry_features: torch.Tensor
    industry_scores: torch.Tensor
    news_attention: torch.Tensor


class EmbeddingTextBackbone(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class FinFEBERTEncoder(nn.Module):

    def __init__(
        self,
        num_industries: int,
        config: TextEncoderConfig,
        ablation: AblationConfig,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.num_industries = num_industries
        self.config = config
        self.ablation = ablation
        if backbone is not None:
            self.backbone = backbone
            backbone_dim = self._infer_hidden_size(backbone)
        elif ablation.use_finbert:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "FinBERT requires the optional 'transformers' package. "
                    "Install it in the active Conda environment or select ablation='no_finbert'."
                ) from exc
            self.backbone = AutoModel.from_pretrained(config.pretrained_model_name)
            backbone_dim = int(self.backbone.config.hidden_size)
            self._freeze_bottom_layers(config.freeze_finbert_layers)
        else:
            self.backbone = EmbeddingTextBackbone(
                config.fallback_vocab_size, config.fallback_hidden_size
            )
            backbone_dim = config.fallback_hidden_size

        if ablation.use_bilstm:
            self.context_encoder = nn.LSTM(
                backbone_dim,
                config.lstm_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            context_dim = 2 * config.lstm_hidden_size
        else:
            self.context_encoder = nn.Linear(backbone_dim, 2 * config.lstm_hidden_size)
            context_dim = 2 * config.lstm_hidden_size

        self.key_projection = nn.Linear(context_dim, config.attention_key_dim, bias=False)
        self.value_projection = nn.Linear(context_dim, config.industry_feature_dim, bias=False)
        self.pool_projection = nn.Linear(context_dim, config.industry_feature_dim, bias=False)
        self.industry_queries = nn.Parameter(
            torch.empty(num_industries, config.attention_key_dim)
        )
        nn.init.xavier_uniform_(self.industry_queries)

        self.industry_weight = nn.Parameter(
            torch.empty(num_industries, config.industry_feature_dim, config.industry_feature_dim)
        )
        self.industry_bias = nn.Parameter(
            torch.zeros(num_industries, config.industry_feature_dim)
        )
        nn.init.xavier_uniform_(self.industry_weight)
        self.dropout = nn.Dropout(config.dropout)
        self.score_head = nn.Linear(config.industry_feature_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        news_mask: Optional[torch.Tensor] = None,
    ) -> TextEncoderOutput:
        if input_ids.ndim != 4:
            raise ValueError("input_ids must use shape [B,T,M,L]")
        batch, steps, news_count, token_count = input_ids.shape
        if news_count == 0 or token_count == 0:
            raise ValueError("News and token dimensions must be positive")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if news_mask is None:
            news_mask = attention_mask.any(dim=-1)
        if news_mask.shape != input_ids.shape[:-1]:
            raise ValueError("news_mask must use shape [B,T,M]")
        flat_news_mask = news_mask.reshape(batch * steps, news_count).bool()
        empty_days = ~flat_news_mask.any(dim=1)
        flat_ids = input_ids.reshape(-1, token_count)
        flat_attention = attention_mask.reshape(-1, token_count)
        sentence_features = self._encode_sentences(flat_ids, flat_attention)
        sentence_features = sentence_features.reshape(batch * steps, news_count, -1)
        sentence_features = sentence_features.masked_fill(~flat_news_mask.unsqueeze(-1), 0.0)

        if self.ablation.use_bilstm:

            order = torch.argsort((~flat_news_mask).long(), dim=1, stable=True)
            gather_index = order.unsqueeze(-1).expand_as(sentence_features)
            packed_input = sentence_features.gather(1, gather_index)
            lengths = flat_news_mask.sum(dim=1).clamp_min(1).cpu()
            packed = pack_padded_sequence(packed_input, lengths, batch_first=True, enforce_sorted=False)
            encoded, _ = self.context_encoder(packed)
            encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=news_count)
            contextual = torch.zeros_like(encoded).scatter(
                1, order.unsqueeze(-1).expand_as(encoded), encoded
            )
        else:
            contextual = self.context_encoder(sentence_features)

        safe_mask = flat_news_mask.clone()
        safe_mask[empty_days, 0] = True
        contextual = contextual.masked_fill(~safe_mask.unsqueeze(-1), 0.0)

        valid_weights = flat_news_mask.unsqueeze(-1).to(contextual.dtype)
        pooled = (contextual * valid_weights).sum(dim=1) / valid_weights.sum(dim=1).clamp_min(1.0)
        pooled = self.pool_projection(pooled)

        if self.ablation.use_industry_attention:
            keys = self.key_projection(contextual)
            values = self.value_projection(contextual)
            logits = torch.einsum("id,bmd->bim", self.industry_queries, keys)
            logits = logits / (self.config.attention_key_dim**0.5)
            logits = logits.masked_fill(~safe_mask.unsqueeze(1), torch.finfo(logits.dtype).min)
            weights = F.softmax(logits, dim=-1)
            weights = weights.masked_fill(empty_days[:, None, None], 0.0)
            industry = torch.einsum("bim,bmd->bid", weights, values)
        else:
            weights = torch.zeros(
                batch * steps,
                self.num_industries,
                news_count,
                device=input_ids.device,
                dtype=sentence_features.dtype,
            )
            industry = torch.zeros_like(pooled).unsqueeze(1).expand(-1, self.num_industries, -1)

        industry = industry + pooled.unsqueeze(1)
        industry = torch.einsum("bid,idf->bif", industry, self.industry_weight)
        industry = F.gelu(industry + self.industry_bias.unsqueeze(0))
        industry = self.dropout(industry)
        scores = self.score_head(industry).squeeze(-1)
        return TextEncoderOutput(
            industry.reshape(batch, steps, self.num_industries, -1),
            scores.reshape(batch, steps, self.num_industries),
            weights.reshape(batch, steps, self.num_industries, news_count),
        )

    def _encode_sentences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(output, torch.Tensor):
            return output if output.ndim == 2 else output[:, 0]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        if isinstance(output, (tuple, list)):
            return output[0][:, 0]
        raise TypeError("Unsupported text backbone output")

    @staticmethod
    def _infer_hidden_size(backbone: nn.Module) -> int:
        if hasattr(backbone, "hidden_size"):
            return int(backbone.hidden_size)
        if hasattr(backbone, "config") and hasattr(backbone.config, "hidden_size"):
            return int(backbone.config.hidden_size)
        raise ValueError("Injected backbone must expose hidden_size or config.hidden_size")

    def _freeze_bottom_layers(self, count: int) -> None:
        if count <= 0:
            return
        layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
        if layers is None:
            raise ValueError("Configured FinBERT backbone does not expose encoder.layer")
        for layer in layers[:count]:
            for parameter in layer.parameters():
                parameter.requires_grad = False


class ListMLELoss(nn.Module):

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if scores.shape != relevance.shape:
            raise ValueError("scores and relevance must have identical shapes")
        flat_scores = scores.reshape(-1, scores.shape[-1])
        flat_relevance = relevance.reshape(-1, relevance.shape[-1])
        if mask is None:
            flat_mask = torch.ones_like(flat_scores, dtype=torch.bool)
        else:
            flat_mask = mask.reshape_as(flat_scores).bool()

        losses = []
        for row_scores, row_relevance, row_mask in zip(flat_scores, flat_relevance, flat_mask):
            valid_scores = row_scores[row_mask]
            valid_relevance = row_relevance[row_mask]
            if valid_scores.numel() < 2:
                continue
            order = torch.argsort(valid_relevance, descending=True)
            ordered_scores = valid_scores[order]
            denominators = torch.logcumsumexp(ordered_scores.flip(0), dim=0).flip(0)
            losses.append((denominators - ordered_scores).sum())
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()
