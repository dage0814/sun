from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Dict


def _positive_integers(config, names) -> None:
    for name in names:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _nonnegative(config, names) -> None:
    for name in names:
        value = getattr(config, name)
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class AblationConfig:

    use_ic_selection: bool = True
    use_genetic_factors: bool = True
    use_text: bool = True
    use_finbert: bool = True
    use_bilstm: bool = True
    use_industry_attention: bool = True
    use_listmle: bool = True
    use_market_cap_weight: bool = True
    use_cross_attention_ca1: bool = True
    use_cross_attention_ca2: bool = True
    use_cross_attention_ca3: bool = True

    use_revin: bool = True
    use_prior_graph: bool = True
    use_adaptive_graph: bool = True
    use_dynamic_graph: bool = True
    use_gat: bool = True
    use_transformer: bool = True
    use_positional_encoding: bool = True
    use_tcn: bool = True
    use_sgfrelu: bool = True
    use_focal_loss: bool = True
    use_mcc_loss: bool = True
    use_mixup: bool = True

    @classmethod
    def preset(cls, name: str) -> "AblationConfig":

        key = name.lower().replace("-", "_")
        changes: Dict[str, Dict[str, bool]] = {
            "full": {},
            "no_ic": {"use_ic_selection": False},
            "no_gp": {"use_genetic_factors": False},
            "no_text": {
                "use_text": False,
                "use_finbert": False,
                "use_bilstm": False,
                "use_industry_attention": False,
                "use_listmle": False,
                "use_market_cap_weight": False,
                "use_cross_attention_ca1": False,
                "use_cross_attention_ca2": False,
                "use_cross_attention_ca3": False,
            },
            "no_finbert": {"use_finbert": False},
            "no_bilstm": {"use_bilstm": False},
            "no_industry_attention": {"use_industry_attention": False},
            "no_listmle": {"use_listmle": False},
            "no_market_cap": {"use_market_cap_weight": False},
            "no_cross_attention": {
                "use_cross_attention_ca1": False,
                "use_cross_attention_ca2": False,
                "use_cross_attention_ca3": False,
            },
            "no_ca3": {"use_cross_attention_ca3": False},
            "no_revin": {"use_revin": False},
            "no_prior_graph": {"use_prior_graph": False},
            "no_adaptive_graph": {
                "use_adaptive_graph": False,
                "use_dynamic_graph": False,
            },
            "static_adaptive_graph": {"use_dynamic_graph": False},
            "no_gat": {"use_gat": False},
            "no_transformer": {"use_transformer": False},
            "no_position": {"use_positional_encoding": False},
            "no_tcn": {"use_tcn": False},
            "relu": {"use_sgfrelu": False},
            "focal_only": {"use_mcc_loss": False},
            "cross_entropy": {"use_focal_loss": False, "use_mcc_loss": False},
            "no_mixup": {"use_mixup": False},
        }
        if key not in changes:
            raise ValueError(f"Unknown ablation preset: {name}. Available: {sorted(changes)}")
        return replace(cls(), **changes[key])

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class TextEncoderConfig:
    pretrained_model_name: str = "ProsusAI/finbert"
    fallback_vocab_size: int = 30_522
    fallback_hidden_size: int = 128
    lstm_hidden_size: int = 128
    industry_feature_dim: int = 256
    attention_key_dim: int = 128
    dropout: float = 0.1
    freeze_finbert_layers: int = 0

    def __post_init__(self) -> None:
        _positive_integers(self, (
            "fallback_vocab_size", "fallback_hidden_size", "lstm_hidden_size",
            "industry_feature_dim", "attention_key_dim",
        ))
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (isinstance(self.freeze_finbert_layers, bool)
                or not isinstance(self.freeze_finbert_layers, int)
                or self.freeze_finbert_layers < 0):
            raise ValueError("freeze_finbert_layers must be a nonnegative integer")


@dataclass(frozen=True)
class ModelConfig:
    num_stocks: int
    num_input_features: int
    pre_len: int = 1
    hidden_dim: int = 128
    graph_heads: int = 4
    graph_layers: int = 2
    graph_top_k: int = 12
    node_embedding_dim: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 6
    transformer_ff_dim: int = 512
    tcn_channels: int = 128
    tcn_layers: int = 4
    tcn_kernel_size: int = 7
    dropout: float = 0.2
    prior_graph_weight: float = 1.0
    adaptive_graph_weight: float = 1.0
    dynamic_graph_weight: float = 1.0
    num_classes: int = 2

    def __post_init__(self) -> None:
        _positive_integers(self, (
            "num_stocks", "num_input_features", "pre_len", "hidden_dim",
            "graph_heads", "graph_layers", "graph_top_k", "node_embedding_dim",
            "transformer_heads", "transformer_layers", "transformer_ff_dim",
            "tcn_channels", "tcn_layers", "tcn_kernel_size", "num_classes",
        ))
        if self.hidden_dim % self.graph_heads:
            raise ValueError("hidden_dim must be divisible by graph_heads")
        if self.hidden_dim % self.transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if self.tcn_kernel_size < 2:
            raise ValueError("tcn_kernel_size must be at least 2")
        if self.num_classes != 2:
            raise ValueError("Stock direction targets and losses require two classes")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        _nonnegative(self, (
            "prior_graph_weight", "adaptive_graph_weight", "dynamic_graph_weight",
        ))


@dataclass(frozen=True)
class TrainingConfig:
    stage1_epochs: int = 10
    stage2_epochs: int = 50
    stage1_lr: float = 2e-5
    stage2_lr: float = 1e-4
    encoder_finetune_lr: float = 5e-6
    weight_decay: float = 1e-4
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    focal_weight: float = 1.0
    mcc_weight: float = 0.2
    mixup_alpha: float = 1.0
    grad_clip_norm: float = 5.0
    patience: int = 8

    def __post_init__(self) -> None:
        _positive_integers(self, ("stage2_epochs", "patience"))
        if (isinstance(self.stage1_epochs, bool)
                or not isinstance(self.stage1_epochs, int) or self.stage1_epochs < 0):
            raise ValueError("stage1_epochs must be a nonnegative integer")
        _nonnegative(self, (
            "stage1_lr", "stage2_lr", "encoder_finetune_lr", "weight_decay",
            "focal_gamma", "focal_weight", "mcc_weight", "mixup_alpha", "grad_clip_norm",
        ))
        for name in ("stage1_lr", "stage2_lr", "encoder_finetune_lr", "focal_weight", "grad_clip_norm"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.focal_alpha <= 1:
            raise ValueError("focal_alpha must be in [0, 1]")
