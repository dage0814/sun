from __future__ import annotations

from typing import Dict, Iterable

from .config import AblationConfig


DEFAULT_ABLATIONS = (
    "full",
    "no_text",
    "no_gp",
    "no_market_cap",
    "no_cross_attention",
    "no_ca3",
    "no_revin",
    "no_prior_graph",
    "no_adaptive_graph",
    "no_tcn",
    "no_transformer",
    "relu",
    "focal_only",
    "no_mixup",
)


def build_ablation_matrix(names: Iterable[str] = DEFAULT_ABLATIONS) -> Dict[str, AblationConfig]:
    return {name: AblationConfig.preset(name) for name in names}
