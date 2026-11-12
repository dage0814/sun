from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import AblationConfig, TrainingConfig


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1).long()
        log_probabilities = F.log_softmax(flat_logits, dim=-1)
        log_pt = log_probabilities.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        alpha_t = torch.where(
            flat_targets == 1,
            torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype),
            torch.as_tensor(1.0 - self.alpha, device=logits.device, dtype=logits.dtype),
        )
        return (-alpha_t * (1.0 - pt).pow(self.gamma) * log_pt).mean()


class SoftMCCLoss(nn.Module):

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probability = F.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)[:, 1]
        truth = targets.reshape(-1).to(probability.dtype)
        tp = (probability * truth).sum()
        tn = ((1.0 - probability) * (1.0 - truth)).sum()
        fp = (probability * (1.0 - truth)).sum()
        fn = ((1.0 - probability) * truth).sum()
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + self.eps
        )
        return 1.0 - numerator / denominator


@dataclass
class ObjectiveOutput:
    total: torch.Tensor
    components: Dict[str, torch.Tensor]


class CombinedStockObjective(nn.Module):
    def __init__(self, training: TrainingConfig, ablation: AblationConfig) -> None:
        super().__init__()
        self.training = training
        self.ablation = ablation
        self.focal = FocalLoss(training.focal_alpha, training.focal_gamma)
        self.mcc = SoftMCCLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> ObjectiveOutput:
        components: Dict[str, torch.Tensor] = {}
        if self.ablation.use_focal_loss:
            components["focal"] = self.focal(logits, targets)
        else:
            components["cross_entropy"] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long()
            )
        if self.ablation.use_mcc_loss:
            components["mcc"] = self.mcc(logits, targets)
        classification = components.get("focal", components.get("cross_entropy"))
        total = self.training.focal_weight * classification
        if "mcc" in components:
            total = total + self.training.mcc_weight * components["mcc"]
        return ObjectiveOutput(total, components)


def temporal_mixup(
    features: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    *,
    return_permutation: bool = False,
):
    """Mix complete temporal samples after multimodal encoding."""

    if alpha <= 0:
        result = (features, targets, targets, 1.0)
        permutation = torch.arange(features.shape[0], device=features.device)
        return (*result, permutation) if return_permutation else result
    distribution = torch.distributions.Beta(alpha, alpha)
    coefficient = float(distribution.sample().item())
    permutation = torch.randperm(features.shape[0], device=features.device)
    mixed = coefficient * features + (1.0 - coefficient) * features[permutation]
    result = (mixed, targets, targets[permutation], coefficient)
    return (*result, permutation) if return_permutation else result


def mixup_objective(
    objective: CombinedStockObjective,
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    coefficient: float,
) -> ObjectiveOutput:
    left = objective(logits, targets_a)
    right = objective(logits, targets_b)
    total = coefficient * left.total + (1.0 - coefficient) * right.total
    names = set(left.components).union(right.components)
    components = {
        name: coefficient * left.components.get(name, 0.0)
        + (1.0 - coefficient) * right.components.get(name, 0.0)
        for name in names
    }
    return ObjectiveOutput(total, components)
