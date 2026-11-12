from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class RevIN(nn.Module):
    """Normalize [B, T, N, F] along T, independently for each sample and stock."""

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, 1, num_features))
        self._stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"RevIN expects [B,T,N,F], got {tuple(x.shape)}")
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
        self._stats = mean, stdev
        output = (x - mean) / stdev
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._stats is None:
            raise RuntimeError("normalize must be called before denormalize")
        mean, stdev = self._stats
        output = x
        if self.affine:
            output = (output - self.bias) / (self.weight + self.eps)
        return output * stdev + mean

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            return self.normalize(x)
        if mode == "denorm":
            return self.denormalize(x)
        raise ValueError(f"Unknown RevIN mode: {mode}")
