from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def sgfrelu(x: torch.Tensor, shift: float = 0.7524614220708874) -> torch.Tensor:

    return torch.clamp_min(F.gelu(x - shift), 0.0)


class SGFReLU(nn.Module):
    def __init__(self, shift: float = 0.7524614220708874) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sgfrelu(x, self.shift)


def make_activation(use_sgfrelu: bool) -> nn.Module:
    return SGFReLU() if use_sgfrelu else nn.ReLU()
