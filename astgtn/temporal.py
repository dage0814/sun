from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .activations import make_activation, sgfrelu
from .config import AblationConfig, ModelConfig


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10_000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_len, hidden_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence length exceeds positional encoding capacity")
        return sequence + self.encoding[:, : sequence.shape[1]].to(sequence.dtype)


class TransformerTemporalBranch(nn.Module):

    def __init__(self, config: ModelConfig, ablation: AblationConfig) -> None:
        super().__init__()
        activation = sgfrelu if ablation.use_sgfrelu else F.relu
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.position = SinusoidalPositionEncoding(config.hidden_dim)
        self.use_position = ablation.use_positional_encoding

    def forward(self, features: torch.Tensor, pre_len: int) -> torch.Tensor:
        batch, steps, stocks, hidden = features.shape
        sequence = features.permute(0, 2, 1, 3).reshape(batch * stocks, steps, hidden)
        if self.use_position:
            sequence = self.position(sequence)
        encoded = self.encoder(sequence)
        encoded = encoded.reshape(batch, stocks, steps, hidden).permute(0, 2, 1, 3)
        return encoded[:, -pre_len:]


class CausalConv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
            )
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(sequence, (self.left_padding, 0)))


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_sgfrelu: bool,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.activation1 = make_activation(use_sgfrelu)
        self.activation2 = make_activation(use_sgfrelu)
        self.output_activation = make_activation(use_sgfrelu)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.activation1(self.conv1(sequence)))
        output = self.dropout(self.activation2(self.conv2(output)))
        return self.output_activation(output + self.residual(sequence))


class TemporalConvNet(nn.Module):

    def __init__(self, config: ModelConfig, ablation: AblationConfig) -> None:
        super().__init__()
        blocks = []
        in_channels = config.num_input_features
        for level in range(config.tcn_layers):
            blocks.append(
                TCNResidualBlock(
                    in_channels,
                    config.tcn_channels,
                    config.tcn_kernel_size,
                    dilation=2**level,
                    dropout=config.dropout,
                    use_sgfrelu=ablation.use_sgfrelu,
                )
            )
            in_channels = config.tcn_channels
        self.blocks = nn.Sequential(*blocks)
        self.output_projection = nn.Conv1d(config.tcn_channels, config.hidden_dim, kernel_size=1)

    def forward(self, raw_features: torch.Tensor, pre_len: int) -> torch.Tensor:
        batch, steps, stocks, feature_count = raw_features.shape
        sequence = raw_features.permute(0, 2, 3, 1).reshape(
            batch * stocks, feature_count, steps
        )
        encoded = self.output_projection(self.blocks(sequence))
        encoded = encoded.transpose(1, 2).reshape(batch, stocks, steps, -1)
        encoded = encoded.permute(0, 2, 1, 3)
        return encoded[:, -pre_len:]
