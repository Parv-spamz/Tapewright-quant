"""
Causal dilated WaveNet architecture with gated activations, residual/skip connections, and exact receptive field tracking.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Convolution with causal left-padding to prevent forward-looking information leakage.

    Pads the sequence dimension on the left by (kernel_size - 1) * dilation so that the output
    at time step t depends strictly on inputs at time steps tau <= t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, T].

        Returns
        -------
        torch.Tensor
            Causally convolved tensor of shape [B, C_out, T].
        """
        # Left-pad sequence: pad=(pad_left, pad_right) for last dimension
        x_padded = F.pad(x, (self.pad, 0))
        return self.conv(x_padded)


class WaveNetResidualBlock(nn.Module):
    """Gated residual block combining dilated causal convolutions with residual and skip projections."""

    def __init__(
        self,
        channels: int,
        skip_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size

        # Dilated causal convolutions for filter and gate
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)

        # 1x1 convolutions for residual and skip outputs
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        tuple of (residual_output, skip_output)
            Both tensors have shape [B, C, T] and [B, C_skip, T].
        """
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        z = f * g

        res = x + self.residual_conv(z)
        skip = self.skip_conv(z)
        return res, skip


class WaveNetForecaster(nn.Module):
    """Deep dilated WaveNet forecaster for multi-horizon sequence modeling.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden : int, default 16
        Channel dimension in residual layers.
    dilations : sequence of int, default (1, 2, 4, 8)
        Dilation schedule across stacked causal blocks.
    kernel_size : int, default 2
        Convolution filter kernel size.
    horizons : int, default 1
        Number of forward prediction steps.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 2,
        horizons: int = 1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.dilations = list(dilations)
        self.kernel_size = kernel_size
        self.horizons = horizons

        # Initial 1x1 projection from input features to hidden channels
        self.input_conv = nn.Conv1d(n_features, hidden, kernel_size=1)

        # Stack of dilated causal residual blocks
        self.blocks = nn.ModuleList()
        for d in self.dilations:
            self.blocks.append(
                WaveNetResidualBlock(
                    channels=hidden,
                    skip_channels=hidden,
                    kernel_size=kernel_size,
                    dilation=d,
                )
            )

        # Projection head on aggregated skip representations
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, horizons, kernel_size=1),
        )

    def receptive_field(self) -> int:
        """Calculates theoretical receptive field length in time steps.

        Formula:
            RF = 1 + sum_{l=1}^L (kernel_size - 1) * dilation_l
        """
        rf = 1
        for d in self.dilations:
            rf += (self.kernel_size - 1) * d
        return rf

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates causal representations across all sequence time steps [B, horizons, T].

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, F].
        """
        # Transpose [B, T, F] -> [B, F, T]
        x_ch = x.transpose(1, 2)
        h = self.input_conv(x_ch)

        skips: list[torch.Tensor] = []
        for block in self.blocks:
            h, skip = block(h)
            skips.append(skip)

        total_skip = sum(skips)
        out_seq = self.head(total_skip)  # [B, horizons, T]
        return out_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating prediction for final sequence time step [B, horizons]."""
        out_seq = self.forward_sequence(x)
        # Select last time step: [B, horizons, -1] -> [B, horizons]
        return out_seq[:, :, -1]
