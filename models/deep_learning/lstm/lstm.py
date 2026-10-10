"""
LSTM sequence forecaster with configurable recurrent layers and multi-horizon projection heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """LSTM sequence model for time-series forecasting and directional classification.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    hidden : int, default 16
        Number of hidden units in the recurrent state.
    layers : int, default 1
        Number of stacked LSTM layers.
    dropout : float, default 0.0
        Dropout probability between recurrent layers.
    task : str, default 'regression'
        Learning task: 'regression' or 'classification'.
    horizons : int, default 1
        Number of forward forecasting horizons (output dimension).
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        layers: int = 1,
        dropout: float = 0.0,
        task: str = "regression",
        horizons: int = 1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.layers = layers
        self.task = task
        self.horizons = horizons

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )

        self.head = nn.Linear(hidden, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, sequence_length, n_features].

        Returns
        -------
        torch.Tensor
            Forecast tensor of shape [batch_size, horizons].
        """
        # x: [B, T, F]
        lstm_out, _ = self.lstm(x)
        # Use last time-step representation
        last_hidden = lstm_out[:, -1, :]  # [B, hidden]
        out = self.head(last_hidden)  # [B, horizons]
        return out
