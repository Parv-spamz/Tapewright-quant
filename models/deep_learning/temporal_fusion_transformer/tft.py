"""
Simplified Temporal Fusion Transformer (SimpleTFT) implementation.

Document: Simplified, not the full TFT paper.
This implementation contains the core building blocks of TFT:
- Gated Residual Networks (GRN) with Gated Linear Units (GLU) and LayerNorm.
- Per-timestep Variable Selection Networks (VSN) with exposed softmax attention weights.
- LSTM encoder for causal temporal processing.
- Multi-head self-attention with strict upper-triangular causal mask.
- Quantile prediction heads (default: 0.1, 0.5, 0.9) and task-specific heads.
- Pinball quantile loss function.
- sort_quantiles helper to enforce monotonic quantile outputs.
- variable_importance extraction helper.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def quantile_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """Computes the multi-quantile pinball loss for quantile forecasting.

    For each quantile q in quantiles:
        error = target - pred_q
        loss_q = max(q * error, (q - 1) * error)
    The total loss is averaged across all quantiles and samples.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted quantiles of shape [B, Q] or [B, H, Q].
    target : torch.Tensor
        Ground truth observations of shape [B], [B, 1], [B, H], or [B, H, 1].
    quantiles : sequence of float, default (0.1, 0.5, 0.9)
        Quantile levels to evaluate.

    Returns
    -------
    torch.Tensor
        Scalar pinball loss tensor.
    """
    if target.ndim == 1:
        target_tensor = target.unsqueeze(-1)
    elif preds.ndim > target.ndim:
        target_tensor = target.unsqueeze(-1)
    else:
        target_tensor = target

    q_tensor = torch.as_tensor(quantiles, dtype=preds.dtype, device=preds.device)
    view_shape = [1] * (preds.ndim - 1) + [len(quantiles)]
    q_broadcast = q_tensor.view(*view_shape)

    error = target_tensor - preds
    loss = torch.maximum(q_broadcast * error, (q_broadcast - 1.0) * error)
    return loss.mean()


def sort_quantiles(preds: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Ensures predicted quantiles are monotonically non-decreasing along the quantile axis.

    Parameters
    ----------
    preds : torch.Tensor or np.ndarray
        Quantile predictions of shape [..., Q].

    Returns
    -------
    torch.Tensor or np.ndarray
        Monotonically sorted quantile predictions.
    """
    if isinstance(preds, torch.Tensor):
        return torch.sort(preds, dim=-1).values
    return np.sort(preds, axis=-1)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) providing non-linear representation learning with GLU gating."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.linear1 = nn.Linear(d_in, d_out)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(d_out, d_out * 2)
        self.glu = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.elu(self.linear1(x))
        h = self.glu(self.linear2(h))
        h = self.dropout(h)
        return self.layer_norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """Per-timestep Variable Selection Network (VSN) producing feature weights and fused embeddings."""

    def __init__(self, n_features: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.feature_projections = nn.ModuleList([nn.Linear(1, hidden) for _ in range(n_features)])
        self.weight_grn = GatedResidualNetwork(n_features, n_features, dropout=dropout)
        self.last_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluates variable selection weights and weighted feature combination.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, F].

        Returns
        -------
        tuple of (selected_features, weights)
            selected_features: [B, T, hidden]
            weights: [B, T, F] softmax attention weights over input features.
        """
        weight_logits = self.weight_grn(x)
        weights = torch.softmax(weight_logits, dim=-1)
        self.last_weights = weights.detach()

        proj_list = [proj(x[..., i : i + 1]) for i, proj in enumerate(self.feature_projections)]
        proj_stack = torch.stack(proj_list, dim=-2)  # [B, T, F, hidden]
        selected = torch.sum(proj_stack * weights.unsqueeze(-1), dim=-2)  # [B, T, hidden]
        return selected, weights


class SimpleTFT(nn.Module):
    """Simplified Temporal Fusion Transformer (SimpleTFT).

    Combines per-timestep variable selection, an LSTM recurrent encoder,
    multi-head self-attention with strict causal masking, and quantile prediction heads.

    Parameters
    ----------
    n_features : int
        Number of input time series features.
    hidden : int, default 16
        Latent embedding dimension.
    quantiles : sequence of float, default (0.1, 0.5, 0.9)
        Target quantiles for quantile forecasting.
    task : str, default 'quantile'
        Forecasting task: 'quantile', 'classification', or 'regression'.
    dropout : float, default 0.0
        Dropout probability.
    num_heads : int, default 2
        Number of attention heads in causal multi-head self-attention.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        task: str = "quantile",
        dropout: float = 0.0,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.quantiles = tuple(quantiles)
        self.task = task.lower()

        # 1. Variable selection
        self.vsn = VariableSelectionNetwork(n_features, hidden, dropout=dropout)

        # 2. LSTM temporal encoder
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.post_lstm_grn = GatedResidualNetwork(hidden, hidden, dropout=dropout)

        # 3. Causal Multi-Head Self-Attention
        heads = num_heads if (hidden % num_heads == 0 and hidden >= num_heads) else 1
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True,
            dropout=dropout,
        )
        self.post_attn_grn = GatedResidualNetwork(hidden, hidden, dropout=dropout)

        # 4. Output head
        if self.task == "quantile":
            self.head = nn.Linear(hidden, len(self.quantiles))
        elif self.task == "classification":
            self.head = nn.Linear(hidden, 1)
        else:
            self.head = nn.Linear(hidden, 1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates model representations across all sequence time steps [B, T, out_dim].

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, F].
        """
        selected, _ = self.vsn(x)
        lstm_out, _ = self.lstm(selected)
        lstm_repr = self.post_lstm_grn(lstm_out + selected)

        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )

        attn_out, _ = self.mha(
            lstm_repr,
            lstm_repr,
            lstm_repr,
            attn_mask=causal_mask,
            need_weights=False,
        )
        attn_repr = self.post_attn_grn(attn_out + lstm_repr)
        return self.head(attn_repr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting targets for the final sequence time step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, F].

        Returns
        -------
        torch.Tensor
            Output predictions of shape [B, len(quantiles)] or [B, 1].
        """
        seq_out = self.forward_sequence(x)
        return seq_out[:, -1, :]

    def variable_importance(self, x: Optional[torch.Tensor] = None) -> np.ndarray:
        """Extracts normalized variable importance weights across input features.

        Parameters
        ----------
        x : torch.Tensor, optional
            Evaluation batch of shape [B, T, F]. If None, returns weights from the
            most recent forward pass or uniform defaults.

        Returns
        -------
        np.ndarray
            Importance array of shape [F] summing to 1.0.
        """
        if x is not None:
            self.eval()
            with torch.no_grad():
                _, weights = self.vsn(x)
            imp = weights.mean(dim=(0, 1)).cpu().numpy()
            total = float(np.sum(imp))
            if total > 0:
                return (imp / total).astype(np.float32)
            return imp.astype(np.float32)

        if self.vsn.last_weights is not None:
            imp = self.vsn.last_weights.mean(dim=(0, 1)).cpu().numpy()
            total = float(np.sum(imp))
            if total > 0:
                return (imp / total).astype(np.float32)
            return imp.astype(np.float32)

        return np.full((self.n_features,), 1.0 / self.n_features, dtype=np.float32)
