"""
Common utilities, dataset formatting, time-series scaling, and training loop for PyTorch sequence models.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def make_sequences(
    features_df: Union[pd.DataFrame, np.ndarray],
    target: Union[pd.Series, np.ndarray],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates causal sliding-window sequences for time-series forecasting.

    For each sample i (where 0 <= i < N = T - window):
        Input X[i] contains feature rows from index i up to i + window - 1.
        Target y[i] is the target observation at index i + window (strictly the bar AFTER the window).

    Parameters
    ----------
    features_df : pd.DataFrame or np.ndarray
        Historical feature matrix of shape [T, F].
    target : pd.Series or np.ndarray
        Target labels or values of shape [T] or [T, H].
    window : int
        Lookback sequence length in bars.

    Returns
    -------
    tuple of (X, y)
        X : np.ndarray of shape [N, window, F], dtype float32.
        y : np.ndarray of shape [N] or [N, H], dtype float32.
    """
    if isinstance(features_df, pd.DataFrame):
        X_raw = features_df.values.astype(np.float32)
    else:
        X_raw = np.asarray(features_df, dtype=np.float32)

    if isinstance(target, (pd.Series, pd.DataFrame)):
        y_raw = target.values.astype(np.float32)
    else:
        y_raw = np.asarray(target, dtype=np.float32)

    T_len, F_dim = X_raw.shape
    if T_len <= window:
        raise ValueError(f"Sequence length {T_len} must be strictly greater than window {window}")

    N = T_len - window
    X = np.empty((N, window, F_dim), dtype=np.float32)
    y = np.empty((N,) + y_raw.shape[1:], dtype=np.float32)

    for i in range(N):
        X[i] = X_raw[i : i + window]
        y[i] = y_raw[i + window]

    return X, y


class StandardScalerTS:
    """Time-series feature standardizer strictly fitted on historical training data only."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> StandardScalerTS:
        """Computes mean and standard deviation per feature across samples and time steps.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape [N, window, F] or [N, F].

        Returns
        -------
        StandardScalerTS
            Fitted scaler.
        """
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 3:
            # Aggregate across sample and sequence dimensions
            self.mean_ = np.mean(arr, axis=(0, 1), keepdims=True)
            self.std_ = np.std(arr, axis=(0, 1), keepdims=True)
        elif arr.ndim == 2:
            self.mean_ = np.mean(arr, axis=0, keepdims=True)
            self.std_ = np.std(arr, axis=0, keepdims=True)
        else:
            self.mean_ = np.array([np.mean(arr)], dtype=np.float32)
            self.std_ = np.array([np.std(arr)], dtype=np.float32)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms input array using training statistics.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Standardized array.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScalerTS is not fitted yet.")
        arr = np.asarray(X, dtype=np.float32)
        return (arr - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SeqDataset(Dataset):
    """PyTorch Dataset holding sequence tensor pairs (X, y)."""

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> None:
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X.astype(np.float32))
        else:
            self.X = X.float()

        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y.astype(np.float32))
        else:
            self.y = y.float()

        if len(self.X) != len(self.y):
            raise ValueError(f"Length mismatch: len(X)={len(self.X)} != len(y)={len(self.y)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    epochs: int = 25,
    lr: float = 0.01,
    patience: int = 5,
    device: str = "cpu",
    seed: int = 42,
    batch_size: int = 32,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Trains a PyTorch sequence model with reproducible seeding, gradient clipping,
    and early stopping that restores the best validation weights.

    Parameters
    ----------
    model : nn.Module
        PyTorch neural network.
    train_ds : Dataset
        Training dataset.
    val_ds : Dataset
        Validation dataset.
    loss_fn : callable, optional
        Loss function (defaults to MSELoss).
    epochs : int, default 25
        Maximum training epochs.
    lr : float, default 0.01
        Learning rate for AdamW optimizer.
    patience : int, default 5
        Early stopping patience epochs.
    device : str, default 'cpu'
        Target compute device.
    seed : int, default 42
        Random seed for reproducibility.
    batch_size : int, default 32
        Mini-batch size.
    grad_clip : float, default 1.0
        Maximum gradient norm for clipping.

    Returns
    -------
    dict
        Training summary with history, best_val_loss, and best_epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.to(device)
    criterion = loss_fn if loss_fn is not None else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        # 1. Training loop
        model.train()
        train_losses: list[float] = []
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            out = model(x_b)

            # Match target dimensions if needed
            if out.shape != y_b.shape and out.squeeze().shape == y_b.squeeze().shape:
                loss = criterion(out.squeeze(), y_b.squeeze())
            else:
                loss = criterion(out, y_b)

            loss.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # 2. Validation loop
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                out = model(x_b)
                if out.shape != y_b.shape and out.squeeze().shape == y_b.squeeze().shape:
                    val_loss = criterion(out.squeeze(), y_b.squeeze())
                else:
                    val_loss = criterion(out, y_b)
                val_losses.append(val_loss.item())

        mean_val_loss = float(np.mean(val_losses)) if val_losses else mean_train_loss

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)

        # Early stopping check
        if mean_val_loss < best_val_loss - 1e-4:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best validation weights
    model.load_state_dict(best_weights)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def predict(
    model: nn.Module,
    ds: Dataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Generates out-of-sample predictions in eval mode without gradient tracking."""
    model.eval()
    model.to(device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for x_b, _ in loader:
            x_b = x_b.to(device)
            out = model(x_b)
            preds.append(out.cpu().numpy())

    if not preds:
        return np.array([])
    return np.concatenate(preds, axis=0)
