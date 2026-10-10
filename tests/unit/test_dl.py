"""
Unit tests for deep learning sequence models (LSTM, WaveNet, SimpleTFT, common utilities).
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from models.deep_learning.common import (
    SeqDataset,
    StandardScalerTS,
    make_sequences,
    predict,
    train_model,
)
from models.deep_learning.lstm import LSTMForecaster
from models.deep_learning.temporal_fusion_transformer import (
    GatedResidualNetwork,
    SimpleTFT,
    VariableSelectionNetwork,
    quantile_loss,
    sort_quantiles,
)
from models.deep_learning.wavenet import CausalConv1d, WaveNetForecaster


def test_make_sequences_alignment() -> None:
    """Tests make_sequences alignment on a hand frame: first sample's target is strictly
    the value at row window.
    """
    df = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    target = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    window = 3

    X, y = make_sequences(df, target, window=window)

    # N = 6 - 3 = 3
    assert X.shape == (3, 3, 2)
    assert y.shape == (3,)

    # First sample: uses rows 0, 1, 2
    np.testing.assert_allclose(X[0, 0, :], [1.0, 10.0])
    np.testing.assert_allclose(X[0, 1, :], [2.0, 20.0])
    np.testing.assert_allclose(X[0, 2, :], [3.0, 30.0])

    # First sample's target MUST be the value at row `window` (row 3: 400.0)
    assert y[0] == target.iloc[window]
    assert y[0] == 400.0

    # Second sample's target is row 4 (500.0)
    assert y[1] == target.iloc[window + 1]
    assert y[1] == 500.0


def test_standard_scaler_ts_train_only() -> None:
    """Tests that StandardScalerTS is fitted on training data only and normalizes properly."""
    np.random.seed(42)
    # Train set with mean 10.0 and std 2.0
    X_train = np.random.normal(loc=10.0, scale=2.0, size=(100, 10, 3)).astype(np.float32)
    # Val set shifted with mean 50.0
    X_val = np.random.normal(loc=50.0, scale=2.0, size=(50, 10, 3)).astype(np.float32)

    scaler = StandardScalerTS()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train scaled mean is ~0.0 and std is ~1.0
    train_mean = np.mean(X_train_scaled, axis=(0, 1))
    train_std = np.std(X_train_scaled, axis=(0, 1))
    np.testing.assert_allclose(train_mean, np.zeros(3), atol=1e-4)
    np.testing.assert_allclose(train_std, np.ones(3), atol=1e-2)

    # Shifted val mean should NOT be 0 (expect ~20.0)
    val_mean = np.mean(X_val_scaled, axis=(0, 1))
    assert np.all(np.abs(val_mean) > 10.0)


def test_causality_wavenet_and_attention() -> None:
    """Tests causality: perturbing inputs at time t+k does NOT change outputs at times <= t.

    Also tests that receptive_field matches a hand computation.
    """
    torch.manual_seed(42)
    B, T, F, hidden = 2, 16, 4, 16

    # 1. WaveNet causality
    wn = WaveNetForecaster(n_features=F, hidden=hidden, dilations=(1, 2, 4, 8), kernel_size=2)
    wn.eval()

    # Theoretical RF hand computation:
    # RF = 1 + sum((k - 1) * d) = 1 + (2-1)*1 + (2-1)*2 + (2-1)*4 + (2-1)*8 = 1 + 1 + 2 + 4 + 8 = 16
    assert wn.receptive_field() == 16

    x = torch.randn(B, T, F)
    x_pert = x.clone()
    perturb_idx = 8
    x_pert[:, perturb_idx:, :] += 10.0

    with torch.no_grad():
        seq_out_wn1 = wn.forward_sequence(x)  # [B, 1, T]
        seq_out_wn2 = wn.forward_sequence(x_pert)

    # Outputs at times < perturb_idx (i.e. times 0..7) must be completely unchanged
    diff_wn = torch.abs(seq_out_wn1[:, :, :perturb_idx] - seq_out_wn2[:, :, :perturb_idx]).max().item()
    assert diff_wn < 1e-6, f"WaveNet causality violation: diff={diff_wn}"

    # 2. SimpleTFT causal attention mask
    tft = SimpleTFT(n_features=F, hidden=hidden, num_heads=2)
    tft.eval()

    with torch.no_grad():
        seq_out_tft1 = tft.forward_sequence(x)  # [B, T, 3]
        seq_out_tft2 = tft.forward_sequence(x_pert)

    # Outputs at times < perturb_idx must be completely unchanged
    diff_tft = torch.abs(seq_out_tft1[:, :perturb_idx, :] - seq_out_tft2[:, :perturb_idx, :]).max().item()
    assert diff_tft < 1e-6, f"TFT causal attention violation: diff={diff_tft}"


def test_model_shapes_and_multi_horizon() -> None:
    """Tests output tensor shapes for LSTMForecaster, WaveNetForecaster, and SimpleTFT."""
    B, T, F = 3, 12, 5
    x = torch.randn(B, T, F)

    # 1. Multi-horizon LSTM
    lstm_single = LSTMForecaster(n_features=F, hidden=16, layers=1, horizons=1)
    assert lstm_single(x).shape == (B, 1)

    lstm_multi = LSTMForecaster(n_features=F, hidden=16, layers=2, horizons=4)
    assert lstm_multi(x).shape == (B, 4)

    # 2. WaveNet multi-horizon
    wn = WaveNetForecaster(n_features=F, hidden=16, horizons=3)
    assert wn(x).shape == (B, 3)
    assert wn.forward_sequence(x).shape == (B, 3, T)

    # 3. SimpleTFT quantile output
    tft = SimpleTFT(n_features=F, hidden=16, quantiles=(0.1, 0.5, 0.9))
    assert tft(x).shape == (B, 3)
    assert tft.forward_sequence(x).shape == (B, T, 3)


def test_pinball_loss_and_sort_quantiles() -> None:
    """Tests pinball quantile loss against a hand-computed small case and verifies sort_quantiles."""
    # Hand calculation:
    # target = 10.0, preds = [8.0, 10.0, 12.0], quantiles = (0.1, 0.5, 0.9)
    # q=0.1: error = 2.0 -> max(0.1*2, -0.9*2) = 0.2
    # q=0.5: error = 0.0 -> max(0.5*0, -0.5*0) = 0.0
    # q=0.9: error = -2.0 -> max(0.9*-2, -0.1*-2) = 0.2
    # mean = (0.2 + 0.0 + 0.2) / 3 = 0.4 / 3 = 0.1333333333...
    target = torch.tensor([[10.0]])
    preds = torch.tensor([[8.0, 10.0, 12.0]])
    loss = quantile_loss(preds, target, quantiles=(0.1, 0.5, 0.9))

    expected_loss = 0.4 / 3.0
    assert abs(loss.item() - expected_loss) < 1e-6

    # Test sort_quantiles on crossing quantiles
    unsorted_preds = torch.tensor([[15.0, 5.0, 10.0], [2.0, -1.0, 0.0]])
    sorted_preds = sort_quantiles(unsorted_preds)

    expected_sorted = torch.tensor([[5.0, 10.0, 15.0], [-1.0, 0.0, 2.0]])
    torch.testing.assert_close(sorted_preds, expected_sorted)
    assert torch.all(sorted_preds[:, 1:] >= sorted_preds[:, :-1])

    # Test sort_quantiles on numpy array
    arr_unsorted = np.array([[12.0, 2.0, 7.0]])
    arr_sorted = sort_quantiles(arr_unsorted)
    np.testing.assert_allclose(arr_sorted, np.array([[2.0, 7.0, 12.0]]))


def test_variable_importance_tft() -> None:
    """Tests variable importance extraction on SimpleTFT."""
    n_features = 4
    tft = SimpleTFT(n_features=n_features, hidden=16)
    x = torch.randn(4, 10, n_features)

    # Evaluation with input batch
    imp = tft.variable_importance(x)
    assert imp.shape == (n_features,)
    assert np.all(imp >= 0.0)
    assert abs(float(np.sum(imp)) - 1.0) < 1e-5


def test_determinism_and_early_stopping_and_roundtrip() -> None:
    """Tests determinism with same seed, early stopping best-weights restore, and roundtrip serialization."""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(60, 8, 3).astype(np.float32)
    y = np.random.randn(60, 1).astype(np.float32)

    train_ds = SeqDataset(X[:40], y[:40])
    val_ds = SeqDataset(X[40:], y[40:])

    # 1. Determinism
    torch.manual_seed(42)
    m1 = LSTMForecaster(n_features=3, hidden=8, layers=1, horizons=1)
    res1 = train_model(m1, train_ds, val_ds, epochs=10, lr=0.01, seed=42)

    torch.manual_seed(42)
    m2 = LSTMForecaster(n_features=3, hidden=8, layers=1, horizons=1)
    res2 = train_model(m2, train_ds, val_ds, epochs=10, lr=0.01, seed=42)

    assert abs(res1["best_val_loss"] - res2["best_val_loss"]) < 1e-6
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1, p2)

    # 2. Early stopping restored best weights
    assert res1["best_epoch"] <= 10
    # Current model val loss matches best_val_loss
    val_preds = predict(m1, val_ds)
    current_val_loss = float(np.mean((val_preds - y[40:]) ** 2))
    assert abs(current_val_loss - res1["best_val_loss"]) < 1e-4

    # 3. State dict serialization roundtrip
    buffer = io.BytesIO()
    torch.save(m1.state_dict(), buffer)
    buffer.seek(0)

    m_loaded = LSTMForecaster(n_features=3, hidden=8, layers=1, horizons=1)
    m_loaded.load_state_dict(torch.load(buffer))

    preds_orig = predict(m1, val_ds)
    preds_loaded = predict(m_loaded, val_ds)
    np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-6)


def test_learning_benchmark() -> None:
    """Learning test: task y = sign(mean of feature 0 over the last 5 steps).

    Verifies each model (LSTM, WaveNet, SimpleTFT) reaches validation accuracy > 0.8
    within budget (hidden <= 16, epochs <= 25).
    """
    np.random.seed(42)
    torch.manual_seed(42)

    T = 600
    n_features = 4
    window = 10

    # Synthetic data generation: feature 0 has strong temporal momentum
    X_raw = np.random.randn(T, n_features).astype(np.float32)
    for i in range(1, T):
        X_raw[i, 0] = 0.7 * X_raw[i - 1, 0] + 0.3 * np.random.randn()

    X_seq, _ = make_sequences(X_raw, np.zeros(T), window=window)
    N = len(X_seq)
    y_all = np.zeros(N, dtype=np.float32)
    for i in range(N):
        # Target at bar after window: sign of mean of feature 0 over last 5 steps
        val = float(np.mean(X_raw[i + window - 5 : i + window, 0]))
        y_all[i] = 1.0 if val >= 0.0 else 0.0

    n_train = 400
    X_train, y_train = X_seq[:n_train], y_all[:n_train]
    X_val, y_val = X_seq[n_train:], y_all[n_train:]

    scaler = StandardScalerTS().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_ds = SeqDataset(X_train_scaled, y_train)
    val_ds = SeqDataset(X_val_scaled, y_val)
    loss_fn = nn.BCEWithLogitsLoss()

    # 1. LSTMForecaster
    lstm = LSTMForecaster(n_features=n_features, hidden=16, layers=1, task="classification")
    train_model(lstm, train_ds, val_ds, loss_fn=loss_fn, epochs=25, lr=0.01, seed=42)
    preds_lstm = predict(lstm, val_ds).squeeze()
    acc_lstm = float(np.mean((preds_lstm >= 0.0) == (y_val >= 0.5)))
    print(f"\n[Learning Test] LSTM Val Acc: {acc_lstm:.4f}")
    assert acc_lstm > 0.8, f"LSTM accuracy failed threshold: {acc_lstm:.4f}"

    # 2. WaveNetForecaster
    wn = WaveNetForecaster(n_features=n_features, hidden=16, dilations=(1, 2, 4, 8), horizons=1)
    train_model(wn, train_ds, val_ds, loss_fn=loss_fn, epochs=25, lr=0.01, seed=42)
    preds_wn = predict(wn, val_ds).squeeze()
    acc_wn = float(np.mean((preds_wn >= 0.0) == (y_val >= 0.5)))
    print(f"[Learning Test] WaveNet Val Acc: {acc_wn:.4f}")
    assert acc_wn > 0.8, f"WaveNet accuracy failed threshold: {acc_wn:.4f}"

    # 3. SimpleTFT
    tft = SimpleTFT(n_features=n_features, hidden=16, task="classification")
    train_model(tft, train_ds, val_ds, loss_fn=loss_fn, epochs=25, lr=0.01, seed=42)
    preds_tft = predict(tft, val_ds).squeeze()
    acc_tft = float(np.mean((preds_tft >= 0.0) == (y_val >= 0.5)))
    print(f"[Learning Test] SimpleTFT Val Acc: {acc_tft:.4f}")
    assert acc_tft > 0.8, f"SimpleTFT accuracy failed threshold: {acc_tft:.4f}"
