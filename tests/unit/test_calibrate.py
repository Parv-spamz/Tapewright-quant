"""Unit tests for microstructure calibration, impact fitting, and queue simulation.

Enforces:
- Exact mathematical book walk cost on a hand-crafted 3-level order book
- Accurate recovery of planted square-root exponent (gamma = 0.50 +/- 0.05)
- Independent queue fill logic verification against hand calculations
- Imbalance IC recovers planted signal sign and direction
- Differential simulator comparison execution on synthetic order book tapes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.microstructure.calibrate import (
    book_walk_cost,
    imbalance_ic,
    impact_fit,
    queue_fill_probability,
    spread_stats,
)


def test_walk_cost_on_3_level_book_equals_hand_vwap_minus_mid() -> None:
    """book_walk_cost on a hand 3-level book strictly reproduces analytical VWAP minus mid in basis points."""
    snapshot = {
        "mid": 100.0,
        "bid_px_1": 99.0,
        "bid_sz_1": 1.0,
        "ask_px_1": 101.0,
        "ask_sz_1": 1.0,
        "ask_px_2": 102.0,
        "ask_sz_2": 2.0,
        "ask_px_3": 103.0,
        "ask_sz_3": 3.0,
    }

    # BUY size = 3.0: consumes level 1 (1.0 @ 101) + level 2 (2.0 @ 102) = 101 + 204 = 305 dollars
    # VWAP = 305 / 3 = 101.6666666667
    # (VWAP - mid) / mid * 10000 = (1.6666666667 / 100) * 10000 = 166.6666666667 bp
    cost_bp = book_walk_cost(snapshot, size=3.0, side="BUY")
    expected_bp = ((305.0 / 3.0 - 100.0) / 100.0) * 10_000.0

    assert cost_bp == pytest.approx(expected_bp, rel=1e-6)

    # SELL size = 1.0 @ bid 99.0: (100 - 99) / 100 * 10000 = 100.0 bp
    sell_cost = book_walk_cost(snapshot, size=1.0, side="SELL")
    assert sell_cost == pytest.approx(100.0, rel=1e-6)


def test_impact_fit_recovers_planted_exponent_within_tolerance() -> None:
    """impact_fit recovers a planted exponent gamma = 0.50 within +/- 0.05 on synthetic power-law books."""
    rng = np.random.default_rng(seed=42)
    n_books = 20
    records: list[dict[str, float]] = []

    sizes_grid = [0.2, 0.4, 0.6, 1.0, 1.4, 1.8, 2.0]

    for b in range(n_books):
        row: dict[str, float] = {
            "mid": 50000.0,
            "spread": 0.0,
            "bid_px_1": 50000.0,
            "bid_sz_1": 2.0,
            "ask_px_1": 50000.0,
            "ask_sz_1": 0.1,
        }
        cum = 0.0
        sz_step = 0.1
        for k in range(1, 21):
            mid_s = cum + 0.5 * sz_step
            cum += sz_step
            # Planted derivative marginal price d/ds (c * s^1.5) = 1.5 * c * s^0.5
            extra_px = 75.0 * (mid_s ** 0.5) * (1.0 + rng.normal(0.0, 0.005))
            row[f"ask_px_{k}"] = 50000.0 + extra_px
            row[f"ask_sz_{k}"] = sz_step
        records.append(row)

    book_df = pd.DataFrame(records)
    fit_res = impact_fit(book_df, sizes=sizes_grid)

    gamma = fit_res["gamma"]
    # Verify recovered exponent is within 0.05 of planted 0.50
    assert 0.45 <= gamma <= 0.55
    assert fit_res["r2"] > 0.95


def test_queue_fill_probability_on_hand_tape_equals_hand_computed_outcome() -> None:
    """queue_fill_probability independent calculation precisely matches hand-computed fill conditions."""
    ts0 = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    ts1 = ts0 + pd.Timedelta(seconds=10)
    ts2 = ts0 + pd.Timedelta(seconds=20)
    ts_end = ts0 + pd.Timedelta(seconds=40)

    book_df = pd.DataFrame([
        {
            "bid_px_1": 100.0,
            "bid_sz_1": 1.5,
            "ask_px_1": 101.0,
            "ask_sz_1": 1.0,
        },
    ], index=[ts0])

    # Case A: Aggressor sell volume (1.0 + 0.7 = 1.7) exceeds queue_ahead (1.5) + my_size (0.1) = 1.6 -> FILLED
    trades_filled = pd.DataFrame([
        {"ts": int(ts1.timestamp() * 1000), "price": 100.0, "size": 1.0, "sign": -1},
        {"ts": int(ts2.timestamp() * 1000), "price": 100.0, "size": 0.7, "sign": -1},
    ], index=[ts1, ts2])

    res_filled = queue_fill_probability(book_df, trades_filled, horizons_s=(30,), my_size=0.1)
    assert res_filled["fill_rates"]["independent"] == 1.0

    # Case B: Aggressor sell volume (1.0 + 0.4 = 1.4) does NOT exceed 1.6 -> NOT FILLED
    trades_unfilled = pd.DataFrame([
        {"ts": int(ts1.timestamp() * 1000), "price": 100.0, "size": 1.0, "sign": -1},
        {"ts": int(ts2.timestamp() * 1000), "price": 100.0, "size": 0.4, "sign": -1},
    ], index=[ts1, ts2])

    res_unfilled = queue_fill_probability(book_df, trades_unfilled, horizons_s=(30,), my_size=0.1)
    assert res_unfilled["fill_rates"]["independent"] == 0.0


def test_imbalance_ic_on_planted_signal_recovers_planted_positive_sign() -> None:
    """imbalance_ic on a synthetic series with planted positive signal recovers positive Spearman correlation."""
    n = 100
    times = pd.date_range("2026-01-01 10:00:00", periods=n, freq="1s", tz="UTC")

    # Generate synthetic book with alternating imbalance
    rng = np.random.default_rng(123)
    bid_sz = rng.uniform(1.0, 5.0, size=n)
    ask_sz = rng.uniform(1.0, 5.0, size=n)

    imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
    mid_prices = np.empty(n)
    mid_prices[0] = 50000.0
    for t in range(n - 1):
        # Mid price moves strongly in direction of imbalance
        mid_prices[t + 1] = mid_prices[t] * np.exp(0.001 * imb[t] + rng.normal(0, 0.0001))

    book_df = pd.DataFrame({
        "mid": mid_prices,
        "bid_px_1": mid_prices - 0.5,
        "ask_px_1": mid_prices + 0.5,
        "bid_sz_1": bid_sz,
        "ask_sz_1": ask_sz,
    }, index=times)

    res = imbalance_ic(book_df, horizons=(1,), levels=(1,), block_size=15, n_bootstrap=50)
    real_ic = res["real"]["level_1_h_1"]["ic"]

    assert real_ic > 0.80
    assert res["real"]["level_1_h_1"]["ci_95"][0] > 0.50


def test_real_vs_simulator_comparison_runs_on_synthetic_tapes() -> None:
    """Differential queue_fill_probability comparison runs cleanly on synthetic tapes and outputs agreement statistics."""
    n_snaps = 20
    times = pd.date_range("2026-01-01 10:00:00", periods=n_snaps, freq="2s", tz="UTC")

    book_rows = []
    for i, t in enumerate(times):
        book_rows.append({
            "mid": 50000.0,
            "bid_px_1": 49999.0,
            "bid_sz_1": 1.0,
            "ask_px_1": 50001.0,
            "ask_sz_1": 1.0,
        })
    book_df = pd.DataFrame(book_rows, index=times)

    # Trades tape
    tr_times = pd.date_range("2026-01-01 10:00:00", periods=40, freq="1s", tz="UTC")
    trade_rows = []
    for t in tr_times:
        trade_rows.append({
            "ts": int(t.timestamp() * 1000),
            "price": 49999.0,
            "size": 0.2,
            "sign": -1,
        })
    trades_df = pd.DataFrame(trade_rows, index=tr_times)

    comp = queue_fill_probability(book_df, trades_df, horizons_s=(10,), my_size=0.1)

    assert 0.0 <= comp["agreement_rate"] <= 1.0
    assert "fill_rates" in comp
    assert isinstance(comp["disagreements"], list)
