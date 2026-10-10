"""
Integration test for engine anti-overfitting audit (PBO & DSR on synthetic noise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.metrics.risk_measures.overfit import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from signals.generation.breakout.breakout import donchian_breakout
from signals.generation.crossover.crossover import dual_ema_signals


class TestOverfitAudit:
    """Audit of engine anti-overfitting statistical defense mechanisms."""

    def test_overfitting_detected_on_pure_noise_panel(self) -> None:
        """Build 30 strategy variants on pure synthetic noise (random-walk prices, seed fixed).
        Compute PBO via CSCV and the DSR of the best variant with n_trials=30.
        Assert DSR(best) < 0.95 and report PBO.
        Proves the engine detects its own overfitting when selecting across multiple trials.
        """
        np.random.seed(42)
        T = 600

        # 1. Pure synthetic random-walk noise panel
        noise_returns = np.random.normal(0.0, 0.01, T)
        prices = pd.Series(100.0 * np.exp(np.cumsum(noise_returns)))
        df_panel = pd.DataFrame({
            "close": prices,
            "high": prices * 1.002,
            "low": prices * 0.998,
        })

        # 2. Build 30 strategy variants (20 crossover parameter combinations + 10 breakout windows)
        variants_returns: list[np.ndarray] = []

        # 20 Crossover variants
        crossover_configs = [
            (fast, slow)
            for fast in [5, 10, 15, 20, 25]
            for slow in [30, 40, 50, 60]
        ]
        assert len(crossover_configs) == 20

        for fast, slow in crossover_configs:
            raw_sig = dual_ema_signals(prices, fast=fast, slow=slow)
            causal_sig = raw_sig.shift(1).fillna(0)
            strat_ret = causal_sig * prices.pct_change().fillna(0)
            variants_returns.append(strat_ret.values)

        # 10 Breakout variants
        breakout_windows = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
        assert len(breakout_windows) == 10

        for window in breakout_windows:
            raw_sig = donchian_breakout(df_panel, n=window)
            causal_sig = raw_sig.shift(1).fillna(0)
            strat_ret = causal_sig * prices.pct_change().fillna(0)
            variants_returns.append(strat_ret.values)

        assert len(variants_returns) == 30, "Exactly 30 strategy variants must be built"

        returns_matrix = np.column_stack(variants_returns)
        assert returns_matrix.shape == (T, 30)

        # 3. Compute Probability of Backtest Overfitting (PBO) via CSCV
        pbo, logits = probability_of_backtest_overfitting(
            returns_matrix,
            n_blocks=8,
            max_combos=2000,
            seed=42,
        )

        # 4. Evaluate individual strategy Sharpe ratios
        sharpes = [
            float(np.mean(r) / (np.std(r, ddof=1) + 1e-8) * np.sqrt(252))
            for r in variants_returns
        ]
        best_idx = int(np.argmax(sharpes))
        best_sr = sharpes[best_idx]
        var_sr = float(np.var(sharpes, ddof=1))

        # 5. Compute Deflated Sharpe Ratio (DSR) discounting selection bias for N=30 trials
        dsr_best = deflated_sharpe_ratio(
            sr_hat=best_sr,
            T=T,
            skew=0.0,
            kurt=3.0,
            n_trials=30,
            var_sr=var_sr,
        )

        # Log measured numbers
        print(f"\n[OVERFIT AUDIT REPORT]")
        print(f"Trials (N): 30")
        print(f"Sample Length (T): {T}")
        print(f"Best Strategy Apparent Sharpe: {best_sr:.4f}")
        print(f"Sharpe Variance Across Trials: {var_sr:.6f}")
        print(f"Deflated Sharpe Ratio (DSR): {dsr_best:.4f}")
        print(f"Probability of Backtest Overfitting (PBO): {pbo:.4f}")

        # Core assertions proving the engine flags false alpha
        assert dsr_best < 0.95, (
            f"DSR for best variant on noise ({dsr_best:.4f}) should be < 0.95 under multi-testing bias!"
        )
        assert pbo > 0.50, (
            f"PBO on noise ({pbo:.4f}) should indicate high probability of overfitting (> 0.50)!"
        )
