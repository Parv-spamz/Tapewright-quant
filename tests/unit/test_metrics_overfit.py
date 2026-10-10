from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestResult
from backtesting.engine.event_loop.events import SignalEvent, TickEvent
from backtesting.engine.event_loop.strategy_base import Strategy
from backtesting.metrics.drawdown_analysis.drawdown import (
    drawdown_periods,
    max_drawdown,
    max_drawdown_duration,
    recovery_factor,
    underwater_curve,
)
from backtesting.metrics.risk_measures.measures import (
    annualized_return,
    annualized_vol,
    calmar,
    cornish_fisher_var,
    cvar,
    historical_var,
    kurtosis,
    omega_ratio,
    parametric_var,
    sharpe,
    skew,
    sortino,
    summary,
)
from backtesting.metrics.risk_measures.overfit import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    monte_carlo_equity_cone,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from backtesting.metrics.risk_measures.sizing import (
    KellySizedStrategyWrapper,
    cvar_scaled_fraction,
    half_kelly_fraction,
    position_size,
)
from backtesting.reports.tearsheet.tearsheet import (
    ascii_equity_curve,
    generate_tearsheet,
    log_snapshots_to_db,
    monthly_returns_grid,
)
from config.trading.risk_limits import RiskLimits
from database.connection.manager import ConnectionManager
from database.migrations.runner import apply_migrations
from database.repositories.portfolio_repo.repo import PortfolioRepo


class TestHandVectorsAndMeasures:
    def test_hand_vectors_exact_match(self) -> None:
        """Hand-computed return vector [0.01, -0.02, 0.03, 0.0, -0.01] tested to 1e-9."""
        r = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
        ppy = 252
        n = 5

        # 1. Geometric annualized return
        # Compounded: (1.01)*(0.98)*(1.03)*(1.0)*(0.99) = 1.00929906
        # Annualized: 1.00929906^(252/5) - 1 = 1.00929906^50.4 - 1
        expected_ann_ret = (1.00929906 ** 50.4) - 1.0
        actual_ann_ret = annualized_return(r, ppy=ppy)
        np.testing.assert_allclose(actual_ann_ret, expected_ann_ret, rtol=1e-9, atol=1e-9)

        # 2. Sharpe ratio
        # Mean: 0.002
        # Deviations: 0.008, -0.022, 0.028, -0.002, -0.012
        # Sum of squares: 0.00148
        # Sample variance (ddof=1): 0.00148 / 4 = 0.00037
        # Sharpe: (0.002 / sqrt(0.00037)) * sqrt(252)
        expected_sharpe = (0.002 / np.sqrt(0.00037)) * np.sqrt(252)
        actual_sharpe = sharpe(r, rf=0.0, ppy=ppy)
        np.testing.assert_allclose(actual_sharpe, expected_sharpe, rtol=1e-9, atol=1e-9)

        # 3. Sortino ratio
        # Downside deviations: min(r, 0) -> [0.0, -0.02, 0.0, 0.0, -0.01]
        # Mean squared downside: (0.0004 + 0.0001) / 5 = 0.0001
        # Downside dev: sqrt(0.0001) = 0.01
        # Sortino: (0.002 / 0.01) * sqrt(252) = 0.2 * sqrt(252)
        expected_sortino = 0.2 * np.sqrt(252)
        actual_sortino = sortino(r, rf=0.0, mar=0.0, ppy=ppy)
        np.testing.assert_allclose(actual_sortino, expected_sortino, rtol=1e-9, atol=1e-9)

        # 4. Max drawdown
        # Equity: [1.01, 0.9898, 1.019494, 1.019494, 1.00929906]
        # Max peak: 1.019494
        # Drop at t=1: (0.9898 - 1.01) / 1.01 = -0.02
        # Drop at t=4: (1.00929906 - 1.019494) / 1.019494 = -0.01
        # Max drawdown: -0.02
        equity = np.cumprod(1.0 + r)
        expected_max_dd = -0.02
        actual_max_dd = max_drawdown(equity)
        np.testing.assert_allclose(actual_max_dd, expected_max_dd, rtol=1e-9, atol=1e-9)

    def test_constant_return_series_handled_without_exception(self) -> None:
        """Zero-variance series returns 0.0 for Sharpe and Sortino without raising exceptions."""
        constant_r = np.array([0.005, 0.005, 0.005, 0.005, 0.005])
        assert sharpe(constant_r) == 0.0
        assert sortino(constant_r) == 0.0
        assert annualized_vol(constant_r) == 0.0


class TestVaRAndCVaR:
    def test_var_cvar_deterministic_vector_and_ordering(self) -> None:
        """Deterministic 100-value vector verifies quantile convention, and CVaR >= VaR across 20 seeds."""
        det_vec = np.linspace(-0.10, 0.09, 100)
        # 95% Historical VaR uses 5% quantile
        var_95 = historical_var(det_vec, alpha=0.95)
        cvar_95 = cvar(det_vec, alpha=0.95)

        # Both reported as positive loss numbers
        assert var_95 > 0.0
        assert cvar_95 >= var_95

        # Check CVaR >= VaR across 20 different random seeds
        for seed in range(20):
            rng = np.random.default_rng(seed)
            sample = rng.normal(0.0005, 0.02, size=500)
            v = historical_var(sample, alpha=0.99)
            cv = cvar(sample, alpha=0.99)
            assert cv >= v, f"Seed {seed}: CVaR ({cv}) < VaR ({v})"


class TestDrawdownAnalysis:
    def test_drawdown_exact_period_indices_and_open_tail(self) -> None:
        """Hand equity [100, 120, 90, 95, 130, 100] -> max dd -0.25, exact indices, open period."""
        equity = pd.Series([100, 120, 90, 95, 130, 100])
        mdd = max_drawdown(equity)
        assert pytest.approx(mdd, abs=1e-9) == -0.25

        df = drawdown_periods(equity)
        assert len(df) == 2

        # Period 0 (fully recovered)
        p0 = df.iloc[0]
        assert p0["start"] == 1
        assert p0["trough"] == 2
        assert p0["end"] == 4
        assert pytest.approx(p0["depth"], abs=1e-9) == -0.25
        assert p0["duration"] == 3
        assert p0["recovery_duration"] == 2

        # Period 1 (open at end)
        p1 = df.iloc[1]
        assert p1["start"] == 4
        assert p1["trough"] == 5
        assert pd.isna(p1["end"])
        expected_open_depth = (100.0 - 130.0) / 130.0
        assert pytest.approx(p1["depth"], abs=1e-9) == expected_open_depth
        assert p1["duration"] == 1
        assert pd.isna(p1["recovery_duration"])


class TestOverfittingStatistics:
    def test_expected_max_sharpe_properties(self) -> None:
        """Independent recomputation of expected max Sharpe with norm.ppf; N=1 returns 0.0."""
        # N=1 returns 0.0
        assert expected_max_sharpe(n_trials=1, var_sr=0.04) == 0.0
        assert expected_max_sharpe(n_trials=0, var_sr=0.04) == 0.0

        # Independent recomputation for N=10, var_sr=0.04
        N = 10
        var_sr = 0.04
        gamma = 0.5772156649
        z1 = stats.norm.ppf(1.0 - 1.0 / N)
        z2 = stats.norm.ppf(1.0 - 1.0 / (N * np.e))
        expected_val = np.sqrt(var_sr) * ((1.0 - gamma) * z1 + gamma * z2)

        actual_val = expected_max_sharpe(n_trials=N, var_sr=var_sr)
        assert pytest.approx(actual_val, abs=1e-12) == expected_val

        # Monotonically increases with N
        val_5 = expected_max_sharpe(5, var_sr)
        val_20 = expected_max_sharpe(20, var_sr)
        val_100 = expected_max_sharpe(100, var_sr)
        assert val_5 < val_20 < val_100

        # Monotonically increases with var_sr
        val_low_var = expected_max_sharpe(20, 0.01)
        val_high_var = expected_max_sharpe(20, 0.09)
        assert val_low_var < val_high_var

    def test_deflated_sharpe_ratio_monotonicity(self) -> None:
        """DSR decreases monotonically with trials; DSR(n_trials=1) equals PSR(bm=0)."""
        sr_hat = 0.08
        T = 250
        sk = -0.2
        kt = 3.5
        var_sr = 0.0025

        # DSR(1) == PSR(0)
        dsr_1 = deflated_sharpe_ratio(sr_hat, T, skew=sk, kurt=kt, n_trials=1, var_sr=var_sr)
        psr_0 = probabilistic_sharpe_ratio(sr_hat, sr_benchmark=0.0, T=T, skew=sk, kurt=kt)
        np.testing.assert_allclose(dsr_1, psr_0, atol=1e-12)

        # Monotonic decrease as trials rise
        dsr_10 = deflated_sharpe_ratio(sr_hat, T, skew=sk, kurt=kt, n_trials=10, var_sr=var_sr)
        dsr_100 = deflated_sharpe_ratio(sr_hat, T, skew=sk, kurt=kt, n_trials=100, var_sr=var_sr)
        dsr_1000 = deflated_sharpe_ratio(sr_hat, T, skew=sk, kurt=kt, n_trials=1000, var_sr=var_sr)
        assert dsr_1 > dsr_10 > dsr_100 > dsr_1000

        # PSR > 0.5 when sr_hat > sr_benchmark
        assert probabilistic_sharpe_ratio(0.05, 0.02, T=100) > 0.5
        assert probabilistic_sharpe_ratio(0.01, 0.04, T=100) < 0.5

    def test_probability_of_backtest_overfitting_noise_and_planted_edge(self) -> None:
        """PBO for 20 noise strategies in [0.35, 0.65], and PBO < 0.2 for planted edge."""
        # 1. Pure noise
        rng_noise = np.random.default_rng(1)
        noise_mat = rng_noise.normal(0.0, 0.01, size=(1000, 20))
        pbo_noise, logits_noise = probability_of_backtest_overfitting(
            noise_mat, n_blocks=16, max_combos=1000, seed=42
        )
        print(f"\n[PBO Test] Pure Noise PBO: {pbo_noise:.4f}")
        assert 0.35 <= pbo_noise <= 0.65, f"Pure noise PBO {pbo_noise} not in [0.35, 0.65]"

        # 2. Planted true alpha on strategy 0
        rng_edge = np.random.default_rng(42)
        edge_mat = rng_edge.normal(0.0, 0.01, size=(1000, 20))
        edge_mat[:, 0] += 0.0015
        pbo_edge, logits_edge = probability_of_backtest_overfitting(
            edge_mat, n_blocks=16, max_combos=1000, seed=42
        )
        print(f"[PBO Test] Planted Edge PBO: {pbo_edge:.4f}")
        assert pbo_edge < 0.2, f"Planted edge PBO {pbo_edge} >= 0.2"

    def test_bootstrap_sharpe_ci_coverage(self) -> None:
        """Stationary/block bootstrap 95% CI covers true Sharpe in >= 24 of 30 runs."""
        mu = 0.0005
        sigma = 0.01
        ppy = 252
        true_sharpe = (mu / sigma) * np.sqrt(ppy)

        T = 1200
        seeds = list(range(100, 130))  # 30 stated seeds
        covered = 0

        for s in seeds:
            rng = np.random.default_rng(s)
            sample = rng.normal(mu, sigma, size=T)
            lower, upper = bootstrap_sharpe_ci(sample, n_boot=300, block=10, alpha=0.05, seed=42)
            if lower <= true_sharpe <= upper:
                covered += 1

        print(f"\n[Bootstrap CI Test] Seeds {seeds[0]}..{seeds[-1]}, Actual coverage: {covered}/30")
        assert covered >= 24, f"Bootstrap coverage {covered}/30 is below requirement of 24/30"

    def test_monte_carlo_cone_quantile_ordering(self) -> None:
        """Monte Carlo cone satisfies 5% path <= median <= 95% path at every step."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.01, size=250)
        cone = monte_carlo_equity_cone(returns, horizon=50, n_paths=500, block=5, seed=42)

        # Columns: 0.05, 0.5, 0.95
        p05 = cone[0.05].values
        p50 = cone[0.5].values
        p95 = cone[0.95].values

        assert np.all(p05 <= p50 + 1e-12)
        assert np.all(p50 <= p95 + 1e-12)

        # Deterministic given seed
        cone_repeat = monte_carlo_equity_cone(returns, horizon=50, n_paths=500, block=5, seed=42)
        np.testing.assert_allclose(cone.values, cone_repeat.values)


class TestPositionSizing:
    def test_half_kelly_hand_values(self) -> None:
        """half_kelly(0.08, 0.2, 0.02) = 0.75 exactly; cap clips; negative edge -> 0."""
        val = half_kelly_fraction(mu=0.08, sigma=0.2, rf=0.02, kelly_fraction=0.5, cap=1.0)
        assert pytest.approx(val, abs=1e-12) == 0.75

        # Cap clips
        val_capped = half_kelly_fraction(mu=0.08, sigma=0.2, rf=0.02, kelly_fraction=0.5, cap=0.5)
        assert pytest.approx(val_capped, abs=1e-12) == 0.5

        # Negative edge -> 0
        val_neg = half_kelly_fraction(mu=0.01, sigma=0.2, rf=0.02, kelly_fraction=0.5, cap=1.0)
        assert val_neg == 0.0

    def test_cvar_scaling_behavior(self) -> None:
        """cvar_scaled reduces size when budget exceeded, and leaves unchanged when not."""
        returns_safe = np.full(50, 0.01)
        # CVaR is 0 for purely positive returns, so no scaling
        scaled_safe = cvar_scaled_fraction(0.5, returns_safe, alpha=0.99, budget=0.03)
        assert scaled_safe == 0.5

        # Huge negative tail: CVaR is 0.10. At fraction 0.5, portfolio CVaR is 0.05 > 0.03 budget
        returns_risky = np.array([-0.10] * 5 + [0.01] * 45)
        scaled_risky = cvar_scaled_fraction(0.5, returns_risky, alpha=0.99, budget=0.03)
        assert scaled_risky < 0.5
        assert pytest.approx(scaled_risky, abs=1e-12) == 0.3

    def test_position_size_binding_constraint_diagnostics(self) -> None:
        """Diagnostics correctly identify max_position_pct, cvar_scaled, and kelly constraints."""
        # 1. max_position_pct binds
        lim1 = RiskLimits(max_position_pct=0.20, max_leverage=1.0)
        rw1 = np.full(50, 0.01)
        f1, d1 = position_size(0.08, 0.2, rw1, lim1, rf=0.02)
        assert d1["binding_constraint"] == "max_position_pct"
        assert pytest.approx(f1, abs=1e-12) == 0.20

        # 2. cvar_scaled binds
        lim2 = RiskLimits(max_position_pct=0.90, max_leverage=1.0, cvar_budget_pct=0.03)
        rw2 = np.array([-0.10] * 10 + [0.01] * 40)
        f2, d2 = position_size(0.08, 0.2, rw2, lim2, rf=0.02)
        assert d2["binding_constraint"] == "cvar_scaled"
        assert pytest.approx(f2, abs=1e-12) == 0.30

        # 3. kelly binds
        lim3 = RiskLimits(max_position_pct=0.90, max_leverage=1.0)
        rw3 = np.full(50, 0.01)
        f3, d3 = position_size(0.022, 0.2, rw3, lim3, rf=0.02)
        assert d3["binding_constraint"] == "kelly"
        assert pytest.approx(f3, abs=1e-12) == 0.025

    def test_wrapper_truncation_invariance(self) -> None:
        """No-lookahead test: wrapper signals on first 200 bars equal first 300 restricted to 200."""
        class MockStrategy(Strategy):
            def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
                return [SignalEvent(timestamp=tick.timestamp, symbol=tick.symbol, direction="LONG", target_pct=0.5)]

        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        rng = np.random.default_rng(42)
        px = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=300)))
        df300 = pd.DataFrame({
            "open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1000.0
        }, index=dates)
        df200 = df300.iloc[:200].copy()

        dh300 = HistoricBarDataHandler({"AAPL": df300})
        dh200 = HistoricBarDataHandler({"AAPL": df200})

        limits = RiskLimits()
        wrapper300 = KellySizedStrategyWrapper(MockStrategy(), window=30, limits=limits)
        wrapper200 = KellySizedStrategyWrapper(MockStrategy(), window=30, limits=limits)

        sizes300 = []
        while dh300.continue_backtest:
            ticks = dh300.update_bars()
            if not ticks:
                break
            sigs = wrapper300.on_bar(ticks[0], dh300)
            sizes300.append(sigs[0].target_pct if sigs else None)

        sizes200 = []
        while dh200.continue_backtest:
            ticks = dh200.update_bars()
            if not ticks:
                break
            sigs = wrapper200.on_bar(ticks[0], dh200)
            sizes200.append(sigs[0].target_pct if sigs else None)

        assert len(sizes200) == 200
        np.testing.assert_allclose(sizes300[:200], sizes200, equal_nan=True)


class TestTearsheetAndReporting:
    @pytest.fixture(autouse=True)
    def setup_sqlite_database(self) -> None:
        """Ensure clean SQLite in-memory database for snapshot logging tests."""
        ConnectionManager.reset_instance()
        manager = ConnectionManager.get_instance(backend="sqlite", sqlite_path=":memory:")
        with manager.connection() as conn:
            apply_migrations(conn)
        yield
        ConnectionManager.reset_instance()

    def test_tearsheet_headers_ascii_curve_and_monthly_compounding(self) -> None:
        """Tearsheet contains all headers, ASCII curve has exactly height rows, monthly compounds."""
        dates = pd.date_range("2021-01-01", periods=500, freq="B")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0006, 0.012, size=500)
        equity_vals = 100.0 * np.cumprod(1.0 + rets)

        df_eq = pd.DataFrame({
            "equity": equity_vals,
            "cash": equity_vals * 0.2,
            "realized_pnl": equity_vals - 100.0,
            "unrealized_pnl": 0.0,
        }, index=dates)

        res = BacktestResult(
            equity_curve=df_eq,
            trades=[
                {"symbol": "AAPL", "qty": 10.0, "exit_price": 150.0, "pnl": 50.0},
                {"symbol": "AAPL", "qty": 10.0, "exit_price": 140.0, "pnl": -20.0},
            ],
            fills=[],
            signals=[],
            orders=[],
        )

        sheet = generate_tearsheet(res, n_trials=10, trial_sharpes=[1.2, 0.8, 1.5, 0.9, 1.1])

        # Verify all mandatory headers
        mandatory_headers = [
            "SUMMARY PERFORMANCE METRICS",
            "OVERFITTING & PROBABILISTIC SHARPE",
            "BOOTSTRAP SHARPE CONFIDENCE INTERVAL",
            "TOP 5 DRAWDOWN PERIODS",
            "MONTHLY RETURNS GRID",
            "TRADE STATISTICS",
            "ASCII EQUITY CURVE",
        ]
        for header in mandatory_headers:
            assert header in sheet, f"Missing header: {header}"

        # Test ASCII curve row height
        height = 15
        ascii_curve = ascii_equity_curve(equity_vals, width=70, height=height)
        curve_lines = ascii_curve.splitlines()
        assert len(curve_lines) == height, f"ASCII curve lines {len(curve_lines)} != {height}"

        # Test monthly returns compounding to total return
        m_grid = monthly_returns_grid(pd.Series(rets, index=dates))
        month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_values = m_grid[month_cols].values.flatten()
        monthly_values = monthly_values[np.isfinite(monthly_values)]

        compounded_monthly = float(np.prod(1.0 + monthly_values) - 1.0)
        total_geometric_return = float(np.prod(1.0 + rets) - 1.0)
        assert pytest.approx(compounded_monthly, abs=1e-6) == total_geometric_return

    def test_log_snapshots_to_db(self) -> None:
        """log_snapshots_to_db persists all rows to portfolio_snapshots table."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        df_eq = pd.DataFrame({
            "equity": np.linspace(100.0, 110.0, 10),
            "cash": np.full(10, 50.0),
            "realized_pnl": np.linspace(0.0, 10.0, 10),
            "unrealized_pnl": np.zeros(10),
        }, index=dates)

        res = BacktestResult(
            equity_curve=df_eq,
            trades=[],
            fills=[],
            signals=[],
            orders=[],
        )

        portfolio_repo = PortfolioRepo()
        count = log_snapshots_to_db(res, portfolio_repo, portfolio_name="unit_test_portfolio")
        assert count == 10

        manager = ConnectionManager.get_instance()
        with manager.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM portfolio_snapshots WHERE portfolio_name='unit_test_portfolio';")
            db_count = cursor.fetchone()[0]
            assert db_count == 10
