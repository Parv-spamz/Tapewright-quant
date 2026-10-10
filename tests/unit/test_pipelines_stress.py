"""
Unit tests for orchestration, drift monitoring, macro stress testing, and visualization suite.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from backtesting.reports.tearsheet.figures import (
    causal_graph_figure,
    drawdown_figure,
    drift_table_frame,
    equity_figure,
    frontier_figure,
    kpi_dict,
    lob_queue_figure,
    mc_cone_figure,
    monthly_heatmap_figure,
    price_with_trades_figure,
    returns_hist_figure,
    risk_profile_radar,
    rolling_sharpe_figure,
    stress_bar_figure,
)
from backtesting.reports.tearsheet.stress import (
    BUILTIN_SCENARIOS,
    DataUnavailableError,
    Scenario,
    apply_scenario,
    estimate_betas,
    replay_historical_window,
    stress_table,
)
from data.caching.parquet_store.store import ParquetStore
from pipelines.etl.etl import run_etl
from pipelines.inference.daily import run_daily_signals
from pipelines.monitoring.drift import (
    ModelMonitor,
    feature_drift_report,
    ks_drift,
    performance_decay,
    population_stability_index,
)


class TestDriftMonitoring:
    """Tests for PSI, KS drift, and performance decay monitors."""

    def test_psi_identical_distributions(self) -> None:
        """PSI on identical distributions must be < 0.01."""
        rng = np.random.RandomState(42)
        dist = rng.normal(0.0, 1.0, 5000)
        psi = population_stability_index(dist, dist, bins=10)
        assert psi < 0.01

    def test_psi_shifted_distribution(self) -> None:
        """PSI on 2-sigma shifted distribution must flag major drift (> 0.25)."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0.0, 1.0, 5000)
        actual = rng.normal(2.0, 1.0, 5000)
        psi = population_stability_index(expected, actual, bins=10)
        assert psi > 0.25

    def test_psi_hand_2bin_toy_exact(self) -> None:
        """Exact hand-computed 2-bin toy check:
        Expected: [1, 1, 2, 2] -> P_exp = [0.5, 0.5]
        Actual: [1, 2, 2, 2] -> P_act = [0.25, 0.75]
        PSI = (0.25 - 0.5)*ln(0.25/0.5) + (0.75 - 0.5)*ln(0.75/0.5)
            = -0.25*(-0.69314718) + 0.25*(0.40546511)
            = 0.274653072167...
        """
        edges = np.array([-np.inf, 1.5, np.inf])
        expected = [1.0, 1.0, 2.0, 2.0]
        actual = [1.0, 2.0, 2.0, 2.0]
        psi = population_stability_index(expected, actual, bin_edges=edges)
        hand_expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
        np.testing.assert_allclose(psi, hand_expected, rtol=1e-6)

    def test_ks_drift_statistics(self) -> None:
        """KS drift detection on identical vs shifted samples."""
        rng = np.random.RandomState(42)
        s1 = rng.normal(0.0, 1.0, 1000)
        s2 = rng.normal(0.0, 1.0, 1000)
        s3 = rng.normal(1.5, 1.0, 1000)

        _, p_same, flag_same = ks_drift(s1, s2, alpha=0.01)
        assert flag_same is False
        assert p_same > 0.01

        _, p_diff, flag_diff = ks_drift(s1, s3, alpha=0.01)
        assert flag_diff is True
        assert p_diff < 0.01

    def test_performance_decay_triggers_on_zero_edge_second_half(self) -> None:
        """Decay flag triggers on a series whose second half has zero edge."""
        rng = np.random.RandomState(42)
        # First 150 days: strong positive drift (+0.01/day)
        h1 = 0.01 + rng.normal(0.0, 0.005, 150)
        # Second 150 days: zero edge (0.00/day)
        h2 = rng.normal(0.0, 0.005, 150)
        returns = np.concatenate([h1, h2])

        report = performance_decay(returns, window=60, decay_ratio=0.5)
        assert report["is_decayed"] is True
        assert report["full_sharpe"] > 0.0
        assert report["rolling_sharpe"] < report["full_sharpe"] * 0.5

    def test_feature_drift_report_and_model_monitor(self) -> None:
        """Verifies feature_drift_report and ModelMonitor orchestration."""
        rng = np.random.RandomState(42)
        train_df = pd.DataFrame({
            "feat_stable": rng.normal(0, 1, 1000),
            "feat_drifted": rng.normal(0, 1, 1000),
        })
        live_df = pd.DataFrame({
            "feat_stable": rng.normal(0, 1, 1000),
            "feat_drifted": rng.normal(2.5, 1, 1000),
        })

        report = feature_drift_report(train_df, live_df)
        assert "feat_drifted" in report["feature"].values
        drift_row = report.loc[report["feature"] == "feat_drifted"].iloc[0]
        assert drift_row["psi_flag"] == "major"
        assert drift_row["ks_drift"] is True or drift_row["ks_drift"] == 1

        monitor = ModelMonitor(reference_features=train_df)
        drift_df = monitor.check_drift(live_df)
        assert "feat_drifted" in drift_df["feature"].values
        drift_row_m = drift_df.loc[drift_df["feature"] == "feat_drifted"].iloc[0]
        assert drift_row_m["psi_flag"] == "major"


class TestETLAndInference:
    """Tests for idempotent ETL and causal truncation-invariant daily inference."""

    @pytest.fixture
    def temp_store(self) -> ParquetStore:
        temp_dir = tempfile.mkdtemp()
        store = ParquetStore(temp_dir)
        yield store
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_etl_idempotency(self, temp_store: ParquetStore) -> None:
        """ETL idempotent: identical row counts after a second execution."""
        r1 = run_etl(["AAPL", "MSFT"], start="2023-01-01", end="2023-06-01", store=temp_store)
        r2 = run_etl(["AAPL", "MSFT"], start="2023-01-01", end="2023-06-01", store=temp_store)

        assert r1.raw_row_counts == r2.raw_row_counts
        assert r1.feature_row_counts == r2.feature_row_counts
        for sym in ["AAPL", "MSFT"]:
            df1 = temp_store.read(sym, interval="1d")
            assert len(df1) == r1.raw_row_counts[sym]

    def test_inference_truncation_invariance(self, temp_store: ParquetStore) -> None:
        """No-lookahead: run_daily_signals(asof=t) on full history equals run on data truncated to t."""
        # Ingest full history up to 2023-08-01
        run_etl(["AAPL"], start="2023-01-01", end="2023-08-01", store=temp_store)

        asof_date = "2023-04-15"
        sig_from_full = run_daily_signals(
            symbols=["AAPL"],
            asof=asof_date,
            store=temp_store,
            strategy_names=["crossover"],
            signal_repo=None,
        )

        # Create separate store with data truncated strictly to asof_date
        temp_dir2 = tempfile.mkdtemp()
        truncated_store = ParquetStore(temp_dir2)
        try:
            run_etl(["AAPL"], start="2023-01-01", end=asof_date, store=truncated_store)
            sig_from_truncated = run_daily_signals(
                symbols=["AAPL"],
                asof=asof_date,
                store=truncated_store,
                strategy_names=["crossover"],
                signal_repo=None,
            )
            assert len(sig_from_full) == len(sig_from_truncated)
            if sig_from_full:
                assert sig_from_full[0]["symbol"] == sig_from_truncated[0]["symbol"]
                assert sig_from_full[0]["signal_type"] == sig_from_truncated[0]["signal_type"]
                np.testing.assert_allclose(
                    sig_from_full[0]["target_weight"],
                    sig_from_truncated[0]["target_weight"],
                )
        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)


class TestStressTesting:
    """Tests for macro stress scenarios, correlation overrides, and liquidation costs."""

    def test_stress_hand_calculation_exact(self) -> None:
        """Hand example:
        2 assets, weights [0.6, 0.4], betas [1.0, 0.5], equity_shock -0.40
        pnl_pct = 0.6 * 1.0 * (-0.40) + 0.4 * 0.5 * (-0.40) = -0.24 - 0.08 = -0.32 exactly.
        """
        weights = [0.6, 0.4]
        betas = [1.0, 0.5]
        scen = Scenario(
            name="HandTest",
            description="Hand shock",
            equity_shock=-0.40,
            vol_multiplier=1.0,
            correlation_override=None,
            spread_multiplier=1.0,
            adv_haircut=0.0,
            rate_shock_bp=0,
            source_note="approximate, owner-supplied, verify against your own data before citing",
        )
        res = apply_scenario(
            weights=weights,
            betas=betas,
            scenario=scen,
            equity=100_000.0,
            baseline_vol=0.15,
        )
        assert np.isclose(res.pnl_pct, -0.32)
        assert np.isclose(res.pnl_dollar, -32_000.0)

    def test_correlation_override_unity_equals_weighted_vol(self) -> None:
        """Correlation override 1.0 gives portfolio vol equal to the weighted sum of vols."""
        weights = [0.6, 0.4]
        betas = [1.0, 0.5]
        asset_vols = [0.25, 0.15]
        expected_vol = 0.6 * 0.25 + 0.4 * 0.15  # 0.15 + 0.06 = 0.21

        scen = Scenario(
            name="UnityCorr",
            description="Perfect correlation",
            equity_shock=-0.10,
            vol_multiplier=1.0,
            correlation_override=1.0,
            spread_multiplier=1.0,
            adv_haircut=0.1,
            rate_shock_bp=0,
            source_note="approximate",
        )
        res = apply_scenario(
            weights=weights,
            betas=betas,
            scenario=scen,
            asset_vols=asset_vols,
        )
        np.testing.assert_allclose(res.stressed_vol, expected_vol, rtol=1e-5)

    def test_stressed_cvar_greater_equal_var(self) -> None:
        """Stressed CVaR must always be greater than or equal to stressed VaR."""
        for _, scen in BUILTIN_SCENARIOS.items():
            res = apply_scenario(
                weights=[0.5, 0.5],
                betas=[1.0, 0.8],
                scenario=scen,
                equity=500_000.0,
                baseline_vol=0.20,
            )
            assert res.stressed_cvar_99_1d >= res.stressed_var_99_1d

    def test_liquidation_cost_increases_with_adv_haircut(self) -> None:
        """Liquidation cost increases when adv_haircut increases."""
        scen_low = Scenario(
            name="LowHaircut",
            description="Haircut 20%",
            equity_shock=-0.10,
            vol_multiplier=1.0,
            correlation_override=None,
            spread_multiplier=2.0,
            adv_haircut=0.20,
            rate_shock_bp=0,
            source_note="approximate",
        )
        scen_high = Scenario(
            name="HighHaircut",
            description="Haircut 60%",
            equity_shock=-0.10,
            vol_multiplier=1.0,
            correlation_override=None,
            spread_multiplier=2.0,
            adv_haircut=0.60,
            rate_shock_bp=0,
            source_note="approximate",
        )
        r_low = apply_scenario([1.0], [1.0], scen_low, equity=100_000.0, adv=50_000.0)
        r_high = apply_scenario([1.0], [1.0], scen_high, equity=100_000.0, adv=50_000.0)
        assert r_high.liquidation_cost > r_low.liquidation_cost
        assert r_high.days_to_liquidate > r_low.days_to_liquidate

    def test_builtin_scenarios_source_notes(self) -> None:
        """Every BUILTIN scenario has a non-empty source_note containing 'approximate'."""
        assert len(BUILTIN_SCENARIOS) >= 3
        for name, scen in BUILTIN_SCENARIOS.items():
            assert scen.source_note, f"Scenario {name} has empty source_note"
            assert "approximate" in scen.source_note.lower(), (
                f"Scenario {name} note must contain 'approximate': {scen.source_note}"
            )

    def test_replay_historical_window_raises_on_empty(self) -> None:
        """replay_historical_window raises DataUnavailableError on an empty store or dataframe."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataUnavailableError):
            replay_historical_window([1.0], empty_df, "2008-09-01", "2008-10-01")


class TestFiguresSuite:
    """Tests for Plotly visualization module invariants."""

    def test_figures_return_types_and_invariants(self) -> None:
        """Figures return go.Figure, length matching, and correct axis bounds."""
        dates = pd.date_range("2023-01-01", periods=100)
        equity = pd.Series(np.linspace(100_000, 120_000, 100), index=dates)
        orig_equity = copy.deepcopy(equity)

        # 1. Equity figure
        fig_eq = equity_figure(equity)
        assert isinstance(fig_eq, go.Figure)
        assert len(fig_eq.data[0].x) == len(equity)
        # Non-mutation guarantee
        pd.testing.assert_series_equal(equity, orig_equity)

        # 2. Drawdown figure: y <= 0
        dd_vals = pd.Series([-0.05, -0.02, 0.0, -0.01], index=dates[:4])
        orig_dd = copy.deepcopy(dd_vals)
        fig_dd = drawdown_figure(dd_vals)
        assert isinstance(fig_dd, go.Figure)
        assert all(y <= 0 for y in fig_dd.data[0].y)
        pd.testing.assert_series_equal(dd_vals, orig_dd)

        # 3. Monthly heatmap: 12 columns
        rets = pd.Series(np.random.RandomState(42).normal(0.001, 0.01, 300), index=dates[:300] if len(dates) >= 300 else pd.date_range("2023-01-01", periods=300))
        orig_rets = copy.deepcopy(rets)
        fig_hm = monthly_heatmap_figure(rets)
        assert isinstance(fig_hm, go.Figure)
        assert len(fig_hm.data[0].x) == 12
        pd.testing.assert_series_equal(rets, orig_rets)

        # 4. Price with trades: buy/sell marker counts == fills
        prices_df = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0, 103.0]},
            index=dates[:4],
        )
        fills = [
            {"side": "BUY", "timestamp": dates[0], "executed_price": 100.0},
            {"side": "BUY", "timestamp": dates[1], "executed_price": 101.0},
            {"side": "SELL", "timestamp": dates[2], "executed_price": 102.0},
        ]
        fig_trades = price_with_trades_figure(prices_df, fills)
        assert isinstance(fig_trades, go.Figure)
        buy_pts = len(fig_trades.data[1].x) if len(fig_trades.data) > 1 else 0
        sell_pts = len(fig_trades.data[2].x) if len(fig_trades.data) > 2 else 0
        assert buy_pts + sell_pts == len(fills)

        # 5. Risk radar title contains "proxy"
        metrics = {"sharpe": 1.5, "volatility": 0.14, "cvar_95": 0.03}
        fig_radar = risk_profile_radar(metrics)
        assert isinstance(fig_radar, go.Figure)
        assert "proxy" in fig_radar.layout.title.text.lower()

        # 6. Returns histogram, rolling Sharpe, MC cone, stress bar, causal graph, frontier
        fig_hist = returns_hist_figure(rets.values, var=0.02, cvar=0.03)
        assert isinstance(fig_hist, go.Figure)

        fig_rolling = rolling_sharpe_figure(rets)
        assert isinstance(fig_rolling, go.Figure)

        cone_df = pd.DataFrame({
            "step": [1, 2, 3],
            "q_0.05": [98, 97, 96],
            "q_0.5": [100, 101, 102],
            "q_0.95": [102, 104, 106],
        })
        fig_cone = mc_cone_figure(cone_df, equity)
        assert isinstance(fig_cone, go.Figure)

        s_table = stress_table([0.5, 0.5], [1.0, 1.0])
        fig_bar = stress_bar_figure(s_table)
        assert isinstance(fig_bar, go.Figure)

        fig_causal = causal_graph_figure([("A", "B", 0.5), ("C", "B", 0.8)])
        assert isinstance(fig_causal, go.Figure)

        fig_queue = lob_queue_figure([])
        assert isinstance(fig_queue, go.Figure)

        fig_front = frontier_figure([(0.1, 0.05), (0.2, 0.1)])
        assert isinstance(fig_front, go.Figure)

        # Helpers
        kpis = kpi_dict({"total_return": 0.15, "sharpe": 1.8})
        assert "Total Return" in kpis
        assert "Sharpe Ratio" in kpis
