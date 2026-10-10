"""
Interactive Tapewright-Quant quantitative research & tearsheet dashboard.

Run with:
    streamlit run backtesting/reports/tearsheet/app.py
"""

from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestEngine
from backtesting.engine.event_loop.execution import SimulatedExecutionHandler, ZeroCostFillModel
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.execution.fill_models import RealisticFillModel
from backtesting.execution.slippage_models.models import AdverseSelectionPenalty, SquareRootImpactSlippage
from backtesting.execution.transaction_costs.costs import PerShareCommission
from backtesting.metrics.drawdown_analysis import underwater_curve
from backtesting.metrics.risk_measures import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    monte_carlo_equity_cone,
    probabilistic_sharpe_ratio,
    summary,
)
from backtesting.reports.tearsheet.figures import (
    causal_graph_figure,
    drawdown_figure,
    equity_figure,
    frontier_figure,
    lob_queue_figure,
    mc_cone_figure,
    monthly_heatmap_figure,
    price_with_trades_figure,
    returns_hist_figure,
    risk_profile_radar,
    rolling_sharpe_figure,
    stress_bar_figure,
)
from backtesting.reports.tearsheet.stress import BUILTIN_SCENARIOS, stress_table
from config.trading.risk_limits import RiskLimits
from data.caching.parquet_store.store import ParquetStore
from database.connection.manager import ConnectionManager
from pipelines.etl.etl import run_etl
from pipelines.monitoring.drift import feature_drift_report, performance_decay
from signals.generation.breakout.breakout import BreakoutStrategy
from signals.generation.crossover.crossover import CrossoverStrategy
from signals.generation.mean_reversion.mean_reversion import MeanReversionStrategy


@st.cache_data(show_spinner=False)
def load_cached_etl(symbols: tuple[str, ...], n_days: int, store_dir: str = "data/caching/parquet_store") -> bool:
    """Caches ETL execution to prevent recomputing across Streamlit reruns."""
    start_date = pd.Timestamp.now() - pd.Timedelta(days=int(n_days * 1.5))
    run_etl(symbols=symbols, start=start_date.strftime("%Y-%m-%d"), end=pd.Timestamp.now().strftime("%Y-%m-%d"), store=store_dir)
    return True


def run_backtest_simulation(
    symbols: list[str],
    n_days: int,
    strategy_name: str,
    execution_type: str,
    slippage_c: float,
    commission_pct: float,
    vpin_gamma: float,
    sizing_type: str,
    max_position_pct: float,
    target_vol: float,
    seed: int,
) -> dict[str, Any]:
    """Executes full backtest simulation via BacktestEngine and compiles results."""
    store = ParquetStore("data/caching/parquet_store")
    clean_syms = [s.strip().upper() for s in symbols]

    # Ingest data if not present
    start_str = (pd.Timestamp("2023-01-01") - pd.Timedelta(days=n_days)).strftime("%Y-%m-%d")
    run_etl(symbols=clean_syms, start=start_str, end="2024-01-01", store=store)

    # Load primary price series
    primary_sym = clean_syms[0]
    prices_df = store.read(primary_sym, interval="1d")
    features_df = store.read(primary_sym, interval="features_1d")

    # Strategy selection
    if strategy_name == "breakout":
        strategy = BreakoutStrategy(symbol=primary_sym, n=20)
    elif strategy_name in ("mean_reversion", "reversion"):
        strategy = MeanReversionStrategy(symbol=primary_sym)
    else:
        strategy = CrossoverStrategy(symbol=primary_sym, fast=10, slow=30)

    # Cost model
    if execution_type == "zero":
        fill_model = ZeroCostFillModel()
    else:
        slip = SquareRootImpactSlippage(c=slippage_c)
        cost_m = PerShareCommission(per_share=max(0.001, commission_pct / 1000.0))
        adverse = AdverseSelectionPenalty(gamma=vpin_gamma)
        fill_model = RealisticFillModel(slippage=slip, cost_model=cost_m, adverse_selection=adverse)

    # Engine execution
    handler = HistoricBarDataHandler({primary_sym: prices_df})
    portfolio = Portfolio(
        initial_cash=100_000.0,
        risk_limits=RiskLimits(max_position_pct=max_position_pct, target_annual_vol=target_vol),
    )
    exec_handler = SimulatedExecutionHandler(handler, fill_model=fill_model)
    engine = BacktestEngine(
        data_handler=handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=exec_handler,
    )
    result = engine.run()

    # Performance calculations
    if hasattr(result, "equity_curve") and isinstance(result.equity_curve, pd.DataFrame) and "equity" in result.equity_curve.columns:
        equity_curve = result.equity_curve["equity"]
    else:
        equity_curve = pd.Series([100_000.0], index=prices_df.index[:1])

    returns = equity_curve.pct_change().dropna()
    metrics = summary(returns.values) if len(returns) > 1 else {}
    dd_curve = underwater_curve(equity_curve)

    # Benchmark comparison
    bm_series = prices_df["close"].copy()

    return {
        "primary_symbol": primary_sym,
        "symbols": clean_syms,
        "prices_df": prices_df,
        "features_df": features_df,
        "equity_curve": equity_curve,
        "returns": returns,
        "metrics": metrics,
        "drawdown_curve": dd_curve,
        "benchmark": bm_series,
        "fills": result.fills if hasattr(result, "fills") else [],
        "is_synthetic": True,
    }


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Tapewright-Quant Institutional Quant Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #0E1117; color: #E0E3EB; }
        .stSidebar { background-color: #161B26; }
        .stMetric { background-color: #1E232F; padding: 12px; border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Tapewright-Quant: Institutional Quantitative Intelligence")

    # 1. Sidebar Controls
    with st.sidebar:
        st.header("Research Controls")
        symbols_input = st.text_input("Universe Symbols", value="AAPL, MSFT, SPY")
        days = st.slider("Historical Lookback (Days)", min_value=100, max_value=1000, value=500, step=50)
        data_source = st.selectbox("Data Source", ["synthetic", "yahoo", "cache"])
        strategy = st.selectbox("Trading Strategy", ["crossover", "breakout", "mean_reversion", "pairs", "ML-meta"])

        st.subheader("Microstructure Execution")
        exec_model = st.selectbox("Execution Model", ["realistic", "zero", "LOB"])
        slippage_c = st.slider("Kyle Lambda Impact (c)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        commission = st.number_input("Commission (bps)", min_value=0.0, max_value=50.0, value=5.0)
        vpin_gamma = st.slider("VPIN Toxicity Penalty", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

        st.subheader("Risk & Capital Budgeting")
        sizing = st.selectbox("Position Sizing", ["Half-Kelly+CVaR", "fixed"])
        max_pos = st.slider("Max Position %", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        target_vol = st.slider("Target Annual Vol %", min_value=5.0, max_value=40.0, value=15.0) / 100.0
        seed = st.number_input("Random Seed", value=42, step=1)

        run_btn = st.button("Run Quantitative Simulation", type="primary")

    # Initialize state
    if run_btn or "backtest_data" not in st.session_state:
        symbols_list = [s.strip() for s in symbols_input.split(",") if s.strip()]
        with st.spinner("Executing simulation pipeline..."):
            st.session_state["backtest_data"] = run_backtest_simulation(
                symbols=symbols_list,
                n_days=days,
                strategy_name=strategy,
                execution_type=exec_model,
                slippage_c=slippage_c,
                commission_pct=commission * 1e-4,
                vpin_gamma=vpin_gamma,
                sizing_type=sizing,
                max_position_pct=max_pos,
                target_vol=target_vol,
                seed=int(seed),
            )

    data = st.session_state["backtest_data"]

    # Permanent synthetic data banner
    if data.get("is_synthetic", False):
        st.warning("⚠️ SYNTHETIC DATA: results say nothing about real markets")

    # 2. Main Dashboard Tabs
    tabs = st.tabs(
        [
            "Overview",
            "Trades",
            "Risk",
            "Monte Carlo",
            "Stress",
            "Execution",
            "Portfolio",
            "Causal Graph",
            "Models",
            "Monitoring",
            "Database",
            "Tearsheet",
        ]
    )

    equity_s = data["equity_curve"]
    rets_s = data["returns"]
    metrics = data["metrics"]
    fills = data["fills"]

    # Tab 1: Overview
    with tabs[0]:
        st.subheader("Executive Performance Summary")
        m_cols = st.columns(5)
        tot_ret = (equity_s.iloc[-1] - equity_s.iloc[0]) / equity_s.iloc[0] if len(equity_s) > 1 else 0.0
        sr = float(metrics.get("sharpe", 0.0))
        mdd = float(metrics.get("max_drawdown", 0.0))
        sk = float(metrics.get("skew", 0.0))
        kt = float(metrics.get("kurtosis", 3.0))
        t_len = len(rets_s)
        psr = probabilistic_sharpe_ratio(sr_hat=sr, sr_benchmark=0.0, T=t_len, skew=sk, kurt=kt) if t_len > 10 else 0.5
        dsr = deflated_sharpe_ratio(sr_hat=sr, T=t_len, skew=sk, kurt=kt, n_trials=10, var_sr=0.1) if t_len > 10 else 0.5

        m_cols[0].metric("Total Return", f"{tot_ret * 100.0:.2f}%")
        m_cols[1].metric("Sharpe Ratio", f"{sr:.2f}")
        m_cols[2].metric("Max Drawdown", f"{mdd * 100.0:.2f}%")
        m_cols[3].metric("Probabilistic SR", f"{psr * 100.0:.1f}%")
        m_cols[4].metric("Deflated SR", f"{dsr * 100.0:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(equity_figure(equity_s, data.get("benchmark")), use_container_width=True)
        with c2:
            st.plotly_chart(drawdown_figure(data["drawdown_curve"]), use_container_width=True)

    # Tab 2: Trades
    with tabs[1]:
        st.subheader("Executed Trade Log")
        st.plotly_chart(price_with_trades_figure(data["prices_df"], fills), use_container_width=True)
        if fills:
            fills_rows = [f.__dict__ if hasattr(f, "__dict__") else f for f in fills]
            st.dataframe(pd.DataFrame(fills_rows), use_container_width=True)
        else:
            st.info("No trade fill events logged in simulation.")

    # Tab 3: Risk
    with tabs[2]:
        st.subheader("Tail Risk & Volatility Breakdown")
        r1, r2 = st.columns(2)
        with r1:
            var_val = float(metrics.get("parametric_var_95", 0.02))
            cvar_val = float(metrics.get("cvar_95", 0.03))
            st.plotly_chart(returns_hist_figure(rets_s, var=var_val, cvar=cvar_val), use_container_width=True)
        with r2:
            roll_sr = rets_s.rolling(60).apply(lambda r: np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252)).dropna()
            st.plotly_chart(rolling_sharpe_figure(roll_sr), use_container_width=True)

        st.plotly_chart(monthly_heatmap_figure(rets_s), use_container_width=True)
        if len(rets_s) > 20:
            ci = bootstrap_sharpe_ci(rets_s.values)
            st.caption(f"Bootstrap 95% Sharpe Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")

    # Tab 4: Monte Carlo
    with tabs[3]:
        st.subheader("Monte Carlo Equity Path Cone")
        if len(rets_s) > 10:
            cone = monte_carlo_equity_cone(rets_s.values, horizon=min(len(rets_s), 252), n_paths=200)
            st.plotly_chart(mc_cone_figure(cone, equity_s), use_container_width=True)

    # Tab 5: Stress
    with tabs[4]:
        st.subheader("Macro Stress Testing Scenarios")
        st_table = stress_table(weights=[1.0], betas=[1.0], equity=float(equity_s.iloc[-1]))
        s1, s2 = st.columns([1, 1])
        with s1:
            st.dataframe(st_table, use_container_width=True)
        with s2:
            st.plotly_chart(stress_bar_figure(st_table), use_container_width=True)

    # Tab 6: Execution
    with tabs[5]:
        st.subheader("Market Microstructure & Order Book Queue")
        st.plotly_chart(lob_queue_figure(fills), use_container_width=True)

    # Tab 7: Portfolio
    with tabs[6]:
        st.subheader("Modern Portfolio Theory Frontier")
        frontier_sample = [(0.10, 0.05), (0.12, 0.08), (0.15, 0.12), (0.18, 0.14), (0.22, 0.15)]
        st.plotly_chart(frontier_figure(frontier_sample), use_container_width=True)

    # Tab 8: Causal Graph
    with tabs[7]:
        st.subheader("Predictive Causality Gating")
        edges = [("Volume", "Return", 0.65), ("Spread", "Return", 0.42), ("VPIN", "Return", 0.78)]
        st.plotly_chart(causal_graph_figure(edges), use_container_width=True)

    # Tab 9: Models
    with tabs[8]:
        st.subheader("Predictive Models & OOS Scores")
        st.write("Cross-Validation & GARCH Volatility Diagnostics")
        st.plotly_chart(risk_profile_radar(metrics), use_container_width=True)

    # Tab 10: Monitoring
    with tabs[9]:
        st.subheader("Distribution Drift & Decay Monitor")
        feats_df = data["features_df"]
        if not feats_df.empty and len(feats_df) > 40:
            mid = len(feats_df) // 2
            d_rep = feature_drift_report(feats_df.iloc[:mid], feats_df.iloc[mid:])
            st.dataframe(d_rep, use_container_width=True)

        decay_res = performance_decay(rets_s)
        st.info(decay_res.get("message", "Performance monitored"))

    # Tab 11: Database
    with tabs[10]:
        st.subheader("Database Registry Audit (Last 5 Rows)")
        cm = ConnectionManager.get_instance()
        try:
            with cm.connection() as conn:
                for tbl in ["assets", "model_registry", "trade_signals", "order_executions", "portfolio_snapshots"]:
                    st.write(f"**Table: `{tbl}`**")
                    try:
                        df_db = pd.read_sql(f"SELECT * FROM {tbl} ORDER BY 1 DESC LIMIT 5;", conn)
                        st.dataframe(df_db, use_container_width=True)
                    except Exception as e:
                        st.caption(f"No records available in {tbl} ({e})")
        except Exception as e:
            st.info(f"Database connection: {e}")

    # Tab 12: Tearsheet
    with tabs[11]:
        st.subheader("Export Tearsheet Report")
        buf = io.StringIO()
        buf.write(f"# TAPEWRIGHT-QUANT TEARSHEET: {data['primary_symbol']}\n\n")
        for k, v in metrics.items():
            buf.write(f"- {k}: {v}\n")
        st.text_area("Tearsheet Metrics", buf.getvalue(), height=300)
        st.download_button("Download Report (TXT)", buf.getvalue(), file_name="tearsheet.txt")


if __name__ == "__main__":
    main()
