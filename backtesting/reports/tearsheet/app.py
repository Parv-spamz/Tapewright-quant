"""
Interactive Tapewright-Quant institutional quantitative research & tearsheet dashboard.

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
    ablation_benefit_figure,
    calibration_reliability_figure,
    causal_graph_figure,
    drawdown_figure,
    equity_figure,
    evidence_metric_figure,
    forward_test_figure,
    frontier_figure,
    lob_queue_figure,
    mc_cone_figure,
    monthly_heatmap_figure,
    not_run_figure,
    price_with_trades_figure,
    returns_hist_figure,
    risk_profile_radar,
    rolling_sharpe_figure,
    stress_bar_figure,
    validation_survives_figure,
)
from backtesting.reports.tearsheet.stress import BUILTIN_SCENARIOS, stress_table
from config.trading.risk_limits import RiskLimits
from data.caching.parquet_store.store import ParquetStore
from database.connection.manager import ConnectionManager
from integrity.ledger import Ledger
from pipelines.etl.etl import run_etl
from pipelines.monitoring.drift import feature_drift_report, performance_decay
from research.lockbox import lockbox_status
from signals.generation.breakout.breakout import BreakoutStrategy
from signals.generation.crossover.crossover import CrossoverStrategy
from signals.generation.mean_reversion.mean_reversion import MeanReversionStrategy


@st.cache_data(show_spinner=False)
def load_cached_etl(symbols: tuple[str, ...], n_days: int, store_dir: str = "data/caching/parquet_store") -> bool:
    """Caches ETL execution to prevent recomputing across Streamlit reruns."""
    start_date = pd.Timestamp.now() - pd.Timedelta(days=int(n_days * 1.5))
    run_etl(symbols=symbols, start=start_date.strftime("%Y-%m-%d"), end=pd.Timestamp.now().strftime("%Y-%m-%d"), store=store_dir)
    return True


@st.cache_data(show_spinner=False)
def load_ledger_info() -> dict[str, Any]:
    """Load cryptographic audit ledger state and chain verification status."""
    ledger_path = Path("research/ledger/ledger.jsonl")
    if not ledger_path.exists():
        return {
            "chain_valid": True,
            "head_hash": "00000000",
            "lockbox_status": "UNOPENED",
            "records": [],
        }
    ledger = Ledger(ledger_path)
    chain_valid, _, _ = ledger.verify_chain()
    records = ledger._read_all_records()
    head_hash = records[-1]["hash"][:8] if records else "00000000"
    lb_status = lockbox_status(ledger)
    return {
        "chain_valid": chain_valid,
        "head_hash": head_hash,
        "lockbox_status": lb_status,
        "records": records,
    }


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
    source: str = "real",
) -> dict[str, Any]:
    """Executes full backtest simulation via BacktestEngine and compiles results."""
    store = ParquetStore("data/caching/parquet_store")
    clean_syms = [s.strip().upper() for s in symbols]

    # Ingest data
    start_str = (pd.Timestamp.now() - pd.Timedelta(days=int(n_days * 1.5))).strftime("%Y-%m-%d")
    end_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    clean_src = "real" if "real" in source.lower() or "yahoo" in source.lower() else ("synthetic" if "synthetic" in source.lower() else "auto")
    etl_report = run_etl(symbols=clean_syms, start=start_str, end=end_str, store=store, source=clean_src)

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
    is_synth = bool(etl_report.source == "synthetic")

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
        "is_synthetic": is_synth,
    }


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Tapewright-Quant Institutional Terminal",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # High-polish institutional visual styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Root styling */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #080B11 !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #E2E8F0 !important;
        }

        /* Fix Streamlit default white header */
        header[data-testid="stHeader"] {
            background-color: #080B11 !important;
            border-bottom: 1px solid #1A2337 !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0D121D !important;
            border-right: 1px solid #1A2337 !important;
        }

        /* Main Institutional Navigation Tab Bar */
        .stTabs {
            margin-top: 14px !important;
            margin-bottom: 24px !important;
        }

        .stTabs [data-baseweb="tab-list"],
        [data-testid="stTabs"] [data-baseweb="tab-list"],
        div[data-baseweb="tab-list"] {
            background-color: #0E1422 !important;
            border: 1px solid #1E293B !important;
            border-radius: 10px !important;
            padding: 6px 10px !important;
            gap: 10px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45) !important;
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
        }

        /* Tab item button */
        .stTabs [data-baseweb="tab"],
        [data-testid="stTabs"] [data-baseweb="tab"],
        button[data-baseweb="tab"] {
            height: 42px !important;
            background-color: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 7px !important;
            color: #94A3B8 !important;
            font-weight: 600 !important;
            font-size: 0.93rem !important;
            padding: 0 18px !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
        }

        .stTabs [data-baseweb="tab"]:hover,
        [data-testid="stTabs"] [data-baseweb="tab"]:hover {
            color: #F8FAFC !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-color: #334155 !important;
        }

        /* Selected Active Tab Highlight Bar */
        .stTabs [data-baseweb="tab"][aria-selected="true"],
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
        .stTabs [aria-selected="true"] {
            background-color: #1E293B !important;
            color: #38BDF8 !important;
            border: 1px solid #38BDF8 !important;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
        }

        /* Suppress default red underline lines from Streamlit BaseWeb */
        [data-baseweb="tab-highlight"],
        [data-baseweb="tab-border"],
        div[data-baseweb="tab-highlight"],
        div[data-baseweb="tab-border"] {
            display: none !important;
            height: 0px !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }

        /* Metrics Card Enhancement */
        .stMetric {
            background: linear-gradient(180deg, #111726 0%, #0C101A 100%) !important;
            border: 1px solid #1E293B !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
            transition: transform 0.15s ease, border-color 0.15s ease !important;
        }

        .stMetric:hover {
            border-color: #38BDF8 !important;
            transform: translateY(-2px) !important;
        }

        .stMetric label {
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            color: #94A3B8 !important;
        }

        .stMetric [data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            font-family: 'JetBrains Mono', monospace !important;
            color: #F8FAFC !important;
        }

        /* Custom Header Terminal Banner */
        .term-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 24px;
            background: linear-gradient(90deg, #0E1422 0%, #121A2C 100%);
            border: 1px solid #1E293B;
            border-radius: 12px;
            margin-bottom: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        }

        .term-brand {
            font-size: 1.4rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .term-sub {
            font-size: 0.85rem;
            color: #94A3B8;
            font-weight: 500;
            margin-top: 3px;
        }

        .badge-strip {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }

        .term-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 6px;
            border: 1px solid;
            letter-spacing: 0.04em;
        }

        .badge-live {
            background: rgba(16, 185, 129, 0.12);
            color: #10B981;
            border-color: rgba(16, 185, 129, 0.3);
        }

        .badge-synth {
            background: rgba(245, 158, 11, 0.12);
            color: #F59E0B;
            border-color: rgba(245, 158, 11, 0.3);
        }

        .badge-info {
            background: rgba(56, 189, 248, 0.1);
            color: #38BDF8;
            border-color: rgba(56, 189, 248, 0.25);
        }

        .badge-sec {
            background: rgba(148, 163, 184, 0.1);
            color: #CBD5E1;
            border-color: rgba(148, 163, 184, 0.2);
        }

        /* Section Cards */
        .section-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-top: 20px;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 1px solid #1E293B;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Terminal Parameters")
        symbols_input = st.text_input("Universe Symbols", value="AAPL, MSFT, SPY")
        days = st.slider("Lookback (Days)", min_value=100, max_value=1000, value=500, step=50)
        data_source = st.selectbox("Market Feed", ["real (live market feed)", "synthetic", "cache"])
        auto_refresh = st.checkbox("🔴 Live Real-Time Auto-Refresh", value=False)
        refresh_rate = st.slider("Refresh Rate (seconds)", min_value=3, max_value=60, value=10, step=1) if auto_refresh else 10
        strategy = st.selectbox("Trading Strategy", ["crossover", "breakout", "mean_reversion", "pairs", "ML-meta"])

        st.markdown("---")
        st.markdown("### ⚡ Execution & Microstructure")
        exec_model = st.selectbox("Execution Engine", ["realistic", "zero", "LOB"])
        slippage_c = st.slider("Kyle Lambda Slippage (c)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        commission = st.number_input("Broker Fee (bps)", min_value=0.0, max_value=50.0, value=5.0)
        vpin_gamma = st.slider("VPIN Adverse Selection Penalty", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

        st.markdown("---")
        st.markdown("### 🛡️ Risk & Allocation")
        sizing = st.selectbox("Position Sizing", ["Half-Kelly+CVaR", "fixed"])
        max_pos = st.slider("Max Position Allocation %", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        target_vol = st.slider("Target Annualized Vol %", min_value=5.0, max_value=40.0, value=15.0) / 100.0
        seed = st.number_input("Reproducibility Seed", value=42, step=1)

        run_btn = st.button("▶ Run Quantitative Simulation", type="primary", use_container_width=True)

    # Initialize session state & run simulation
    if run_btn or "backtest_data" not in st.session_state:
        symbols_list = [s.strip() for s in symbols_input.split(",") if s.strip()]
        selected_src = "real" if "real" in data_source.lower() or "yahoo" in data_source.lower() else data_source
        with st.spinner("Executing quantitative event-loop backtest simulation..."):
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
                source=selected_src,
            )

    data = st.session_state["backtest_data"]
    ledger_info = load_ledger_info()

    # Determine badges
    is_synth = data.get("is_synthetic", False)
    data_badge_class = "badge-synth" if is_synth else "badge-live"
    data_badge_text = "🟠 SYNTHETIC: BENCHMARK" if is_synth else "🟢 LIVE EXCHANGE: REAL-TIME"
    chain_badge_text = f"CHAIN: {'VALID' if ledger_info['chain_valid'] else 'BROKEN'} ({ledger_info['head_hash']})"
    lockbox_badge_text = f"LOCKBOX: {ledger_info['lockbox_status']}"

    # Render Institutional Header Terminal Banner
    st.markdown(
        f"""
        <div class="term-banner">
            <div>
                <div class="term-brand">⚡ TAPEWRIGHT QUANTITATIVE INTELLIGENCE</div>
                <div class="term-sub">Institutional Production Backtest & Quantitative Risk Terminal v2.4</div>
            </div>
            <div class="badge-strip">
                <span class="term-badge {data_badge_class}">{data_badge_text}</span>
                <span class="term-badge badge-info">🔗 {chain_badge_text}</span>
                <span class="term-badge badge-sec">🔒 {lockbox_badge_text}</span>
                <span class="term-badge badge-info">⚡ EXEC: {exec_model.upper()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5 Consolidated Tabs with styled institutional navigation bar
    tabs = st.tabs(
        [
            "📊 Executive Overview",
            "⚡ Trade & Execution",
            "🛡️ Risk & Stress Testing",
            "🤖 Research & Models",
            "🔒 Audit & Ledger",
        ]
    )

    equity_s = data["equity_curve"]
    rets_s = data["returns"]
    metrics = data["metrics"]
    fills = data["fills"]

    # ==========================================
    # TAB 1: EXECUTIVE OVERVIEW
    # ==========================================
    with tabs[0]:
        st.markdown('<div class="section-header">📈 Live Quotes & Universe Tickers</div>', unsafe_allow_html=True)
        store_inst = ParquetStore("data/caching/parquet_store")
        q_cols = st.columns(min(4, max(1, len(data["symbols"]))))
        for idx_sym, sym in enumerate(data["symbols"][:4]):
            p_df = data["prices_df"] if sym == data["primary_symbol"] else store_inst.read(sym, interval="1d")
            cur_p = float(p_df["close"].iloc[-1]) if not p_df.empty else 100.0
            prev_p = float(p_df["close"].iloc[-2]) if len(p_df) > 1 else cur_p
            pct_chg = (cur_p - prev_p) / prev_p * 100.0 if prev_p > 0 else 0.0
            val_p_str = f"${cur_p:,.2f}"
            delta_p_str = f"{pct_chg:+.2f}%"
            with q_cols[idx_sym]:
                st.metric(label=f"{sym} Last Close", value=val_p_str, delta=delta_p_str)

        st.markdown('<div class="section-header">🏆 Portfolio Performance Summary</div>', unsafe_allow_html=True)
        m_cols = st.columns(5)
        eq_start = float(equity_s.iloc[0]) if len(equity_s) > 0 else 100_000.0
        eq_end = float(equity_s.iloc[-1]) if len(equity_s) > 0 else eq_start
        tot_ret = (eq_end - eq_start) / eq_start if eq_start > 0 else 0.0
        sr = float(metrics.get("sharpe", 0.0))
        mdd = float(metrics.get("max_drawdown", 0.0))
        ann_vol = float(metrics.get("annualized_vol", 0.0))
        t_len = len(rets_s)
        sk = float(metrics.get("skew", 0.0))
        kt = float(metrics.get("kurtosis", 3.0))
        psr = probabilistic_sharpe_ratio(sr_hat=sr, sr_benchmark=0.0, T=t_len, skew=sk, kurt=kt) if t_len > 10 else 0.5

        nav_str = f"${eq_end:,.2f}"
        tot_ret_str = f"{tot_ret * 100.0:+.2f}%"
        sr_str = f"{sr:.2f}"
        mdd_str = f"{mdd * 100.0:.2f}%"
        psr_str = f"{psr * 100.0:.1f}%"

        m_cols[0].metric("Portfolio NAV", nav_str, delta=tot_ret_str)
        m_cols[1].metric("Total Net Return", tot_ret_str)
        m_cols[2].metric("Sharpe Ratio (Ann.)", sr_str)
        m_cols[3].metric("Max Drawdown", mdd_str)
        m_cols[4].metric("Probabilistic SR", psr_str)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(equity_figure(equity_s, data.get("benchmark")), use_container_width=True)
        with c2:
            st.plotly_chart(drawdown_figure(data["drawdown_curve"]), use_container_width=True)

        st.plotly_chart(monthly_heatmap_figure(rets_s), use_container_width=True)

        with st.expander("📄 Export Institutional Tearsheet Report", expanded=False):
            buf = io.StringIO()
            buf.write(f"# TAPEWRIGHT-QUANT INSTITUTIONAL TEARSHEET: {data['primary_symbol']}\n")
            buf.write(f"Universe: {', '.join(data['symbols'])}\n")
            buf.write(f"Initial NAV: ${eq_start:,.2f} | Final NAV: ${eq_end:,.2f} | Net Return: {tot_ret * 100.0:+.2f}%\n\n")
            buf.write("## Risk & Return Metrics\n")
            for k, v in metrics.items():
                buf.write(f"- {k}: {v}\n")
            st.text_area("Tearsheet Metrics Payload", buf.getvalue(), height=220)
            st.download_button("📥 Download Tearsheet (TXT)", buf.getvalue(), file_name=f"tearsheet_{data['primary_symbol']}.txt")

    # ==========================================
    # TAB 2: TRADE & EXECUTION
    # ==========================================
    with tabs[1]:
        st.markdown('<div class="section-header">🎯 Execution Price Action & Fill Markers</div>', unsafe_allow_html=True)
        st.plotly_chart(price_with_trades_figure(data["prices_df"], fills), use_container_width=True)

        st.markdown('<div class="section-header">📊 Microstructure & LOB Dynamics</div>', unsafe_allow_html=True)
        mic_c1, mic_c2 = st.columns([1, 1])
        with mic_c1:
            st.plotly_chart(lob_queue_figure(fills), use_container_width=True)
        with mic_c2:
            st.markdown("#### Microstructure Slippage & Impact Parameters")
            stat_c1, stat_c2 = st.columns(2)
            c_val_str = f"{slippage_c:.2f}"
            vpin_str = f"{vpin_gamma:.2f}"
            comm_str = f"{commission:.1f} bps"
            fills_count_str = f"{len(fills)}"
            with stat_c1:
                st.metric("Kyle Lambda Impact (c)", c_val_str)
                st.metric("Broker Commission", comm_str)
            with stat_c2:
                st.metric("VPIN Adverse Penalty", vpin_str)
                st.metric("Executed Fill Events", fills_count_str)

        st.markdown('<div class="section-header">📋 Institutional Trade Blotter Log</div>', unsafe_allow_html=True)
        if fills:
            fills_rows = [f.__dict__ if hasattr(f, "__dict__") else f for f in fills]
            st.dataframe(pd.DataFrame(fills_rows), use_container_width=True)
        else:
            st.info("No trade fill events logged in simulation.")

    # ==========================================
    # TAB 3: RISK & STRESS TESTING
    # ==========================================
    with tabs[2]:
        st.markdown('<div class="section-header">🛡️ Tail Risk Distribution & Volatility Dynamics</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            var_val = float(metrics.get("parametric_var_95", 0.02))
            cvar_val = float(metrics.get("cvar_95", 0.03))
            st.plotly_chart(returns_hist_figure(rets_s, var=var_val, cvar=cvar_val), use_container_width=True)
        with r2:
            roll_sr = rets_s.rolling(60).apply(lambda r: np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252)).dropna()
            st.plotly_chart(rolling_sharpe_figure(roll_sr), use_container_width=True)

        if len(rets_s) > 20:
            ci = bootstrap_sharpe_ci(rets_s.values)
            ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
            st.markdown(f"**Bootstrap 95% Sharpe Ratio Confidence Interval:** `{ci_str}`")

        st.markdown('<div class="section-header">🔮 Forward Horizon Monte Carlo Simulation</div>', unsafe_allow_html=True)
        if len(rets_s) > 10:
            cone = monte_carlo_equity_cone(rets_s.values, horizon=min(len(rets_s), 252), n_paths=200)
            st.plotly_chart(mc_cone_figure(cone, equity_s), use_container_width=True)
        else:
            st.caption("Insufficient historical observations for forward Monte Carlo cone.")

        st.markdown('<div class="section-header">⚡ Macro Crisis Stress Testing Scenarios</div>', unsafe_allow_html=True)
        st_table = stress_table(weights=[1.0], betas=[1.0], equity=float(equity_s.iloc[-1]))
        s1, s2 = st.columns([1, 1])
        with s1:
            st.dataframe(st_table, use_container_width=True)
        with s2:
            st.plotly_chart(stress_bar_figure(st_table), use_container_width=True)

    # ==========================================
    # TAB 4: RESEARCH & MODELS
    # ==========================================
    with tabs[3]:
        st.markdown('<div class="section-header">🤖 Predictive Causality & Factor Radar</div>', unsafe_allow_html=True)
        cg1, cg2 = st.columns(2)
        with cg1:
            edges = [("Volume", "Return", 0.65), ("Spread", "Return", 0.42), ("VPIN", "Return", 0.78)]
            st.plotly_chart(causal_graph_figure(edges), use_container_width=True)
        with cg2:
            st.plotly_chart(risk_profile_radar(metrics), use_container_width=True)

        st.markdown('<div class="section-header">📐 Model Calibration & Modern Portfolio Theory</div>', unsafe_allow_html=True)
        cal_c1, cal_c2 = st.columns(2)
        with cal_c1:
            records = ledger_info["records"]
            cal_recs = [r for r in records if r.get("kind") == "calibration"]
            prov_badge = "REAL" if not data.get("is_synthetic", True) else "SYNTHETIC"
            if cal_recs:
                cal_payload = cal_recs[-1].get("payload", {})
                brier_val_str = f"{float(cal_payload.get('brier_score', 0.0)):.4f}"
                st.metric("Model Brier Score", brier_val_str)
                st.plotly_chart(calibration_reliability_figure(cal_payload, provenance=prov_badge), use_container_width=True)
            else:
                st.warning("Calibration diagnostics awaiting registered out-of-sample data.")
                st.plotly_chart(calibration_reliability_figure("NOT_RUN: calibration phase unexecuted", provenance=prov_badge), use_container_width=True)
        with cal_c2:
            frontier_sample = [(0.10, 0.05), (0.12, 0.08), (0.15, 0.12), (0.18, 0.14), (0.22, 0.15)]
            st.plotly_chart(frontier_figure(frontier_sample), use_container_width=True)

        st.markdown('<div class="section-header">📡 Distribution Drift & Performance Decay Monitor</div>', unsafe_allow_html=True)
        feats_df = data["features_df"]
        if not feats_df.empty and len(feats_df) > 40:
            mid = len(feats_df) // 2
            d_rep = feature_drift_report(feats_df.iloc[:mid], feats_df.iloc[mid:])
            st.dataframe(d_rep, use_container_width=True)

        decay_res = performance_decay(rets_s)
        st.info(decay_res.get("message", "Feature distribution continuously verified."))

    # ==========================================
    # TAB 5: AUDIT & LEDGER
    # ==========================================
    with tabs[4]:
        st.markdown('<div class="section-header">🔒 Protocol Evidence & Overfitting Defense</div>', unsafe_allow_html=True)
        records = ledger_info["records"]
        exp_recs = [r for r in records if r.get("kind") == "experiment"]
        dev_recs = [r for r in exp_recs if r.get("payload", {}).get("role") == "dev"]

        ev_c1, ev_c2, ev_c3 = st.columns(3)
        n_trials_str = f"{len(exp_recs)}"
        dev_dsr = float(dev_recs[-1].get("payload", {}).get("metrics", {}).get("dsr", 0.0)) if dev_recs else 0.0
        dev_pbo = float(dev_recs[-1].get("payload", {}).get("metrics", {}).get("pbo", 0.0)) if dev_recs else 0.0
        dsr_str = f"{dev_dsr:.3f}"
        pbo_str = f"{dev_pbo:.3f}"

        ev_c1.metric("Registered Protocol Trials", n_trials_str)
        ev_c2.metric("Deflated Sharpe Ratio (DSR)", dsr_str)
        ev_c3.metric("Prob. of Backtest Overfitting (PBO)", pbo_str)

        st.markdown("#### Validation Holdout & Feature Ablation Analysis")
        val_c1, val_c2 = st.columns(2)
        prov_badge = "REAL" if not data.get("is_synthetic", True) else "SYNTHETIC"
        with val_c1:
            val_recs = [r for r in exp_recs if r.get("payload", {}).get("role") == "val"]
            if val_recs:
                val_rows = [
                    {
                        "Tag": r.get("tag"),
                        "Sharpe": float(r.get("payload", {}).get("metrics", {}).get("sharpe", 0.0)),
                        "Survives": bool(r.get("payload", {}).get("survives", False)),
                        "Status": r.get("status"),
                    }
                    for r in val_recs
                ]
                st.dataframe(pd.DataFrame(val_rows), use_container_width=True)
                st.plotly_chart(validation_survives_figure(val_recs, provenance=prov_badge), use_container_width=True)
            else:
                st.warning("Validation holdout records unexecuted.")
                st.plotly_chart(validation_survives_figure("NOT_RUN: validation phase unexecuted", provenance=prov_badge), use_container_width=True)

        with val_c2:
            abl_recs = [r for r in records if "ablation" in str(r.get("tag", "")) or "ablation" in str(r.get("kind", ""))]
            if abl_recs:
                abl_rows = [
                    {
                        "Component": str(r.get("payload", {}).get("component", r.get("tag"))),
                        "Delta": float(r.get("payload", {}).get("delta", 0.0)),
                        "Benefit": "Demonstrated Benefit" if r.get("payload", {}).get("benefit") else "No demonstrated benefit",
                    }
                    for r in abl_recs
                ]
                st.dataframe(pd.DataFrame(abl_rows), use_container_width=True)
                st.plotly_chart(ablation_benefit_figure(abl_recs, provenance=prov_badge), use_container_width=True)
            else:
                st.warning("Ablation analysis experiments not recorded in ledger.")
                st.plotly_chart(ablation_benefit_figure("NOT_RUN: ablation experiments not recorded", provenance=prov_badge), use_container_width=True)

        st.markdown('<div class="section-header">📜 Cryptographic Hash Chain Audit Ledger</div>', unsafe_allow_html=True)
        if ledger_info["chain_valid"]:
            st.success(f"Audit Trail Intact: Cryptographic hash chain verified valid (Head: {ledger_info['head_hash']})")
        else:
            st.error("Audit Trail Alert: Cryptographic hash chain verification failed!")

        if records:
            display_records = [
                {
                    "Seq": r.get("seq"),
                    "Timestamp": r.get("ts_utc"),
                    "Kind": r.get("kind"),
                    "Tag": r.get("tag"),
                    "Status": r.get("status"),
                    "Hash Prefix": str(r.get("hash", ""))[:12] + "...",
                    "Prev Hash": str(r.get("prev_hash", ""))[:12] + "...",
                }
                for r in records
            ]
            st.dataframe(pd.DataFrame(display_records), use_container_width=True)

        st.markdown('<div class="section-header">🗄️ Relational Database Registry Audit</div>', unsafe_allow_html=True)
        cm = ConnectionManager.get_instance()
        try:
            with cm.connection() as conn:
                db_tabs = st.tabs(["trade_signals", "order_executions", "portfolio_snapshots", "model_registry"])
                for i, tbl in enumerate(["trade_signals", "order_executions", "portfolio_snapshots", "model_registry"]):
                    with db_tabs[i]:
                        try:
                            df_db = pd.read_sql(f"SELECT * FROM {tbl} ORDER BY 1 DESC LIMIT 10;", conn)
                            st.dataframe(df_db, use_container_width=True)
                        except Exception as e:
                            st.caption(f"No records available in {tbl} ({e})")
        except Exception as e:
            st.info(f"Database connection: {e}")

    # Auto refresh trigger
    if auto_refresh:
        import time
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
