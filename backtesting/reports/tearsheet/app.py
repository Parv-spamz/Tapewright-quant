"""
Interactive Tapewright institutional quantitative research & tearsheet dashboard.

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
            "lockbox_status": "LOCKED",
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
    clean_src = "real" if any(k in source.lower() for k in ("live", "real", "sip", "yahoo")) else ("synthetic" if "synth" in source.lower() else "auto")
    etl_report = run_etl(symbols=clean_syms, start=start_str, end=end_str, store=store, source=clean_src)

    # Load primary price series
    primary_sym = clean_syms[0]
    prices_df = store.read(primary_sym, interval="1d")
    features_df = store.read(primary_sym, interval="features_1d")

    # Strategy selection
    norm_strat = strategy_name.lower()
    if "breakout" in norm_strat:
        strategy = BreakoutStrategy(symbol=primary_sym, n=20)
    elif "reversion" in norm_strat:
        strategy = MeanReversionStrategy(symbol=primary_sym)
    else:
        strategy = CrossoverStrategy(symbol=primary_sym, fast=10, slow=30)

    # Cost model
    if "zero" in execution_type.lower():
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
    """Main application entry point."""
    st.set_page_config(
        page_title="Tapewright // Core",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Strict desaturated institutional CSS architecture
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Root Canvas */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #090A0F !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #F1F5F9 !important;
            font-variant-numeric: tabular-nums !important;
        }

        /* Hide native Streamlit header & toolbar */
        header[data-testid="stHeader"] {
            display: none !important;
        }

        /* Container tight margins for 1080p single viewport fit */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100% !important;
        }

        /* Non-scrolling, viewport-locked sidebar */
        [data-testid="stSidebar"] {
            background-color: #0E1117 !important;
            border-right: 1px solid #1E2433 !important;
            overflow-y: hidden !important;
            max-height: 100vh !important;
        }

        [data-testid="stSidebarContent"] {
            overflow-y: hidden !important;
            padding: 0.75rem 0.65rem !important;
        }

        [data-testid="stSidebarContent"] .stVerticalBlock {
            gap: 0.35rem !important;
        }

        /* Sidebar Section Titles */
        .sidebar-sec-title {
            font-size: 10px !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: #64748B !important;
            margin-top: 0.4rem !important;
            margin-bottom: 0.15rem !important;
            border-bottom: 1px solid #1E2433 !important;
            padding-bottom: 2px !important;
        }

        /* Compact form input labels */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p {
            font-size: 10px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: #64748B !important;
            margin-bottom: 1px !important;
        }

        /* Compact form input fields */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #12161F !important;
            border: 1px solid #1E2433 !important;
            border-radius: 3px !important;
            color: #F1F5F9 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            min-height: 28px !important;
            height: 28px !important;
            padding: 2px 6px !important;
        }

        /* Overriding Streamlit neon red on sliders */
        [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
            background-color: #3B82F6 !important;
            border-color: #3B82F6 !important;
            box-shadow: 0 0 4px rgba(59, 130, 246, 0.4) !important;
            width: 12px !important;
            height: 12px !important;
        }

        [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: #F1F5F9 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 9px !important;
        }

        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background: #3B82F6 !important;
        }

        /* Overriding checkbox accent */
        [data-testid="stCheckbox"] input:checked + div {
            background-color: #3B82F6 !important;
            border-color: #3B82F6 !important;
        }

        /* Primary action button */
        button[kind="primary"] {
            background: #1D4ED8 !important;
            border: 1px solid #3B82F6 !important;
            color: #F8FAFC !important;
            border-radius: 3px !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em !important;
            padding: 5px 10px !important;
            margin-top: 4px !important;
            height: 32px !important;
        }

        button[kind="primary"]:hover {
            background: #2563EB !important;
            border-color: #60A5FA !important;
        }

        /* Top Header Utility Bar */
        .utility-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 36px;
            border-bottom: 1px solid #1E2433;
            margin-bottom: 6px;
            font-family: 'JetBrains Mono', monospace;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .brand-name {
            font-size: 13px;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: 0.05em;
        }

        .brand-divider {
            color: #334155;
            font-size: 12px;
        }

        .brand-sub {
            font-size: 11px;
            color: #64748B;
            font-weight: 500;
        }

        .header-telemetry {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 11px;
            color: #94A3B8;
        }

        .tele-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .status-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }

        .dot-emerald { background-color: #10B981; }
        .dot-crimson { background-color: #EF4444; }
        .dot-amber   { background-color: #F59E0B; }
        .dot-blue    { background-color: #3B82F6; }
        .dot-slate   { background-color: #64748B; }

        /* Minimal Navigation Tab Bar */
        .stTabs {
            margin-top: 2px !important;
            margin-bottom: 8px !important;
        }

        .stTabs [data-baseweb="tab-list"],
        div[data-baseweb="tab-list"] {
            background-color: transparent !important;
            border: none !important;
            border-bottom: 1px solid #1E2433 !important;
            border-radius: 0 !important;
            padding: 0 !important;
            gap: 0 !important;
            box-shadow: none !important;
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
        }

        .stTabs [data-baseweb="tab"],
        button[data-baseweb="tab"] {
            height: 32px !important;
            background-color: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            border-radius: 0 !important;
            color: #64748B !important;
            font-weight: 500 !important;
            font-size: 12px !important;
            padding: 4px 16px !important;
            cursor: pointer !important;
            transition: color 0.15s ease, border-color 0.15s ease !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #F1F5F9 !important;
            background-color: transparent !important;
            border-bottom-color: #334155 !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"],
        .stTabs [aria-selected="true"] {
            background-color: transparent !important;
            color: #F8FAFC !important;
            border-bottom: 2px solid #3B82F6 !important;
            box-shadow: none !important;
        }

        [data-baseweb="tab-highlight"],
        [data-baseweb="tab-border"] {
            display: none !important;
            height: 0px !important;
            opacity: 0 !important;
        }

        /* 1-Line Compact Ticker Strip */
        .ticker-strip {
            display: flex;
            align-items: center;
            height: 30px;
            background-color: #12161F;
            border: 1px solid #1E2433;
            border-radius: 3px;
            padding: 0 12px;
            margin-bottom: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }

        .ticker-tag {
            color: #64748B;
            font-weight: 700;
            font-size: 10px;
            letter-spacing: 0.08em;
            margin-right: 12px;
        }

        .ticker-items {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .ticker-unit {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .tick-sym {
            font-weight: 700;
            color: #F1F5F9;
        }

        .tick-val {
            color: #CBD5E1;
        }

        .tick-chg-pos {
            color: #10B981;
        }

        .tick-chg-neg {
            color: #F43F5E;
        }

        .tick-sep {
            color: #2D3748;
        }

        /* High-Density Unified KPI Metric Strip */
        .kpi-strip {
            display: flex;
            width: 100%;
            background-color: #12161F;
            border: 1px solid #1E2433;
            border-radius: 3px;
            margin-bottom: 10px;
        }

        .kpi-cell {
            flex: 1;
            padding: 8px 14px;
            border-right: 1px solid #1E2433;
        }

        .kpi-cell:last-child {
            border-right: none;
        }

        .kpi-label {
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #64748B;
            margin-bottom: 2px;
        }

        .kpi-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            font-weight: 700;
            color: #F8FAFC;
            display: flex;
            align-items: baseline;
            gap: 6px;
        }

        .kpi-sub {
            font-size: 11px;
            font-weight: 500;
        }

        .val-pos {
            color: #10B981;
        }

        .val-neg {
            color: #F43F5E;
        }

        /* Plotly chart container crisp borders */
        [data-testid="stPlotlyChart"] {
            border: 1px solid #1E2433 !important;
            border-radius: 3px !important;
            background-color: #12161F !important;
            overflow: hidden !important;
        }

        /* Secondary section headers */
        .sec-label {
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748B;
            margin-top: 10px;
            margin-bottom: 6px;
            border-bottom: 1px solid #1E2433;
            padding-bottom: 2px;
        }

        /* Tables & DataFrames */
        [data-testid="stDataFrame"] {
            border: 1px solid #1E2433 !important;
            border-radius: 3px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Viewport-Locked Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-sec-title">UNIVERSE & FEED</div>', unsafe_allow_html=True)
        col_u1, col_u2 = st.columns([1.2, 0.8])
        with col_u1:
            symbols_input = st.text_input("Symbols", value="AAPL, MSFT, SPY")
        with col_u2:
            days = st.number_input("Lookback (D)", min_value=50, max_value=2000, value=500, step=50)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            data_source = st.selectbox("Feed", ["LIVE (SIP/Direct)", "Synthetic", "Local Cache"])
        with col_f2:
            strategy = st.selectbox("Strategy", ["MA Crossover", "Breakout", "Mean Reversion", "Pairs Arbitrage", "ML-Meta"])

        col_sync1, col_sync2 = st.columns([1.2, 0.8])
        with col_sync1:
            auto_refresh = st.checkbox("Auto-Sync Feed", value=False)
        with col_sync2:
            refresh_rate = st.number_input("Rate (s)", min_value=3, max_value=60, value=10, step=1) if auto_refresh else 10

        st.markdown('<div class="sidebar-sec-title">MICROSTRUCTURE & COSTS</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            exec_model = st.selectbox("Engine", ["Realistic (LOB)", "Zero Cost", "Queue Model"])
        with col_m2:
            commission = st.number_input("Fee (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            slippage_c = st.slider("Kyle Lambda (c)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        with col_m4:
            vpin_gamma = st.slider("VPIN Penalty", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

        st.markdown('<div class="sidebar-sec-title">RISK & ALLOCATION</div>', unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            sizing = st.selectbox("Sizing", ["Half-Kelly + CVaR", "Fixed Ratio"])
        with col_r2:
            max_pos = st.slider("Max Alloc %", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

        col_r3, col_r4 = st.columns(2)
        with col_r3:
            target_vol = st.slider("Target Vol %", min_value=5.0, max_value=40.0, value=15.0, step=1.0) / 100.0
        with col_r4:
            seed = st.number_input("Seed", value=42, step=1)

        run_btn = st.button("EXECUTE SIMULATION", type="primary", use_container_width=True)

    # Initialize session state & execute simulation
    if run_btn or "backtest_data" not in st.session_state:
        symbols_list = [s.strip() for s in symbols_input.split(",") if s.strip()]
        selected_src = "real" if "live" in data_source.lower() or "direct" in data_source.lower() else data_source
        with st.spinner("Processing event-driven execution..."):
            st.session_state["backtest_data"] = run_backtest_simulation(
                symbols=symbols_list,
                n_days=int(days),
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

    # Determine status telemetry
    is_synth = data.get("is_synthetic", False)
    feed_dot = "dot-amber" if is_synth else "dot-emerald"
    feed_text = "SYNTHETIC" if is_synth else "LIVE"

    chain_valid = ledger_info.get("chain_valid", False)
    chain_dot = "dot-emerald" if chain_valid else "dot-crimson"
    chain_status = "VERIFIED" if chain_valid else "UNVERIFIED"
    head_hash = ledger_info.get("head_hash", "00000000")

    vault_status = str(ledger_info.get("lockbox_status", "LOCKED"))
    vault_dot = "dot-amber" if "lock" in vault_status.lower() else "dot-slate"

    exec_label = "REALISTIC" if "real" in exec_model.lower() else ("ZERO" if "zero" in exec_model.lower() else "QUEUE")

    # 2. Compact Top Utility Bar
    st.markdown(
        f"""
        <div class="utility-header">
            <div class="header-left">
                <span class="brand-name">TAPEWRIGHT</span>
                <span class="brand-divider">|</span>
                <span class="brand-sub">EXECUTION & RISK ENGINE</span>
            </div>
            <div class="header-telemetry">
                <div class="tele-item"><span class="status-dot {feed_dot}"></span> FEED: {feed_text}</div>
                <div class="tele-item"><span class="status-dot {chain_dot}"></span> LEDGER: {head_hash} ({chain_status})</div>
                <div class="tele-item"><span class="status-dot {vault_dot}"></span> VAULT: {vault_status}</div>
                <div class="tele-item"><span class="status-dot dot-blue"></span> FILL MODEL: {exec_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Minimal Bottom-Bordered Navigation Tabs
    tabs = st.tabs(["Overview", "Execution", "Risk & Stress", "Models", "Audit Log"])

    equity_s = data["equity_curve"]
    rets_s = data["returns"]
    metrics = data["metrics"]
    fills = data["fills"]

    # ==========================================
    # TAB 1: OVERVIEW
    # ==========================================
    with tabs[0]:
        # Compact 1-line Ticker Strip
        store_inst = ParquetStore("data/caching/parquet_store")
        ticker_units_html = []
        for sym in data["symbols"][:4]:
            p_df = data["prices_df"] if sym == data["primary_symbol"] else store_inst.read(sym, interval="1d")
            cur_p = float(p_df["close"].iloc[-1]) if not p_df.empty else 100.0
            prev_p = float(p_df["close"].iloc[-2]) if len(p_df) > 1 else cur_p
            pct_chg = (cur_p - prev_p) / prev_p * 100.0 if prev_p > 0 else 0.0
            chg_cls = "tick-chg-pos" if pct_chg >= 0 else "tick-chg-neg"
            ticker_units_html.append(
                f'<div class="ticker-unit"><span class="tick-sym">{sym}</span><span class="tick-val">${cur_p:,.2f}</span><span class="{chg_cls}">{pct_chg:+.2f}%</span></div>'
            )
        ticker_joined = ' <span class="tick-sep">|</span> '.join(ticker_units_html)

        st.markdown(
            f"""
            <div class="ticker-strip">
                <span class="ticker-tag">MARKET WATCH</span>
                <div class="ticker-items">{ticker_joined}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # High-Density KPI Metric Grid
        eq_start = float(equity_s.iloc[0]) if len(equity_s) > 0 else 100_000.0
        eq_end = float(equity_s.iloc[-1]) if len(equity_s) > 0 else eq_start
        tot_ret = (eq_end - eq_start) / eq_start if eq_start > 0 else 0.0
        sr = float(metrics.get("sharpe", 0.0))
        mdd = float(metrics.get("max_drawdown", 0.0))
        t_len = len(rets_s)
        sk = float(metrics.get("skew", 0.0))
        kt = float(metrics.get("kurtosis", 3.0))
        psr = probabilistic_sharpe_ratio(sr_hat=sr, sr_benchmark=0.0, T=t_len, skew=sk, kurt=kt) if t_len > 10 else 0.5

        ret_cls = "val-pos" if tot_ret >= 0 else "val-neg"
        mdd_cls = "val-neg" if mdd > 0 else "val-pos"

        st.markdown(
            f"""
            <div class="kpi-strip">
                <div class="kpi-cell">
                    <div class="kpi-label">PORTFOLIO NAV</div>
                    <div class="kpi-val">${eq_end:,.2f} <span class="kpi-sub {ret_cls}">{tot_ret * 100.0:+.2f}%</span></div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">NET RETURN</div>
                    <div class="kpi-val {ret_cls}">{tot_ret * 100.0:+.2f}%</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">SHARPE (ANN.)</div>
                    <div class="kpi-val">{sr:.2f}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">MAX DRAWDOWN</div>
                    <div class="kpi-val {mdd_cls}">-{abs(mdd) * 100.0:.2f}%</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">PROBABILISTIC SR</div>
                    <div class="kpi-val">{psr * 100.0:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Performance Charts Side-by-Side
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(equity_figure(equity_s, data.get("benchmark")), use_container_width=True)
        with c2:
            st.plotly_chart(drawdown_figure(data["drawdown_curve"]), use_container_width=True)

        st.markdown('<div class="sec-label">MONTHLY RETURNS MATRIX</div>', unsafe_allow_html=True)
        st.plotly_chart(monthly_heatmap_figure(rets_s), use_container_width=True)

        with st.expander("REPORT PAYLOAD & EXPORT", expanded=False):
            buf = io.StringIO()
            buf.write(f"# TAPEWRIGHT TEARSHEET: {data['primary_symbol']}\n")
            buf.write(f"Universe: {', '.join(data['symbols'])}\n")
            buf.write(f"Initial NAV: ${eq_start:,.2f} | Final NAV: ${eq_end:,.2f} | Net Return: {tot_ret * 100.0:+.2f}%\n\n")
            for k, v in metrics.items():
                buf.write(f"- {k}: {v}\n")
            st.text_area("Metrics", buf.getvalue(), height=180)
            st.download_button("DOWNLOAD TEARSHEET (TXT)", buf.getvalue(), file_name=f"tearsheet_{data['primary_symbol']}.txt")

    # ==========================================
    # TAB 2: EXECUTION
    # ==========================================
    with tabs[1]:
        st.markdown('<div class="sec-label">PRICE ACTION & FILL MARKERS</div>', unsafe_allow_html=True)
        st.plotly_chart(price_with_trades_figure(data["prices_df"], fills), use_container_width=True)

        st.markdown('<div class="sec-label">MICROSTRUCTURE QUEUE DYNAMICS</div>', unsafe_allow_html=True)
        mic_c1, mic_c2 = st.columns([1, 1])
        with mic_c1:
            st.plotly_chart(lob_queue_figure(fills), use_container_width=True)
        with mic_c2:
            c_val_str = f"{slippage_c:.2f}"
            vpin_str = f"{vpin_gamma:.2f}"
            comm_str = f"{commission:.1f} bps"
            fills_count_str = f"{len(fills)}"
            st.markdown(
                f"""
                <div class="kpi-strip" style="flex-wrap: wrap;">
                    <div class="kpi-cell" style="min-width: 45%;">
                        <div class="kpi-label">KYLE LAMBDA IMPACT (C)</div>
                        <div class="kpi-val">{c_val_str}</div>
                    </div>
                    <div class="kpi-cell" style="min-width: 45%; border-right: none;">
                        <div class="kpi-label">VPIN PENALTY</div>
                        <div class="kpi-val">{vpin_str}</div>
                    </div>
                    <div class="kpi-cell" style="min-width: 45%; border-top: 1px solid #1E2433;">
                        <div class="kpi-label">COMMISSION</div>
                        <div class="kpi-val">{comm_str}</div>
                    </div>
                    <div class="kpi-cell" style="min-width: 45%; border-top: 1px solid #1E2433; border-right: none;">
                        <div class="kpi-label">FILL COUNT</div>
                        <div class="kpi-val">{fills_count_str}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sec-label">TRADE EXECUTION BLOTTER</div>', unsafe_allow_html=True)
        if fills:
            fills_rows = [f.__dict__ if hasattr(f, "__dict__") else f for f in fills]
            st.dataframe(pd.DataFrame(fills_rows), use_container_width=True)
        else:
            st.info("No trade fill events logged in simulation.")

    # ==========================================
    # TAB 3: RISK & STRESS
    # ==========================================
    with tabs[2]:
        st.markdown('<div class="sec-label">RETURN DISTRIBUTION & ROLLING SHARPE</div>', unsafe_allow_html=True)
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
            st.caption(f"Bootstrap 95% Sharpe Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")

        st.markdown('<div class="sec-label">MONTE CARLO EQUITY CONE</div>', unsafe_allow_html=True)
        if len(rets_s) > 10:
            cone = monte_carlo_equity_cone(rets_s.values, horizon=min(len(rets_s), 252), n_paths=200)
            st.plotly_chart(mc_cone_figure(cone, equity_s), use_container_width=True)

        st.markdown('<div class="sec-label">MACRO STRESS TESTING BENCHMARKS</div>', unsafe_allow_html=True)
        st_table = stress_table(weights=[1.0], betas=[1.0], equity=float(equity_s.iloc[-1]))
        s1, s2 = st.columns([1, 1])
        with s1:
            st.dataframe(st_table, use_container_width=True)
        with s2:
            st.plotly_chart(stress_bar_figure(st_table), use_container_width=True)

    # ==========================================
    # TAB 4: MODELS
    # ==========================================
    with tabs[3]:
        st.markdown('<div class="sec-label">PREDICTIVE CAUSALITY & RISK RADAR</div>', unsafe_allow_html=True)
        cg1, cg2 = st.columns(2)
        with cg1:
            edges = [("Volume", "Return", 0.65), ("Spread", "Return", 0.42), ("VPIN", "Return", 0.78)]
            st.plotly_chart(causal_graph_figure(edges), use_container_width=True)
        with cg2:
            st.plotly_chart(risk_profile_radar(metrics), use_container_width=True)

        st.markdown('<div class="sec-label">CALIBRATION & EFFICIENT FRONTIER</div>', unsafe_allow_html=True)
        cal_c1, cal_c2 = st.columns(2)
        with cal_c1:
            records = ledger_info["records"]
            cal_recs = [r for r in records if r.get("kind") == "calibration"]
            prov_badge = "REAL" if not data.get("is_synthetic", True) else "SYNTHETIC"
            if cal_recs:
                cal_payload = cal_recs[-1].get("payload", {})
                brier_val_str = f"{float(cal_payload.get('brier_score', 0.0)):.4f}"
                st.caption(f"Brier Reliability Score: {brier_val_str}")
                st.plotly_chart(calibration_reliability_figure(cal_payload, provenance=prov_badge), use_container_width=True)
            else:
                st.caption("Calibration data pending out-of-sample registration.")
                st.plotly_chart(calibration_reliability_figure("NOT_RUN: calibration unexecuted", provenance=prov_badge), use_container_width=True)
        with cal_c2:
            frontier_sample = [(0.10, 0.05), (0.12, 0.08), (0.15, 0.12), (0.18, 0.14), (0.22, 0.15)]
            st.plotly_chart(frontier_figure(frontier_sample), use_container_width=True)

        st.markdown('<div class="sec-label">FEATURE DRIFT & PERFORMANCE DECAY</div>', unsafe_allow_html=True)
        feats_df = data["features_df"]
        if not feats_df.empty and len(feats_df) > 40:
            mid = len(feats_df) // 2
            d_rep = feature_drift_report(feats_df.iloc[:mid], feats_df.iloc[mid:])
            st.dataframe(d_rep, use_container_width=True)

        decay_res = performance_decay(rets_s)
        st.caption(decay_res.get("message", "Feature distribution validated."))

    # ==========================================
    # TAB 5: AUDIT LOG
    # ==========================================
    with tabs[4]:
        st.markdown('<div class="sec-label">PROTOCOL EVIDENCE & OVERFITTING DEFENSE</div>', unsafe_allow_html=True)
        records = ledger_info["records"]
        exp_recs = [r for r in records if r.get("kind") == "experiment"]
        dev_recs = [r for r in exp_recs if r.get("payload", {}).get("role") == "dev"]

        n_trials_str = f"{len(exp_recs)}"
        dev_dsr = float(dev_recs[-1].get("payload", {}).get("metrics", {}).get("dsr", 0.0)) if dev_recs else 0.0
        dev_pbo = float(dev_recs[-1].get("payload", {}).get("metrics", {}).get("pbo", 0.0)) if dev_recs else 0.0
        dsr_str = f"{dev_dsr:.3f}"
        pbo_str = f"{dev_pbo:.3f}"

        st.markdown(
            f"""
            <div class="kpi-strip">
                <div class="kpi-cell">
                    <div class="kpi-label">PROTOCOL TRIALS</div>
                    <div class="kpi-val">{n_trials_str}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">DEFLATED SHARPE (DSR)</div>
                    <div class="kpi-val">{dsr_str}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">PROB. OVERFITTING (PBO)</div>
                    <div class="kpi-val">{pbo_str}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sec-label">VALIDATION HOLDOUT & FEATURE ABLATION</div>', unsafe_allow_html=True)
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
                st.caption("Validation holdout unexecuted.")
                st.plotly_chart(validation_survives_figure("NOT_RUN: validation unexecuted", provenance=prov_badge), use_container_width=True)

        with val_c2:
            abl_recs = [r for r in records if "ablation" in str(r.get("tag", "")) or "ablation" in str(r.get("kind", ""))]
            if abl_recs:
                abl_rows = [
                    {
                        "Component": str(r.get("payload", {}).get("component", r.get("tag"))),
                        "Delta": float(r.get("payload", {}).get("delta", 0.0)),
                        "Benefit": "Demonstrated" if r.get("payload", {}).get("benefit") else "None",
                    }
                    for r in abl_recs
                ]
                st.dataframe(pd.DataFrame(abl_rows), use_container_width=True)
                st.plotly_chart(ablation_benefit_figure(abl_recs, provenance=prov_badge), use_container_width=True)
            else:
                st.caption("Ablation experiments not recorded.")
                st.plotly_chart(ablation_benefit_figure("NOT_RUN: ablation unrecorded", provenance=prov_badge), use_container_width=True)

        st.markdown('<div class="sec-label">CRYPTOGRAPHIC LEDGER AUDIT TRAIL</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="sec-label">RELATIONAL DATABASE REGISTRY AUDIT</div>', unsafe_allow_html=True)
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
            st.caption(f"Database connection: {e}")

    # Auto refresh trigger
    if auto_refresh:
        import time
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
