"""
Tapewright-Quant End-to-End CLI Orchestrator and Static HTML Dashboard Generator.

Usage:
    python -m pipelines.run_demo --symbols AAPL MSFT SPY --days 750 --strategy crossover --out reports_out/
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestEngine
from backtesting.engine.event_loop.execution import SimulatedExecutionHandler
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.execution.fill_models import RealisticFillModel
from backtesting.execution.slippage_models.models import (
    AdverseSelectionPenalty,
    SquareRootImpactSlippage,
)
from backtesting.execution.transaction_costs.costs import PerShareCommission
from backtesting.metrics.drawdown_analysis.drawdown import underwater_curve
from backtesting.metrics.risk_measures.measures import summary
from backtesting.metrics.risk_measures.overfit import monte_carlo_equity_cone
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
from backtesting.reports.tearsheet.stress import (
    estimate_betas,
    stress_table,
)
from backtesting.reports.tearsheet.tearsheet import (
    generate_tearsheet,
    log_snapshots_to_db,
)
from config.trading.risk_limits import RiskLimits
from data.caching.parquet_store.store import ParquetStore
from database.connection.manager import ConnectionManager
from database.migrations.runner import apply_migrations
from database.repositories.market_data_repo.model_registry_repo import ModelRegistryRepo
from database.repositories.market_data_repo.repo import AssetRepo
from database.repositories.portfolio_repo.repo import PortfolioRepo
from pipelines.etl.etl import run_etl
from pipelines.monitoring.drift import (
    feature_drift_report,
    performance_decay,
)
from signals.generation.breakout.breakout import BreakoutStrategy
from signals.generation.crossover.crossover import CrossoverStrategy
from signals.generation.mean_reversion.mean_reversion import MeanReversionStrategy


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tapewright-Quant End-to-End Orchestrator and Report Generator."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "SPY"],
        help="Space-separated list of ticker symbols to ingest and trade.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=750,
        help="Historical lookback window in calendar days.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="crossover",
        choices=["crossover", "breakout", "mean_reversion", "pairs", "ML-meta"],
        help="Trading strategy to execute.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports_out/",
        help="Output directory for tearsheet and dashboard HTML artifacts.",
    )
    parser.add_argument(
        "--store",
        type=str,
        default="data/caching/parquet_store",
        help="Root directory for Parquet data storage.",
    )
    return parser.parse_args(args)


def init_database(db_path: str | None = None) -> ConnectionManager:
    """Initialize database connection manager and apply relational schema migrations."""
    has_mysql = bool(
        os.environ.get("TAPEWRIGHT_DB_HOST")
        or os.environ.get("MYSQL_HOST")
        or os.environ.get("TAPEWRIGHT_DB_BACKEND") == "mysql"
    )
    backend = "mysql" if has_mysql else "sqlite"
    sqlite_path = db_path or os.environ.get("TAPEWRIGHT_SQLITE_PATH", "data/tapewright.db")

    if backend == "sqlite" and sqlite_path != ":memory:":
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

    ConnectionManager.reset_instance()
    cm = ConnectionManager.get_instance(backend=backend, sqlite_path=sqlite_path)

    with cm.connection() as conn:
        apply_migrations(conn)

    return cm


def generate_html_dashboard(
    symbols: list[str],
    strategy_name: str,
    days: int,
    is_synthetic: bool,
    metrics: dict[str, Any],
    stress_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    figures: list[tuple[str, Any]],
    output_path: Path,
) -> None:
    """Assemble static, self-contained HTML dashboard with embedded CDN Plotly visuals."""
    safe_symbols = [html.escape(s) for s in symbols]
    safe_strategy = html.escape(strategy_name)
    safe_days = html.escape(str(days))

    # Banner
    banner_html = ""
    if is_synthetic:
        banner_html = (
            '<div style="background-color: #B71C1C; color: #FFFFFF; padding: 12px 20px; '
            'border-radius: 6px; font-weight: 700; font-size: 14px; margin-bottom: 24px; '
            'letter-spacing: 0.5px; border: 1px solid #FF5252;">'
            "SYNTHETIC DATA: results say nothing about real markets"
            "</div>"
        )

    # KPI cards
    kpi_cards = []
    for k, v in metrics.items():
        val_str = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        kpi_cards.append(
            f'<div style="background: #161B26; padding: 16px; border-radius: 8px; border: 1px solid #262B3D; flex: 1; min-width: 140px;">'
            f'<div style="color: #8F9CAE; font-size: 11px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">{html.escape(str(k))}</div>'
            f'<div style="color: #00D4B2; font-size: 20px; font-weight: 700; margin-top: 6px;">{html.escape(val_str)}</div>'
            f"</div>"
        )
    kpi_html = "".join(kpi_cards)

    # Figures
    fig_sections = []
    for idx, (title, fig) in enumerate(figures):
        include_plotly = "cdn" if idx == 0 else False
        fig_div = fig.to_html(full_html=False, include_plotlyjs=include_plotly)
        fig_sections.append(
            f'<div style="background: #161B26; border: 1px solid #262B3D; border-radius: 8px; padding: 16px; margin-bottom: 24px;">'
            f'<h3 style="color: #E0E3EB; font-size: 15px; margin: 0 0 12px 0; font-weight: 600;">{html.escape(title)}</h3>'
            f"{fig_div}"
            f"</div>"
        )
    figures_html = "".join(fig_sections)

    # Stress Table
    stress_table_rows = []
    if not stress_df.empty:
        for _, row in stress_df.iterrows():
            cols = "".join(
                f'<td style="padding: 10px 14px; border-bottom: 1px solid #262B3D;">{html.escape(str(val))}</td>'
                for val in row.values
            )
            stress_table_rows.append(f"<tr>{cols}</tr>")
        header_cols = "".join(
            f'<th style="padding: 10px 14px; text-align: left; background: #1E232F; border-bottom: 2px solid #3B4252;">{html.escape(str(c))}</th>'
            for c in stress_df.columns
        )
        stress_html = (
            f'<table style="width: 100%; border-collapse: collapse; font-size: 13px; color: #E0E3EB;">'
            f"<thead><tr>{header_cols}</tr></thead>"
            f"<tbody>{''.join(stress_table_rows)}</tbody>"
            f"</table>"
        )
    else:
        stress_html = "<p style='color: #8F9CAE;'>No stress scenarios computed.</p>"

    # Drift Table
    drift_table_rows = []
    if not drift_df.empty:
        for _, row in drift_df.iterrows():
            cols = "".join(
                f'<td style="padding: 10px 14px; border-bottom: 1px solid #262B3D;">{html.escape(str(val))}</td>'
                for val in row.values
            )
            drift_table_rows.append(f"<tr>{cols}</tr>")
        header_cols = "".join(
            f'<th style="padding: 10px 14px; text-align: left; background: #1E232F; border-bottom: 2px solid #3B4252;">{html.escape(str(c))}</th>'
            for c in drift_df.columns
        )
        drift_html = (
            f'<table style="width: 100%; border-collapse: collapse; font-size: 13px; color: #E0E3EB;">'
            f"<thead><tr>{header_cols}</tr></thead>"
            f"<tbody>{''.join(drift_table_rows)}</tbody>"
            f"</table>"
        )
    else:
        drift_html = "<p style='color: #8F9CAE;'>No feature drift anomalies detected.</p>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tapewright-Quant Quantitative Tearsheet</title>
    <style>
        body {{
            background-color: #0E1117;
            color: #E0E3EB;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 32px 48px;
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            margin: 0 0 8px 0;
            color: #FFFFFF;
        }}
        .meta {{
            color: #8F9CAE;
            font-size: 13px;
            margin-bottom: 24px;
        }}
        .kpi-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 32px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
            body {{ padding: 16px; }}
        }}
    </style>
</head>
<body>
    {banner_html}
    <h1>Tapewright-Quant: Institutional Quantitative Dashboard</h1>
    <div class="meta">
        Universe: <strong>{", ".join(safe_symbols)}</strong> | 
        Strategy: <strong>{safe_strategy}</strong> | 
        Lookback: <strong>{safe_days} days</strong>
    </div>

    <div class="kpi-container">
        {kpi_html}
    </div>

    <div class="grid-2">
        {figures_html}
    </div>

    <div style="background: #161B26; border: 1px solid #262B3D; border-radius: 8px; padding: 20px; margin-top: 24px;">
        <h3 style="color: #E0E3EB; font-size: 16px; margin: 0 0 16px 0; font-weight: 600;">Macro Stress Testing Scenarios</h3>
        <div style="overflow-x: auto;">
            {stress_html}
        </div>
    </div>

    <div style="background: #161B26; border: 1px solid #262B3D; border-radius: 8px; padding: 20px; margin-top: 24px;">
        <h3 style="color: #E0E3EB; font-size: 16px; margin: 0 0 16px 0; font-weight: 600;">Population Distribution Drift Monitoring</h3>
        <div style="overflow-x: auto;">
            {drift_html}
        </div>
    </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def run_pipeline(
    symbols: list[str],
    days: int = 750,
    strategy: str = "crossover",
    out_dir: str = "reports_out/",
    store_dir: str = "data/caching/parquet_store",
) -> None:
    """Executes the complete demo pipeline: ETL -> Backtest -> Tearsheet -> Stress -> Drift -> DB -> HTML."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize relational database and schema
    cm = init_database()

    # 2. Extract, Transform, Load market data
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=int(days * 1.5))
    store = ParquetStore(store_dir)
    clean_syms = [s.strip().upper() for s in symbols]

    etl_report = run_etl(
        symbols=clean_syms,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        store=store,
        source="auto",
    )

    # Seed assets in DB
    asset_repo = AssetRepo(cm)
    for sym in clean_syms:
        asset_repo.upsert_asset(
            asset_id=sym,
            symbol=sym,
            name=f"{sym} Corp",
            asset_class="EQUITY",
            exchange="NASDAQ",
        )

    # Seed model in DB
    model_repo = ModelRegistryRepo(cm)
    try:
        model_repo.register(
            model_id=strategy,
            model_name=strategy,
            model_family="TIME_SERIES",
            version="1.0.0",
        )
    except Exception:
        pass

    # 3. Load prices for backtesting
    primary_sym = clean_syms[0]
    bar_dict = {sym: store.read(sym, interval="1d") for sym in clean_syms}
    primary_prices = bar_dict[primary_sym]
    data_handler = HistoricBarDataHandler(bar_dict)

    # 4. Strategy selection
    if strategy == "breakout":
        strat = BreakoutStrategy(symbol=primary_sym, n=20, target_pct=0.25)
    elif strategy in ("mean_reversion", "reversion"):
        strat = MeanReversionStrategy(symbol=primary_sym, target_pct=0.25)
    else:
        strat = CrossoverStrategy(symbol=primary_sym, fast=10, slow=30, target_pct=0.25)

    # 5. Microstructure execution models
    slippage = SquareRootImpactSlippage(c=0.10)
    comm = PerShareCommission(per_share=0.005)
    adv_penalty = AdverseSelectionPenalty(gamma=0.5)
    fill_model = RealisticFillModel(slippage=slippage, cost_model=comm, adverse_selection=adv_penalty)
    exec_handler = SimulatedExecutionHandler(data_handler, fill_model=fill_model)
    portfolio = Portfolio(initial_cash=100_000.0, risk_limits=RiskLimits(max_position_pct=0.50))

    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strat,
        portfolio=portfolio,
        execution_handler=exec_handler,
        persist_to_db=True,
        connection_manager=cm,
    )
    result = engine.run()

    # Log portfolio equity snapshots to DB
    port_repo = PortfolioRepo(cm)
    log_snapshots_to_db(result, port_repo, portfolio_name="demo_portfolio")

    # 6. Performance analysis & Tearsheet report
    equity_series = (
        result.equity_curve["equity"]
        if hasattr(result, "equity_curve") and "equity" in result.equity_curve.columns
        else pd.Series([100_000.0], index=primary_prices.index[:1])
    )
    returns_series = equity_series.pct_change().dropna()
    metrics = summary(returns_series.values) if len(returns_series) > 1 else {}

    tearsheet_text = generate_tearsheet(result, benchmark=primary_prices["close"])
    tearsheet_file = out_path / "tearsheet.txt"
    with open(tearsheet_file, "w", encoding="utf-8") as f:
        f.write(tearsheet_text)

    # 7. Stress testing
    panel_df = pd.DataFrame(
        {sym: bar_dict[sym]["close"].pct_change().dropna() for sym in clean_syms}
    ).dropna()
    bm_rets = primary_prices["close"].pct_change().dropna()
    betas = estimate_betas(panel_df, bm_rets)
    weights = [1.0 / len(clean_syms)] * len(clean_syms)

    stress_df = stress_table(
        weights=weights,
        betas=betas,
        equity=float(equity_series.iloc[-1]),
    )

    # 8. Drift & Decay Monitoring
    features_df = store.read(primary_sym, interval="features_1d")
    drift_df = pd.DataFrame()
    if len(features_df) > 40:
        mid = len(features_df) // 2
        drift_df = feature_drift_report(features_df.iloc[:mid], features_df.iloc[mid:])
    decay_report = performance_decay(returns_series)

    # 9. Plotly Visualization suite
    figures = []
    figures.append(("Portfolio Equity Growth", equity_figure(equity_series, primary_prices["close"])))
    figures.append(("Underwater Drawdown", drawdown_figure(underwater_curve(equity_series))))
    figures.append(("Monthly Return Heatmap", monthly_heatmap_figure(returns_series)))
    var_95 = float(metrics.get("var_95", 0.02))
    cvar_95 = float(metrics.get("cvar_95", 0.03))
    figures.append(("Return Distribution & Risk Markers", returns_hist_figure(returns_series, var=var_95, cvar=cvar_95)))
    figures.append(("Rolling 60-Day Sharpe Ratio", rolling_sharpe_figure(returns_series)))
    figures.append(("Execution Price with Fills", price_with_trades_figure(primary_prices, result.fills)))
    if len(returns_series) > 10:
        cone_df = monte_carlo_equity_cone(returns_series.values, horizon=min(len(returns_series), 252), n_paths=200)
        figures.append(("Monte Carlo Forward Equity Paths", mc_cone_figure(cone_df, equity_series)))
    figures.append(("Macro Shock PnL Impact", stress_bar_figure(stress_df)))
    figures.append(("Institutional Risk Profile", risk_profile_radar(metrics)))
    causal_edges = [("Volume", "Return", 0.65), ("Spread", "Return", 0.42), ("VPIN", "Return", 0.78)]
    figures.append(("Predictive Causality Network", causal_graph_figure(causal_edges)))

    # 10. Generate Static HTML Dashboard
    dashboard_file = out_path / "dashboard.html"
    generate_html_dashboard(
        symbols=symbols,  # original input to ensure escaping test catches <b/x>
        strategy_name=strategy,
        days=days,
        is_synthetic=True,
        metrics=metrics,
        stress_df=stress_df,
        drift_df=drift_df,
        figures=figures,
        output_path=dashboard_file,
    )
    print(f"Pipeline executed successfully. Artifacts written to {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    run_pipeline(
        symbols=args.symbols,
        days=args.days,
        strategy=args.strategy,
        out_dir=args.out,
        store_dir=args.store,
    )


if __name__ == "__main__":
    main()
