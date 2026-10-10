from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from backtesting.metrics.drawdown_analysis.drawdown import (
    drawdown_periods,
    max_drawdown,
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
    sharpe,
    skew,
    sortino,
    summary,
)
from backtesting.metrics.risk_measures.overfit import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


def ascii_equity_curve(
    equity: pd.Series | np.ndarray | list[float],
    width: int = 70,
    height: int = 15,
) -> str:
    """Generate an ASCII equity curve with min and max vertical axis labels.

    Args:
        equity: Portfolio equity series or values.
        width: Total character width of the chart.
        height: Total line height of the chart.

    Returns:
        str: ASCII plot having exactly `height` lines.
    """
    vals = np.asarray(equity, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return "\n".join([" " * width for _ in range(height)])

    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    span = max_val - min_val

    # Reserve 12 characters for numeric axis label + " | "
    label_width = 12
    plot_width = max(10, width - label_width)

    if len(vals) == 1:
        resampled = np.full(plot_width, vals[0])
    else:
        indices = np.linspace(0, len(vals) - 1, plot_width)
        resampled = np.interp(indices, np.arange(len(vals)), vals)

    grid = [[" " for _ in range(plot_width)] for _ in range(height)]

    for c in range(plot_width):
        y = resampled[c]
        if span > 1e-12:
            norm = (y - min_val) / span
            row = int(round((1.0 - norm) * (height - 1)))
            row = min(max(row, 0), height - 1)
        else:
            row = height // 2
        grid[row][c] = "#"

    lines: list[str] = []
    for r in range(height):
        if r == 0:
            prefix = f"{max_val:>9.2f} | "
        elif r == height - 1:
            prefix = f"{min_val:>9.2f} | "
        else:
            prefix = f"{' ':>9} | "
        line = prefix + "".join(grid[r])
        lines.append(line)

    return "\n".join(lines)


def monthly_returns_grid(returns: pd.Series) -> pd.DataFrame:
    """Aggregate periodic returns into a Year x Month grid plus compounded annual returns.

    Args:
        returns: Periodic simple return series.

    Returns:
        pd.DataFrame: Table with columns Jan..Dec and Year, indexed by year.
    """
    s = pd.Series(returns).copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        # Create a synthetic daily business calendar if index is not timestamps
        s.index = pd.date_range("2020-01-01", periods=len(s), freq="B")

    s = s.dropna()
    if s.empty:
        cols = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Year"
        ]
        return pd.DataFrame(columns=cols)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    records: dict[int, dict[str, float]] = {}

    years = sorted(s.index.year.unique())
    for yr in years:
        yr_series = s[s.index.year == yr]
        records[yr] = {}
        for m in range(1, 13):
            m_series = yr_series[yr_series.index.month == m]
            if len(m_series) > 0:
                m_ret = float(np.prod(1.0 + m_series.values) - 1.0)
                records[yr][month_names[m - 1]] = m_ret
            else:
                records[yr][month_names[m - 1]] = np.nan

        # Full year compound return
        records[yr]["Year"] = float(np.prod(1.0 + yr_series.values) - 1.0)

    grid = pd.DataFrame.from_dict(records, orient="index")
    cols = month_names + ["Year"]
    grid = grid.reindex(columns=cols)
    return grid


def generate_tearsheet(
    result: Any,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    n_trials: int = 1,
    trial_sharpes: list[float] | None = None,
) -> str:
    """Generate a comprehensive institutional quant tearsheet report.

    Sections:
        1. Summary Performance Metrics
        2. Overfitting & Probabilistic Sharpe (PSR / DSR)
        3. Bootstrap Sharpe Confidence Interval
        4. Top 5 Drawdowns Table
        5. Monthly Returns Grid
        6. Trade Statistics
        7. ASCII Equity Curve

    Args:
        result: BacktestResult instance or dict/DataFrame containing backtest data.
        benchmark: Optional benchmark equity or return series.
        rf: Risk-free rate (annualized). Default is 0.0.
        n_trials: Number of trial strategies evaluated. Default is 1.
        trial_sharpes: Optional list of Sharpe ratios across all trial backtests for DSR.

    Returns:
        str: Full formatted tearsheet text.
    """
    # Extract equity curve
    if hasattr(result, "equity_curve"):
        df_eq = result.equity_curve
        trades = getattr(result, "trades", [])
    elif isinstance(result, pd.DataFrame):
        df_eq = result
        trades = []
    elif isinstance(result, dict):
        df_eq = result.get("equity_curve", pd.DataFrame())
        trades = result.get("trades", [])
    else:
        df_eq = pd.DataFrame()
        trades = []

    if isinstance(df_eq, pd.DataFrame) and "equity" in df_eq.columns:
        equity_series = df_eq["equity"]
    elif isinstance(df_eq, pd.Series):
        equity_series = df_eq
    else:
        equity_series = pd.Series([1.0])

    equity_vals = equity_series.values.astype(float)
    if len(equity_vals) > 1:
        rets = np.diff(equity_vals) / equity_vals[:-1]
    else:
        rets = np.array([0.0])

    ret_series = pd.Series(rets, index=equity_series.index[1:] if len(equity_series) > 1 else equity_series.index)

    # 1. Summary Metrics
    summ = summary(ret_series, rf=rf, ppy=252)
    mdd = max_drawdown(equity_series)

    # 2. Overfitting Statistics
    sr_hat = sharpe(ret_series, rf=rf, ppy=252, annualized=False)
    T = len(ret_series)
    sk = float(skew(ret_series))
    kt = float(kurtosis(ret_series))

    psr_val = probabilistic_sharpe_ratio(
        sr_hat=sr_hat,
        sr_benchmark=0.0,
        T=T,
        skew=sk,
        kurt=kt,
    )

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        n_t = len(trial_sharpes)
        var_sr = float(np.var(trial_sharpes, ddof=1))
        dsr_val = deflated_sharpe_ratio(
            sr_hat=sr_hat,
            T=T,
            skew=sk,
            kurt=kt,
            n_trials=n_t,
            var_sr=var_sr,
        )
        dsr_str = f"{dsr_val:.4f} (trials={n_t}, var_sr={var_sr:.6f})"
    elif n_trials > 1:
        dsr_str = f"Not computed (trial_sharpes variance not provided, n_trials={n_trials})"
    else:
        dsr_str = "Not computed (trial_sharpes not provided, n_trials=1)"

    # 3. Bootstrap CI
    ci_lower, ci_upper = bootstrap_sharpe_ci(ret_series, n_boot=1000, block=10, alpha=0.05, seed=42)

    # 4. Drawdowns
    dd_df = drawdown_periods(equity_series)
    if not dd_df.empty:
        top_5_dd = dd_df.sort_values(by="depth").head(5)
    else:
        top_5_dd = pd.DataFrame()

    # 5. Monthly Returns Grid
    m_grid = monthly_returns_grid(ret_series)

    # 6. Trade Statistics
    if trades:
        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        wins = [p for p in pnls if p > 0.0]
        losses = [p for p in pnls if p < 0.0]
        total_trades = len(pnls)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (float("inf") if gross_profit > 0 else 0.0)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        total_volume = sum(float(t.get("qty", 0.0)) * float(t.get("exit_price", 0.0)) for t in trades)
        avg_equity = float(np.mean(equity_vals)) if len(equity_vals) > 0 else 1.0
        turnover = total_volume / avg_equity if avg_equity > 1e-12 else 0.0
    else:
        total_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        turnover = 0.0

    # 7. ASCII Curve
    curve = ascii_equity_curve(equity_series, width=70, height=15)

    buf = io.StringIO()
    buf.write("=" * 70 + "\n")
    buf.write("                  TAPEWRIGHT-QUANT TEARSHEET REPORT\n")
    buf.write("=" * 70 + "\n\n")

    # Section 1
    buf.write("=== SUMMARY PERFORMANCE METRICS ===\n")
    buf.write(f"  Annualized Return (CAGR) : {summ['annualized_return']:>10.2%}\n")
    buf.write(f"  Annualized Volatility    : {summ['annualized_vol']:>10.2%}\n")
    buf.write(f"  Sharpe Ratio (Rf={rf:.1%})  : {summ['sharpe']:>10.4f}\n")
    buf.write(f"  Sortino Ratio            : {summ['sortino']:>10.4f}\n")
    buf.write(f"  Calmar Ratio             : {summ['calmar']:>10.4f}\n")
    buf.write(f"  Max Drawdown             : {mdd:>10.2%}\n")
    buf.write(f"  Historical VaR (99%)     : {summ['historical_var_99']:>10.2%}\n")
    buf.write(f"  Historical CVaR (99%)    : {summ['cvar_99']:>10.2%}\n")
    buf.write(f"  Cornish-Fisher VaR (99%) : {summ['cornish_fisher_var_99']:>10.2%}\n")
    buf.write(f"  Omega Ratio              : {summ['omega_ratio']:>10.4f}\n")
    buf.write(f"  Skewness                 : {summ['skew']:>10.4f}\n")
    buf.write(f"  Raw Kurtosis             : {summ['kurtosis']:>10.4f}\n\n")

    # Section 2
    buf.write("=== OVERFITTING & PROBABILISTIC SHARPE (PSR/DSR) ===\n")
    buf.write(f"  Probabilistic Sharpe (PSR) : {psr_val:>10.4f}\n")
    buf.write(f"  Deflated Sharpe (DSR)      : {dsr_str}\n\n")

    # Section 3
    buf.write("=== BOOTSTRAP SHARPE CONFIDENCE INTERVAL ===\n")
    buf.write(f"  95% Block Bootstrap CI     : [{ci_lower:.4f}, {ci_upper:.4f}]\n\n")

    # Section 4
    buf.write("=== TOP 5 DRAWDOWN PERIODS ===\n")
    if not top_5_dd.empty:
        buf.write(top_5_dd.to_string(index=False) + "\n\n")
    else:
        buf.write("  No drawdown episodes recorded.\n\n")

    # Section 5
    buf.write("=== MONTHLY RETURNS GRID ===\n")
    if not m_grid.empty:
        formatted_grid = m_grid.map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        buf.write(formatted_grid.to_string() + "\n\n")
    else:
        buf.write("  No monthly returns data.\n\n")

    # Section 6
    buf.write("=== TRADE STATISTICS ===\n")
    buf.write(f"  Total Closed Trades      : {total_trades:>10d}\n")
    buf.write(f"  Win Rate                 : {win_rate:>10.2%}\n")
    buf.write(f"  Profit Factor            : {profit_factor:>10.4f}\n")
    buf.write(f"  Average Win              : {avg_win:>10.2f}\n")
    buf.write(f"  Average Loss             : {avg_loss:>10.2f}\n")
    buf.write(f"  Portfolio Turnover       : {turnover:>10.2f}\n\n")

    # Section 7
    buf.write("=== ASCII EQUITY CURVE ===\n")
    buf.write(curve + "\n")
    buf.write("=" * 70 + "\n")

    return buf.getvalue()


def save_tearsheet(tearsheet_text: str, filepath: str | Path) -> None:
    """Save tearsheet text output to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tearsheet_text)


def log_snapshots_to_db(
    result: Any,
    portfolio_repo: Any,
    portfolio_name: str = "backtest",
) -> int:
    """Persist portfolio equity snapshots into the database via PortfolioRepo."""
    if hasattr(result, "equity_curve"):
        df_eq = result.equity_curve
    elif isinstance(result, pd.DataFrame):
        df_eq = result
    elif isinstance(result, pd.Series):
        df_eq = pd.DataFrame({"equity": result})
    else:
        return 0

    if df_eq.empty:
        return 0

    count = 0
    for idx_val, row in df_eq.iterrows():
        if isinstance(idx_val, (pd.Timestamp, np.datetime64)):
            ts = str(idx_val)
        elif isinstance(idx_val, str):
            ts = idx_val
        else:
            ts = None
        nav = float(row["equity"]) if "equity" in row else float(row.iloc[0])
        cash = float(row["cash"]) if "cash" in row else nav
        unrealized = float(row["unrealized_pnl"]) if "unrealized_pnl" in row else 0.0
        realized = float(row["realized_pnl"]) if "realized_pnl" in row else 0.0

        portfolio_repo.insert_snapshot(
            portfolio_name=portfolio_name,
            total_nav=nav,
            cash_balance=cash,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            timestamp=ts,
        )
        count += 1

    return count
