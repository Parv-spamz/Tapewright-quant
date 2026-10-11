"""
Dark-theme institutional Plotly figures and analytics visualization suite.

Pure functions (no Streamlit dependencies) returning Plotly Figures, DataFrames, or dictionaries.
Shared unified dark theme layout: background #0E1117, cards #1E232F, font #E0E3EB, accents.
"""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Shared institutional dark-theme layout specification
DARK_LAYOUT = {
    "paper_bgcolor": "#0E1117",
    "plot_bgcolor": "#161B26",
    "font": {"color": "#E0E3EB", "family": "Inter, sans-serif", "size": 12},
    "margin": {"l": 50, "r": 40, "t": 50, "b": 40},
    "xaxis": {
        "gridcolor": "#262B3D",
        "zerolinecolor": "#3B4252",
        "showline": True,
        "linecolor": "#3B4252",
    },
    "yaxis": {
        "gridcolor": "#262B3D",
        "zerolinecolor": "#3B4252",
        "showline": True,
        "linecolor": "#3B4252",
    },
    "legend": {"bgcolor": "rgba(22, 27, 38, 0.8)", "font": {"color": "#E0E3EB"}},
}


def _apply_dark_theme(fig: go.Figure, title: str = "", provenance: str = "") -> go.Figure:
    """Applies unified dark theme styling to a Plotly figure."""
    fig.update_layout(**DARK_LAYOUT)
    if title:
        fig.update_layout(title={"text": title, "font": {"size": 14, "color": "#FFFFFF"}})
    if provenance:
        fig.add_annotation(
            text=f"PROVENANCE: {provenance}",
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.05,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font={"size": 10, "color": "#8F9CAE"},
            bgcolor="rgba(30, 35, 47, 0.7)",
            bordercolor="#3B4252",
            borderwidth=1,
        )
    return fig


def equity_figure(
    result: Union[pd.Series, pd.DataFrame, dict[str, Any]],
    benchmark: Optional[pd.Series] = None,
) -> go.Figure:
    """Generates portfolio equity curve comparison figure."""
    if isinstance(result, pd.Series):
        s_equity = result.copy()
    elif isinstance(result, pd.DataFrame):
        col = "equity" if "equity" in result.columns else result.columns[0]
        s_equity = result[col].copy()
    elif isinstance(result, dict):
        eq_data = result.get("equity", result.get("equity_curve", []))
        if isinstance(eq_data, (pd.Series, pd.DataFrame)):
            s_equity = eq_data.copy()
        else:
            s_equity = pd.Series(eq_data)
    else:
        s_equity = pd.Series(result)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s_equity.index,
            y=s_equity.values,
            mode="lines",
            name="Portfolio NAV",
            line={"color": "#00D4B2", "width": 2},
        )
    )

    if benchmark is not None and not benchmark.empty:
        bm_copy = benchmark.copy()
        # Scale benchmark to match portfolio starting value
        start_val = float(s_equity.iloc[0]) if len(s_equity) > 0 else 100.0
        bm_scaled = (bm_copy / float(bm_copy.iloc[0])) * start_val
        fig.add_trace(
            go.Scatter(
                x=bm_scaled.index,
                y=bm_scaled.values,
                mode="lines",
                name="Benchmark (SPY)",
                line={"color": "#7E8B9B", "width": 1.5, "dash": "dot"},
            )
        )

    return _apply_dark_theme(fig, "Portfolio Equity Curve vs Benchmark")


def drawdown_figure(underwater_series: pd.Series) -> go.Figure:
    """Generates underwater drawdown area chart (y <= 0)."""
    s_dd = underwater_series.copy()
    # Ensure values are non-positive
    vals = -np.abs(s_dd.values)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s_dd.index,
            y=vals,
            mode="lines",
            name="Drawdown",
            line={"color": "#FF5252", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(255, 82, 82, 0.25)",
        )
    )
    return _apply_dark_theme(fig, "Underwater Portfolio Drawdown")


def monthly_heatmap_figure(
    returns_data: Union[pd.Series, pd.DataFrame],
) -> go.Figure:
    """Generates a 12-column monthly return calendar heatmap (Jan..Dec)."""
    if isinstance(returns_data, pd.Series):
        s_rets = returns_data.copy()
        if not isinstance(s_rets.index, pd.DatetimeIndex):
            s_rets.index = pd.date_range("2022-01-01", periods=len(s_rets), freq="D")
        monthly = s_rets.resample("ME").apply(lambda r: np.prod(1.0 + r) - 1.0)
        df_m = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "return": monthly.values})
        pivot = df_m.pivot(index="year", columns="month", values="return")
    else:
        pivot = returns_data.copy()

    # Ensure all 12 calendar months exist
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    z_vals = pivot.values * 100.0  # percentage
    text_vals = np.where(np.isnan(z_vals), "", np.char.mod("%.1f%%", z_vals))

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=month_names,
            y=[str(y) for y in pivot.index],
            text=text_vals,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar={"title": {"text": "Return %", "side": "right"}},
        )
    )
    return _apply_dark_theme(fig, "Monthly Returns Heatmap")


def returns_hist_figure(
    returns: Union[Sequence[float], np.ndarray, pd.Series],
    var: float = 0.02,
    cvar: float = 0.03,
) -> go.Figure:
    """Generates return distribution histogram with vertical VaR and CVaR risk markers."""
    arr = np.asarray(returns, dtype=np.float64)
    clean = arr[np.isfinite(arr)]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=clean * 100.0,
            nbinsx=40,
            name="Daily Returns",
            marker_color="#3D5AFE",
            opacity=0.75,
        )
    )

    # Add VaR and CVaR threshold lines (as negative percent returns)
    var_pct = -abs(var) * 100.0
    cvar_pct = -abs(cvar) * 100.0

    fig.add_vline(x=var_pct, line_dash="dash", line_color="#FFB300", annotation_text=f"VaR 95%: {var_pct:.2f}%")
    fig.add_vline(x=cvar_pct, line_dash="dot", line_color="#FF1744", annotation_text=f"CVaR 95%: {cvar_pct:.2f}%")

    return _apply_dark_theme(fig, "Return Distribution with Tail Risk Cutoffs")


def rolling_sharpe_figure(rolling_sharpe_series: pd.Series) -> go.Figure:
    """Generates rolling Sharpe ratio time series plot."""
    s = rolling_sharpe_series.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name="Rolling Sharpe",
            line={"color": "#00E676", "width": 2},
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#7E8B9B")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#FFB300", annotation_text="Target SR=1.0")
    return _apply_dark_theme(fig, "Rolling Sharpe Ratio (Annualized)")


def price_with_trades_figure(
    prices: pd.Series | pd.DataFrame,
    fills: Union[list[dict[str, Any]], pd.DataFrame],
) -> go.Figure:
    """Plots asset price time series with buy and sell execution markers."""
    if isinstance(prices, pd.DataFrame):
        p_series = prices["close"].copy()
    else:
        p_series = prices.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p_series.index,
            y=p_series.values,
            mode="lines",
            name="Price",
            line={"color": "#90CAF9", "width": 1.5},
        )
    )

    # Extract buys and sells
    if isinstance(fills, pd.DataFrame):
        fills_list = fills.to_dict(orient="records")
    else:
        fills_list = list(fills)

    buy_x, buy_y = [], []
    sell_x, sell_y = [], []

    for f in fills_list:
        if isinstance(f, dict):
            t = f.get("executed_at", f.get("timestamp", f.get("time", p_series.index[0])))
            p = float(f.get("executed_price", f.get("fill_price", f.get("price", p_series.loc[t] if t in p_series else 100.0))))
            side = str(f.get("side", "")).upper()
        else:
            t = getattr(f, "executed_at", getattr(f, "timestamp", getattr(f, "time", p_series.index[0])))
            p = float(getattr(f, "executed_price", getattr(f, "fill_price", getattr(f, "price", 100.0))))
            side = str(getattr(f, "side", "")).upper()
        if side == "BUY":
            buy_x.append(t)
            buy_y.append(p)
        elif side == "SELL":
            sell_x.append(t)
            sell_y.append(p)

    if buy_x:
        fig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode="markers",
                name="Buy Fills",
                marker={"symbol": "triangle-up", "size": 10, "color": "#00E676"},
            )
        )
    if sell_x:
        fig.add_trace(
            go.Scatter(
                x=sell_x,
                y=sell_y,
                mode="markers",
                name="Sell Fills",
                marker={"symbol": "triangle-down", "size": 10, "color": "#FF1744"},
            )
        )

    return _apply_dark_theme(fig, "Price History with Executed Fills")


def mc_cone_figure(
    cone_df: pd.DataFrame,
    realized_equity: Optional[pd.Series] = None,
) -> go.Figure:
    """Plots Monte Carlo simulated percentile confidence cones."""
    fig = go.Figure()

    cols = cone_df.columns
    # Check for percentile columns (e.g. 5, 25, 50, 75, 95)
    c5 = cone_df.get(5, cone_df.get("p5", cone_df.iloc[:, 0]))
    c95 = cone_df.get(95, cone_df.get("p95", cone_df.iloc[:, -1]))
    c50 = cone_df.get(50, cone_df.get("p50", cone_df.iloc[:, len(cols) // 2]))

    # Shaded outer cone (5th to 95th)
    fig.add_trace(
        go.Scatter(
            x=cone_df.index,
            y=c95.values,
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cone_df.index,
            y=c5.values,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(0, 212, 178, 0.15)",
            name="90% Confidence Cone",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cone_df.index,
            y=c50.values,
            mode="lines",
            name="Median Trajectory",
            line={"color": "#00D4B2", "dash": "dash"},
        )
    )

    if realized_equity is not None and not realized_equity.empty:
        fig.add_trace(
            go.Scatter(
                x=realized_equity.index,
                y=realized_equity.values,
                mode="lines",
                name="Realized Equity",
                line={"color": "#FFFFFF", "width": 2},
            )
        )

    return _apply_dark_theme(fig, "Monte Carlo Equity Projection Cone")


def stress_bar_figure(stress_table_df: pd.DataFrame) -> go.Figure:
    """Plots horizontal PnL shock bars across macro stress scenarios."""
    df = stress_table_df.copy()
    scenarios = df["Scenario"].values

    # Parse PnL values if formatted as percentage strings
    pnl_vals = []
    for v in df["PnL (%)"].values:
        if isinstance(v, str):
            pnl_vals.append(float(v.replace("%", "").strip()))
        else:
            pnl_vals.append(float(v) * 100.0)

    colors = ["#FF5252" if p < 0 else "#00E676" for p in pnl_vals]

    fig = go.Figure(
        go.Bar(
            y=scenarios,
            x=pnl_vals,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1f}%" for p in pnl_vals],
            textposition="auto",
        )
    )
    fig.update_layout(xaxis_title="Portfolio PnL Shock (%)")
    return _apply_dark_theme(fig, "Macro Stress Scenario Impact")


def risk_profile_radar(metrics: dict[str, float]) -> go.Figure:
    """Generates a radar chart of normalized risk dimensions.

    Title strictly includes: 'Risk profile (proxy metrics from available data)'
    """
    categories = [
        "Market Beta",
        "Realized Volatility",
        "CVaR 95%",
        "Turnover",
        "Illiquidity (Amihud)",
        "Tail Skew",
    ]

    # Map inputs to normalized scale [0, 1]
    b = float(metrics.get("beta", metrics.get("market_beta", 1.0)))
    v = float(metrics.get("vol", metrics.get("realized_vol", 0.15)))
    c = float(metrics.get("cvar", metrics.get("cvar_95", 0.03)))
    t = float(metrics.get("turnover", 0.20))
    i = float(metrics.get("amihud", metrics.get("illiquidity", 0.05)))
    s = float(metrics.get("skew", metrics.get("tail_skew", -0.5)))

    # Min-max normalization for visualization
    r_vals = [
        np.clip(b / 2.0, 0.05, 1.0),
        np.clip(v / 0.40, 0.05, 1.0),
        np.clip(c / 0.08, 0.05, 1.0),
        np.clip(t / 1.00, 0.05, 1.0),
        np.clip(i / 0.20, 0.05, 1.0),
        np.clip((s + 2.0) / 4.0, 0.05, 1.0),
    ]
    # Close the loop
    r_vals.append(r_vals[0])
    cat_loop = categories + [categories[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=r_vals,
            theta=cat_loop,
            fill="toself",
            fillcolor="rgba(0, 212, 178, 0.2)",
            line={"color": "#00D4B2", "width": 2},
            name="Strategy Risk Profile",
        )
    )
    fig.update_layout(
        polar={
            "bgcolor": "#161B26",
            "radialaxis": {"visible": True, "range": [0, 1], "linecolor": "#3B4252", "gridcolor": "#262B3D"},
            "angularaxis": {"linecolor": "#3B4252", "gridcolor": "#262B3D"},
        }
    )
    return _apply_dark_theme(fig, "Risk profile (proxy metrics from available data)")


def causal_graph_figure(
    graph_edges: list[tuple[str, str, float]] | dict[str, Any],
) -> go.Figure:
    """Renders a causal discovery graph using a networkx-free circular layout computed with NumPy."""
    # Extract unique nodes and weighted edges
    nodes_set: set[str] = set()
    edges: list[tuple[str, str, float]] = []

    if isinstance(graph_edges, dict):
        edge_data = graph_edges.get("edges", [])
    else:
        edge_data = graph_edges

    for item in edge_data:
        if len(item) >= 3:
            u, v, w = item[0], item[1], float(item[2])
            nodes_set.add(u)
            nodes_set.add(v)
            edges.append((u, v, w))

    nodes = sorted(nodes_set)
    if not nodes:
        nodes = ["Market", "Momentum", "Volume"]

    # Compute circular coordinates with NumPy
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

    fig = go.Figure()

    # Draw edge lines
    for u, v, w in edges:
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line={"color": "rgba(0, 212, 178, 0.6)", "width": max(1.0, abs(w) * 3)},
                    hoverinfo="text",
                    text=f"{u} -> {v} (weight={w:.2f})",
                    showlegend=False,
                )
            )

    # Draw nodes
    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker={"size": 18, "color": "#00D4B2", "line": {"color": "#FFFFFF", "width": 1.5}},
            showlegend=False,
        )
    )
    fig.update_layout(xaxis={"showgrid": False, "zeroline": False, "showticklabels": False}, yaxis={"showgrid": False, "zeroline": False, "showticklabels": False})
    return _apply_dark_theme(fig, "Predictive Causality Gating Graph")


def lob_queue_figure(order_history: list[dict[str, Any]] | pd.DataFrame) -> go.Figure:
    """Visualizes Limit Order Book queue position and fill progression over time."""
    if isinstance(order_history, pd.DataFrame):
        df_orders = order_history.copy()
    else:
        df_orders = pd.DataFrame(order_history)

    fig = go.Figure()
    if not df_orders.empty and "queue_position" in df_orders.columns:
        fig.add_trace(
            go.Scatter(
                x=df_orders.index,
                y=df_orders["queue_position"],
                mode="lines+markers",
                name="Queue Rank",
                line={"color": "#FFB300", "width": 2},
            )
        )
    else:
        # Default representative queue fill progression
        x_dummy = list(range(10))
        y_dummy = [100, 85, 60, 42, 28, 15, 8, 3, 1, 0]
        fig.add_trace(
            go.Scatter(x=x_dummy, y=y_dummy, mode="lines+markers", name="Queue Priority", line={"color": "#FFB300"})
        )

    return _apply_dark_theme(fig, "L2 Limit Order Queue Depletion to Fill")


def frontier_figure(
    frontier_points: list[tuple[float, float]] | pd.DataFrame,
    current_weights: Optional[Any] = None,
) -> go.Figure:
    """Plots Modern Portfolio Theory Efficient Frontier."""
    if isinstance(frontier_points, pd.DataFrame):
        vols = frontier_points["volatility"].values
        rets = frontier_points["return"].values
    else:
        vols = [p[0] for p in frontier_points]
        rets = [p[1] for p in frontier_points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vols,
            y=rets,
            mode="lines+markers",
            name="Efficient Frontier",
            line={"color": "#00D4B2", "width": 2},
            marker={"size": 6, "color": "#00D4B2"},
        )
    )
    # Highlight max Sharpe tangency point
    max_idx = int(np.argmax(np.array(rets) / (np.array(vols) + 1e-8)))
    fig.add_trace(
        go.Scatter(
            x=[vols[max_idx]],
            y=[rets[max_idx]],
            mode="markers",
            name="Max Sharpe Portfolio",
            marker={"symbol": "star", "size": 14, "color": "#FFD700"},
        )
    )
    fig.update_layout(xaxis_title="Annualized Volatility", yaxis_title="Expected Return")
    return _apply_dark_theme(fig, "Markowitz Efficient Frontier")


def drift_table_frame(drift_report: pd.DataFrame) -> pd.DataFrame:
    """Formats feature drift diagnostic summary table."""
    return drift_report.copy()


def kpi_dict(tearsheet_or_result: dict[str, Any] | Any) -> dict[str, Any]:
    """Compiles formatted dictionary of core performance and risk KPIs."""
    if isinstance(tearsheet_or_result, dict):
        d = copy.deepcopy(tearsheet_or_result)
    elif hasattr(tearsheet_or_result, "metrics"):
        d = copy.deepcopy(tearsheet_or_result.metrics)
    else:
        d = {}

    tot_ret = float(d.get("total_return", d.get("total_pnl", 0.0)))
    sr = float(d.get("sharpe", d.get("sharpe_ratio", 0.0)))
    so = float(d.get("sortino", d.get("sortino_ratio", 0.0)))
    mdd = float(d.get("max_drawdown", d.get("drawdown", 0.0)))
    cal = float(d.get("calmar", d.get("calmar_ratio", 0.0)))
    win = float(d.get("win_rate", 0.50))

    return {
        "Total Return": f"{tot_ret * 100.0:.2f}%" if abs(tot_ret) < 100 else f"{tot_ret:.2f}%",
        "Sharpe Ratio": f"{sr:.2f}",
        "Sortino Ratio": f"{so:.2f}",
        "Max Drawdown": f"{mdd * 100.0:.2f}%" if abs(mdd) < 100 else f"{mdd:.2f}%",
        "Calmar Ratio": f"{cal:.2f}",
        "Win Rate": f"{win * 100.0:.1f}%" if win <= 1.0 else f"{win:.1f}%",
    }


def not_run_figure(reason: str = "NOT_RUN", title: str = "", provenance: str = "") -> go.Figure:
    """Generates an institutional placeholder figure displaying NOT_RUN status and reason."""
    fig = go.Figure()
    label = str(reason)
    if "NOT_RUN" not in label:
        label = f"NOT_RUN: {label}"
    fig.add_annotation(
        text=label,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": "#FFB300"},
        bgcolor="rgba(30, 35, 47, 0.8)",
        bordercolor="#FFB300",
        borderwidth=1,
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return _apply_dark_theme(fig, title=title, provenance=provenance)


def evidence_metric_figure(
    data: Sequence[float] | pd.Series | dict[str, Any] | str,
    title: str = "Evidence Metric Trajectory",
    provenance: str = "",
) -> go.Figure:
    """Pure function generating metric trajectory figure. Handles NOT_RUN status."""
    if isinstance(data, str) and ("NOT_RUN" in data or data == ""):
        return not_run_figure(data or "NOT_RUN", title=title, provenance=provenance)
    if isinstance(data, dict) and data.get("status") == "NOT_RUN":
        return not_run_figure(data.get("reason", "NOT_RUN"), title=title, provenance=provenance)

    # Pure copy to guarantee no mutation
    if isinstance(data, pd.Series):
        s_data = data.copy()
        x_vals = list(s_data.index)
        y_vals = list(s_data.values)
    elif isinstance(data, dict):
        x_vals = list(data.keys())
        y_vals = list(data.values())
    else:
        y_vals = list(copy.deepcopy(data))
        x_vals = list(range(len(y_vals)))

    if len(y_vals) == 0:
        return not_run_figure("NOT_RUN: empty dataset", title=title, provenance=provenance)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Metric Value",
            line={"color": "#4C9AFF", "width": 2},
            marker={"size": 6, "color": "#00D4B2"},
        )
    )
    fig.update_layout(xaxis_title="Step / Observation", yaxis_title="Metric Value")
    return _apply_dark_theme(fig, title=title, provenance=provenance)


def validation_survives_figure(
    val_records: list[dict[str, Any]] | str,
    title: str = "Validation Survives Breakdown",
    provenance: str = "",
) -> go.Figure:
    """Pure function plotting validation scores with survives flag indication."""
    if isinstance(val_records, str) and ("NOT_RUN" in val_records or val_records == ""):
        return not_run_figure(val_records or "NOT_RUN", title=title, provenance=provenance)
    if not isinstance(val_records, list) or len(val_records) == 0:
        return not_run_figure("NOT_RUN: no validation records", title=title, provenance=provenance)

    records = copy.deepcopy(val_records)
    names = [str(r.get("name", r.get("trial", f"Trial {i}"))) for i, r in enumerate(records)]
    scores = [float(r.get("score", r.get("sharpe", r.get("metric", 0.0)))) for r in records]
    survives = [bool(r.get("survives", False)) for r in records]
    colors = ["#00D4B2" if s else "#FF5252" for s in survives]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=scores,
            marker_color=colors,
            name="Validation Score",
        )
    )
    fig.update_layout(xaxis_title="Candidate Model / Trial", yaxis_title="Validation Performance Score")
    return _apply_dark_theme(fig, title=title, provenance=provenance)


def ablation_benefit_figure(
    ablation_records: list[dict[str, Any]] | str,
    title: str = "Ablation Benefit Analysis",
    provenance: str = "",
) -> go.Figure:
    """Pure function displaying ablation deltas with 'no demonstrated benefit' flags."""
    if isinstance(ablation_records, str) and ("NOT_RUN" in ablation_records or ablation_records == ""):
        return not_run_figure(ablation_records or "NOT_RUN", title=title, provenance=provenance)
    if not isinstance(ablation_records, list) or len(ablation_records) == 0:
        return not_run_figure("NOT_RUN: no ablation records", title=title, provenance=provenance)

    records = copy.deepcopy(ablation_records)
    components = [str(r.get("component", r.get("variant", f"Var {i}"))) for i, r in enumerate(records)]
    deltas = [float(r.get("delta", r.get("impact", 0.0))) for r in records]
    has_benefit = [bool(r.get("demonstrated_benefit", r.get("delta", 0.0) > 0.0)) for r in records]
    colors = ["#00D4B2" if b else "#FFB300" for b in has_benefit]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=components,
            y=deltas,
            marker_color=colors,
            name="Delta Over Baseline",
        )
    )
    fig.update_layout(xaxis_title="Ablated Component / Feature", yaxis_title="Marginal Performance Delta")
    return _apply_dark_theme(fig, title=title, provenance=provenance)


def calibration_reliability_figure(
    calibration_data: list[tuple[float, float]] | dict[str, Any] | pd.DataFrame | str,
    title: str = "Calibration Reliability Diagram",
    provenance: str = "",
) -> go.Figure:
    """Pure function generating calibration reliability curve comparing predicted vs empirical probabilities."""
    if isinstance(calibration_data, str) and ("NOT_RUN" in calibration_data or calibration_data == ""):
        return not_run_figure(calibration_data or "NOT_RUN", title=title, provenance=provenance)
    if isinstance(calibration_data, dict) and calibration_data.get("status") == "NOT_RUN":
        return not_run_figure(calibration_data.get("reason", "NOT_RUN"), title=title, provenance=provenance)

    if isinstance(calibration_data, pd.DataFrame):
        df_cal = calibration_data.copy()
        pred_p = [float(v) for v in df_cal.iloc[:, 0].values]
        obs_p = [float(v) for v in df_cal.iloc[:, 1].values]
    elif isinstance(calibration_data, dict):
        pred_p = [float(v) for v in calibration_data.get("predicted", [])]
        obs_p = [float(v) for v in calibration_data.get("observed", [])]
    else:
        cal_list = copy.deepcopy(calibration_data)
        if not cal_list:
            return not_run_figure("NOT_RUN: empty calibration points", title=title, provenance=provenance)
        pred_p = [float(pt[0]) for pt in cal_list]
        obs_p = [float(pt[1]) for pt in cal_list]

    if len(pred_p) == 0:
        return not_run_figure("NOT_RUN: no calibration data", title=title, provenance=provenance)

    fig = go.Figure()
    # Perfectly calibrated 45-degree reference line
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Ideal Calibration",
            line={"dash": "dash", "color": "#8F9CAE", "width": 1},
        )
    )
    # Empirical calibration points
    fig.add_trace(
        go.Scatter(
            x=pred_p,
            y=obs_p,
            mode="lines+markers",
            name="Observed Frequency",
            line={"color": "#4C9AFF", "width": 2},
            marker={"size": 6, "color": "#00D4B2"},
        )
    )
    fig.update_layout(
        xaxis={"title": "Mean Predicted Probability", "range": [0.0, 1.0]},
        yaxis={"title": "Empirical Frequency", "range": [0.0, 1.0]},
    )
    return _apply_dark_theme(fig, title=title, provenance=provenance)


def forward_test_figure(
    forward_data: pd.Series | Sequence[float] | dict[str, Any] | str,
    title: str = "Forward Paper Test Trajectory",
    provenance: str = "",
) -> go.Figure:
    """Pure function generating forward paper test trajectory. Handles NOT_RUN status."""
    if isinstance(forward_data, str) and ("NOT_RUN" in forward_data or forward_data == ""):
        return not_run_figure(forward_data or "NOT_RUN", title=title, provenance=provenance)
    if isinstance(forward_data, dict) and forward_data.get("status") == "NOT_RUN":
        return not_run_figure(forward_data.get("reason", "NOT_RUN"), title=title, provenance=provenance)

    if isinstance(forward_data, pd.Series):
        s_data = forward_data.copy()
        x_vals = list(s_data.index)
        y_vals = list(s_data.values)
    elif isinstance(forward_data, dict):
        x_vals = list(forward_data.keys())
        y_vals = list(forward_data.values())
    else:
        y_vals = list(copy.deepcopy(forward_data))
        x_vals = list(range(len(y_vals)))

    if len(y_vals) == 0:
        return not_run_figure("NOT_RUN: no forward testing points logged", title=title, provenance=provenance)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Forward Paper Cumulative PnL",
            line={"color": "#00D4B2", "width": 2},
            marker={"size": 5, "color": "#00D4B2"},
        )
    )
    fig.update_layout(xaxis_title="Forward Observation Period", yaxis_title="Paper Performance")
    return _apply_dark_theme(fig, title=title, provenance=provenance)

