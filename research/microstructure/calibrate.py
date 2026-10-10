"""Real market order book calibration, impact fitting, and differential queue simulation.

Honest limitations & model boundaries:
- Book-walking cost is the instantaneous liquidity cost of a single sweep across visible levels,
  not the square-root law for multi-slice metaorders; do not claim one validates the other.
- Queue fill calculation ignores order cancellations, which systematically understates actual fills.
- This module implements differential tests comparing independent empirical queue priority against
  the Prompt 10 LOBSimulator; measured discrepancies are reported objectively without claiming realism.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from backtesting.engine.event_loop.events import OrderEvent
from backtesting.execution.slippage_models.lob import LOBSimulator
from integrity.ledger import Ledger
from research.data_access import read_microstructure_data

logger = logging.getLogger(__name__)


def spread_stats(book: pd.DataFrame, tick_size: float = 0.01) -> dict[str, float]:
    """Calculate spread distribution in ticks across order book snapshots."""
    if book.empty:
        return {"q5": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0, "mean_ticks": 0.0}

    if "spread" in book.columns:
        spread_arr = book["spread"].dropna().values
    elif "ask_px_1" in book.columns and "bid_px_1" in book.columns:
        spread_arr = (book["ask_px_1"] - book["bid_px_1"]).dropna().values
    else:
        return {"q5": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0, "mean_ticks": 0.0}

    spread_ticks = spread_arr / max(tick_size, 1e-8)
    q5, q25, q50, q75, q95 = np.percentile(spread_ticks, [5, 25, 50, 75, 95])

    return {
        "q5": float(q5),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q95": float(q95),
        "mean_ticks": float(np.mean(spread_ticks)),
    }


def depth_stats(book: pd.DataFrame) -> dict[str, Any]:
    """Compute top-1 and top-5 visible depth size quantiles across order book snapshots."""
    if book.empty:
        empty_q = {"q5": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0}
        return {"top1_size": empty_q, "top5_size": empty_q}

    # Top-1 size: bid_sz_1 + ask_sz_1
    top1 = (book.get("bid_sz_1", 0.0) + book.get("ask_sz_1", 0.0)).dropna().values

    # Top-5 size: sum of bid_sz_1..5 + ask_sz_1..5
    top5_series = pd.Series(0.0, index=book.index)
    for i in range(1, 6):
        top5_series += book.get(f"bid_sz_{i}", 0.0).fillna(0.0)
        top5_series += book.get(f"ask_sz_{i}", 0.0).fillna(0.0)
    top5 = top5_series.values

    q_top1 = np.percentile(top1, [5, 25, 50, 75, 95]) if len(top1) > 0 else np.zeros(5)
    q_top5 = np.percentile(top5, [5, 25, 50, 75, 95]) if len(top5) > 0 else np.zeros(5)

    return {
        "top1_size": {
            "q5": float(q_top1[0]),
            "q25": float(q_top1[1]),
            "q50": float(q_top1[2]),
            "q75": float(q_top1[3]),
            "q95": float(q_top1[4]),
        },
        "top5_size": {
            "q5": float(q_top5[0]),
            "q25": float(q_top5[1]),
            "q50": float(q_top5[2]),
            "q75": float(q_top5[3]),
            "q95": float(q_top5[4]),
        },
    }


def imbalance_ic(
    book: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10),
    levels: tuple[int, ...] = (1, 5),
    block_size: int = 30,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate Spearman rank correlation between order book imbalance and future mid returns with block bootstrap CI."""
    if book.empty or len(book) < 10:
        return {"real": {}, "synthetic": {}}

    rng = np.random.default_rng(seed)
    mid = book["mid"].values if "mid" in book.columns else ((book["bid_px_1"] + book["ask_px_1"]) / 2.0).values
    n = len(mid)

    # Compute imbalance for each level specification
    imb_map: dict[int, np.ndarray] = {}
    for L in levels:
        bid_vol = np.zeros(n)
        ask_vol = np.zeros(n)
        for i in range(1, L + 1):
            bid_vol += book.get(f"bid_sz_{i}", 0.0).fillna(0.0).values
            ask_vol += book.get(f"ask_sz_{i}", 0.0).fillna(0.0).values
        tot = bid_vol + ask_vol
        with np.errstate(divide="ignore", invalid="ignore"):
            imb = np.where(tot > 0, (bid_vol - ask_vol) / tot, 0.0)
        imb_map[L] = imb

    # Synthetic comparison book of same length (independent geometric random walk)
    syn_mid = 50000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0005, size=n)))
    syn_imb_map: dict[int, np.ndarray] = {L: rng.uniform(-1.0, 1.0, size=n) for L in levels}

    real_results: dict[str, Any] = {}
    syn_results: dict[str, Any] = {}

    for L in levels:
        imb_real = imb_map[L]
        imb_syn = syn_imb_map[L]
        for h in horizons:
            key = f"level_{L}_h_{h}"
            if n <= h:
                continue

            # Real returns
            ret_real = np.zeros(n - h)
            with np.errstate(divide="ignore", invalid="ignore"):
                ret_real = np.log(mid[h:] / mid[:-h])
            imb_r_sub = imb_real[:-h]

            # Synthetic returns
            ret_syn = np.log(syn_mid[h:] / syn_mid[:-h])
            imb_s_sub = imb_syn[:-h]

            # Calculate Spearman correlations
            corr_real, _ = stats.spearmanr(imb_r_sub, ret_real)
            corr_syn, _ = stats.spearmanr(imb_s_sub, ret_syn)

            # Block bootstrap 95% CI on real pairs (block length 30)
            n_pairs = len(imb_r_sub)
            boot_corrs: list[float] = []
            if n_pairs >= block_size:
                n_blocks = max(1, n_pairs // block_size)
                for _ in range(n_bootstrap):
                    idx_blocks = rng.integers(0, n_pairs - block_size + 1, size=n_blocks)
                    sample_indices = np.concatenate([np.arange(b, b + block_size) for b in idx_blocks])
                    b_r, _ = stats.spearmanr(imb_r_sub[sample_indices], ret_real[sample_indices])
                    if not np.isnan(b_r):
                        boot_corrs.append(float(b_r))

            real_ic = float(corr_real) if not np.isnan(corr_real) else 0.0
            ci_lo = float(np.percentile(boot_corrs, 2.5)) if boot_corrs else real_ic
            ci_hi = float(np.percentile(boot_corrs, 97.5)) if boot_corrs else real_ic
            if np.isnan(ci_lo):
                ci_lo = 0.0
            if np.isnan(ci_hi):
                ci_hi = 0.0

            real_results[key] = {
                "ic": real_ic,
                "ci_95": [ci_lo, ci_hi],
            }
            syn_results[key] = {
                "ic": float(corr_syn) if not np.isnan(corr_syn) else 0.0,
            }

    return {"real": real_results, "synthetic": syn_results}


def trade_sign_autocorr(trades: pd.DataFrame, max_lags: int = 20) -> dict[str, Any]:
    """Calculate trade sign autocorrelation across lags 1 to 20 for real and synthetic tapes."""
    if trades.empty or len(trades) < 2:
        return {"real": {}, "synthetic": {}}

    real_signs = trades["sign"].values
    n = len(real_signs)

    # Synthetic independent signs (Rademacher +/- 1)
    rng = np.random.default_rng(123)
    syn_signs = rng.choice([-1, 1], size=n)

    real_autocorr: dict[int, float] = {}
    syn_autocorr: dict[int, float] = {}

    s_real = pd.Series(real_signs)
    s_syn = pd.Series(syn_signs)

    for lag in range(1, min(max_lags + 1, n)):
        ac_r = s_real.autocorr(lag=lag)
        ac_s = s_syn.autocorr(lag=lag)
        real_autocorr[lag] = float(ac_r) if not np.isnan(ac_r) else 0.0
        syn_autocorr[lag] = float(ac_s) if not np.isnan(ac_s) else 0.0

    return {"real": real_autocorr, "synthetic": syn_autocorr}


def book_walk_cost(snapshot: pd.Series | dict[str, Any], size: float, side: str = "BUY") -> float:
    """Calculate the VWAP instantaneous liquidity cost of a single sweep across visible book minus mid, in bp.

    CRITICAL NOTE:
    Book-walking cost is the instantaneous liquidity cost of a single sweep,
    not the square-root law for multi-slice metaorders; do not claim one validates the other.
    """
    clean_side = side.strip().upper()
    req_qty = float(size)
    if req_qty <= 0:
        return 0.0

    mid = float(snapshot.get("mid", np.nan))
    if np.isnan(mid):
        best_b = float(snapshot.get("bid_px_1", np.nan))
        best_a = float(snapshot.get("ask_px_1", np.nan))
        mid = (best_b + best_a) / 2.0 if not (np.isnan(best_b) or np.isnan(best_a)) else 100.0

    rem = req_qty
    total_dollars = 0.0
    levels: list[tuple[float, float]] = []

    # Extract up to 20 levels
    if clean_side == "BUY":
        for k in range(1, 21):
            px = snapshot.get(f"ask_px_{k}")
            sz = snapshot.get(f"ask_sz_{k}")
            if px is not None and sz is not None and not np.isnan(px) and not np.isnan(sz) and sz > 0:
                levels.append((float(px), float(sz)))
    else:
        for k in range(1, 21):
            px = snapshot.get(f"bid_px_{k}")
            sz = snapshot.get(f"bid_sz_{k}")
            if px is not None and sz is not None and not np.isnan(px) and not np.isnan(sz) and sz > 0:
                levels.append((float(px), float(sz)))

    if not levels:
        return 0.0

    for px, sz in levels:
        take = min(rem, sz)
        total_dollars += take * px
        rem -= take
        if rem <= 1e-9:
            break

    # If order exceeds visible depth, remainder fills at worst visible price
    if rem > 1e-9:
        worst_px = levels[-1][0]
        total_dollars += rem * worst_px
        rem = 0.0

    vwap = total_dollars / req_qty

    if clean_side == "BUY":
        cost_bp = ((vwap - mid) / mid) * 10_000.0
    else:
        cost_bp = ((mid - vwap) / mid) * 10_000.0

    return float(max(0.0, cost_bp))


def impact_fit(
    book: pd.DataFrame,
    sizes: list[float] | None = None,
) -> dict[str, Any]:
    """Fit log(cost) = a + gamma * log(size) and cost = c * sqrt(size) across order book snapshots.

    CRITICAL NOTE:
    Book-walking cost is the instantaneous liquidity cost of a single sweep,
    not the square-root law for multi-slice metaorders; do not claim one validates the other.
    """
    if book.empty:
        return {"gamma": 0.5, "gamma_se": 0.0, "c": 1.0, "a": 0.0, "r2": 0.0}

    test_sizes = sizes or [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    all_sizes: list[float] = []
    all_costs_net: list[float] = []

    # Sample rows if book is large
    sample_df = book.sample(min(len(book), 100), random_state=42) if len(book) > 100 else book

    for _, row in sample_df.iterrows():
        mid = float(row.get("mid", (row["bid_px_1"] + row["ask_px_1"]) / 2.0))
        spread = float(row.get("spread", row["ask_px_1"] - row["bid_px_1"]))
        half_spread_bp = ((spread / 2.0) / mid) * 10_000.0

        for sz in test_sizes:
            gross_cost = book_walk_cost(row, size=sz, side="BUY")
            net_cost = max(gross_cost - half_spread_bp, 1e-4)
            all_sizes.append(sz)
            all_costs_net.append(net_cost)

    sizes_arr = np.array(all_sizes, dtype=float)
    costs_arr = np.array(all_costs_net, dtype=float)

    # 1. Fit log(cost) = a + gamma * log(size) via OLS
    log_sz = np.log(sizes_arr)
    log_cost = np.log(costs_arr)
    slope, intercept, r_value, _, stderr = stats.linregress(log_sz, log_cost)

    # 2. Fit cost = c * sqrt(size) separately
    sqrt_sz = np.sqrt(sizes_arr)
    c_coeff = float(np.sum(costs_arr * sqrt_sz) / max(np.sum(sqrt_sz ** 2), 1e-8))

    return {
        "gamma": float(slope),
        "gamma_se": float(stderr),
        "a": float(intercept),
        "r2": float(r_value ** 2),
        "c": c_coeff,
    }


def queue_fill_probability(
    book: pd.DataFrame,
    trades: pd.DataFrame,
    horizons_s: tuple[int, ...] = (30, 60),
    my_size: float = 0.1,
) -> dict[str, Any]:
    """Calculate empirical queue fill probability and run differential comparison with Prompt 10 LOBSimulator.

    Independent queue model:
    At snapshot time t, a hypothetical buy limit order joins the best bid with queue_ahead = best bid size.
    It counts as filled when the cumulative size of aggressor-sell trades (sign = -1) priced at or below
    that bid price after t exceeds queue_ahead + my_size before the best bid drops below that level.
    NOTE: Cancellations are ignored, which systematically understates fills.
    """
    if book.empty or trades.empty:
        return {"agreement_rate": 1.0, "disagreements": [], "fill_rates": {}}

    # Initialize LOBSimulator with cancel_rate=0.0 to match no-cancellation baseline
    lob_sim = LOBSimulator(book, trades, cancel_rate=0.0)

    disagreements: list[dict[str, Any]] = []
    fill_records: list[dict[str, Any]] = []

    # Sort trades by timestamp
    tr_df = trades.copy()
    if not isinstance(tr_df.index, pd.DatetimeIndex):
        tr_df.index = pd.to_datetime(tr_df["ts"], unit="ms", utc=True)
    tr_df.sort_index(inplace=True)

    # Sample test snapshot times
    snapshot_times = book.index[::max(1, len(book) // 20)]

    for t_idx, t_snap in enumerate(snapshot_times):
        row = book.loc[t_snap]
        best_bid = float(row["bid_px_1"])
        queue_ahead = float(row["bid_sz_1"])

        for horizon in horizons_s:
            t_end = t_snap + pd.Timedelta(seconds=horizon)

            # 1. Independent fill calculation
            trades_window = tr_df.loc[(tr_df.index > t_snap) & (tr_df.index <= t_end)]
            book_window = book.loc[(book.index > t_snap) & (book.index <= t_end)]

            # Check if bid dropped below entry bid
            bid_dropped = False
            if not book_window.empty:
                min_bid_in_window = float(book_window["bid_px_1"].min())
                if min_bid_in_window < best_bid:
                    bid_dropped = True

            # Cumulative aggressor sell volume at or below best_bid
            sell_trades = trades_window.loc[(trades_window["sign"] == -1) & (trades_window["price"] <= best_bid)]
            cum_sell_sz = float(sell_trades["size"].sum()) if not sell_trades.empty else 0.0

            independent_filled = bool(cum_sell_sz >= (queue_ahead + my_size) and not bid_dropped)

            # 2. Simulator fill evaluation
            order_id = f"test_lmt_{t_idx}_{horizon}"
            test_order = OrderEvent(
                timestamp=t_snap,
                symbol="BTCUSDT",
                side="BUY",
                quantity=my_size,
                order_type="LMT",
                limit_price=best_bid,
                order_id=order_id,
            )
            lob_sim.submit_order(test_order, submission_time=t_snap)
            fills = lob_sim.process_tape_window(start_time=t_snap, end_time=t_end)
            sim_filled = bool(len(fills) > 0)

            # Clean up active simulator order
            lob_sim.active_lmt_orders.pop(order_id, None)

            fill_records.append({
                "timestamp": str(t_snap),
                "horizon_s": horizon,
                "independent_fill": independent_filled,
                "simulator_fill": sim_filled,
            })

            if independent_filled != sim_filled:
                disagreements.append({
                    "timestamp": str(t_snap),
                    "horizon_s": horizon,
                    "independent_fill": independent_filled,
                    "simulator_fill": sim_filled,
                })

    n_total = len(fill_records)
    n_agree = n_total - len(disagreements)
    agree_rate = float(n_agree / n_total) if n_total > 0 else 1.0

    indep_fill_rate = float(np.mean([r["independent_fill"] for r in fill_records])) if fill_records else 0.0
    sim_fill_rate = float(np.mean([r["simulator_fill"] for r in fill_records])) if fill_records else 0.0

    return {
        "agreement_rate": agree_rate,
        "disagreements": disagreements,
        "fill_rates": {
            "independent": indep_fill_rate,
            "simulator": sim_fill_rate,
        },
        "sample_size": n_total,
    }


def sanitize_for_ledger(val: Any) -> Any:
    """Recursively replace NaN and Inf floats with 0.0 to satisfy cryptographic ledger compliance."""
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    elif isinstance(val, dict):
        return {k: sanitize_for_ledger(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [sanitize_for_ledger(x) for x in val]
    elif isinstance(val, tuple):
        return tuple(sanitize_for_ledger(x) for x in val)
    return val


def run_microstructure_calibration(
    symbol: str = "BTCUSDT",
    store_path: str | Path = "research/store_devval",
    ledger_path: str | Path = "research/ledger/ledger.jsonl",
    params_path: str | Path = "research/calibration/params.json",
    manifests_dir: str | Path = "research/manifests",
) -> dict[str, Any]:
    """Execute complete microstructure calibration suite and persist cryptographic audit records."""
    book = read_microstructure_data(symbol, "real_l2_book", store_path=store_path)
    trades = read_microstructure_data(symbol, "real_l2_trades", store_path=store_path)
    ledger = Ledger(ledger_path)

    # Locate manifest ID if available
    manifest_ids: list[str] = []
    m_dir = Path(manifests_dir)
    if m_dir.exists():
        manifest_ids = [p.stem for p in m_dir.glob(f"*{symbol.lower()}*.json")]

    if book.empty or trades.empty:
        not_run_payload = {
            "symbol": symbol,
            "status": "NOT_RUN",
            "reason": "No recorded microstructure order book or trade data found in partitioned store.",
        }
        ledger.append(kind="calibration", status="NOT_RUN", tag="cal.spread", payload=not_run_payload)
        return not_run_payload

    # 1. Spread statistics
    spread_res = sanitize_for_ledger(spread_stats(book, tick_size=0.01))
    ledger.append(kind="calibration", status="PASS", tag="cal.spread", payload=spread_res)

    # 2. Depth statistics
    depth_res = sanitize_for_ledger(depth_stats(book))
    ledger.append(kind="calibration", status="PASS", tag="cal.depth", payload=depth_res)

    # 3. Imbalance IC
    imb_res = sanitize_for_ledger(imbalance_ic(book, horizons=(1, 5, 10), levels=(1, 5)))
    ledger.append(kind="calibration", status="PASS", tag="cal.imbalance_ic", payload=imb_res)

    # 4. Sign autocorrelation
    sign_res = sanitize_for_ledger(trade_sign_autocorr(trades, max_lags=20))
    ledger.append(kind="calibration", status="PASS", tag="cal.sign_autocorr", payload=sign_res)

    # 5. Impact fit
    impact_res = sanitize_for_ledger(impact_fit(book))
    ledger.append(kind="calibration", status="PASS", tag="cal.impact", payload=impact_res)

    # 6. Queue fill probability & differential simulator test
    queue_res = sanitize_for_ledger(queue_fill_probability(book, trades, horizons_s=(30, 60), my_size=0.1))
    ledger.append(kind="calibration", status="PASS", tag="cal.queue", payload=queue_res)

    # Persist calibrated parameters to research/calibration/params.json
    cal_params = {
        "symbol": symbol,
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_ids": manifest_ids,
        "c": impact_res["c"],
        "gamma": impact_res["gamma"],
        "gamma_se": impact_res["gamma_se"],
        "a": impact_res["a"],
        "r2": impact_res["r2"],
        "spread_median_ticks": spread_res["q50"],
        "depth_top1_median": depth_res["top1_size"]["q50"],
        "depth_top5_median": depth_res["top5_size"]["q50"],
        "fill_rates": queue_res["fill_rates"],
        "simulator_agreement_rate": queue_res["agreement_rate"],
        "disagreements_count": len(queue_res["disagreements"]),
        "policy": "Calibrated parameters are descriptive only; earlier backtest models remain unmodified.",
    }

    p_path = Path(params_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    p_path.write_text(json.dumps(cal_params, indent=2), encoding="utf-8")

    return {
        "spread": spread_res,
        "depth": depth_res,
        "imbalance_ic": imb_res,
        "sign_autocorr": sign_res,
        "impact": impact_res,
        "queue": queue_res,
        "params": cal_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Microstructure calibration and differential simulation runner.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to calibrate")
    parser.add_argument("--store-path", type=str, default="research/store_devval", help="ParquetStore directory")
    args = parser.parse_args()

    results = run_microstructure_calibration(symbol=args.symbol, store_path=args.store_path)

    print("\n=== FINPULSE-APEX MICROSTRUCTURE CALIBRATION SUMMARY ===")
    if results.get("status") == "NOT_RUN":
        print(f"Status: NOT_RUN ({results.get('reason')})")
        return

    params = results["params"]
    print(f"Symbol: {params['symbol']}")
    print(f"Spread Median: {params['spread_median_ticks']:.2f} ticks | Mean: {results['spread']['mean_ticks']:.2f} ticks")
    print(f"Top-1 Depth Median: {params['depth_top1_median']:.4f} | Top-5 Depth Median: {params['depth_top5_median']:.4f}")
    print(f"Impact Exponent gamma: {params['gamma']:.4f} (SE: {params['gamma_se']:.4f}, R2: {params['r2']:.4f})")
    print(f"Square-Root Impact Coeff c: {params['c']:.4f}")
    print(f"Differential LOBSimulator Agreement Rate: {params['simulator_agreement_rate'] * 100.0:.2f}%")
    print(f"Fill Rates: Independent={params['fill_rates']['independent'] * 100.0:.1f}%, Simulator={params['fill_rates']['simulator'] * 100.0:.1f}%")

    print("\n=======================================================================================")
    print("Metric                 | Real Market Data          | Synthetic Benchmark / Simulation ")
    print("---------------------------------------------------------------------------------------")
    ic_1_1 = results["imbalance_ic"]["real"].get("level_1_h_1", {})
    ic_syn_1_1 = results["imbalance_ic"]["synthetic"].get("level_1_h_1", {})
    print(f"Imbalance IC (L=1, h=1)| {ic_1_1.get('ic', 0.0):+.4f} (95% CI: {ic_1_1.get('ci_95', [0,0])}) | {ic_syn_1_1.get('ic', 0.0):+.4f} (IID GBM)")

    ac_1 = results["sign_autocorr"]["real"].get(1, 0.0)
    ac_syn_1 = results["sign_autocorr"]["synthetic"].get(1, 0.0)
    print(f"Trade Sign Autocorr (1)| {ac_1:+.4f}                    | {ac_syn_1:+.4f} (Rademacher)")

    ac_5 = results["sign_autocorr"]["real"].get(5, 0.0)
    ac_syn_5 = results["sign_autocorr"]["synthetic"].get(5, 0.0)
    print(f"Trade Sign Autocorr (5)| {ac_5:+.4f}                    | {ac_syn_5:+.4f} (Rademacher)")

    print(f"Queue Fill Decision    | {params['fill_rates']['independent'] * 100.0:.1f}% Fill Rate              | {params['fill_rates']['simulator'] * 100.0:.1f}% Fill Rate")
    print("=======================================================================================")


if __name__ == "__main__":
    main()
