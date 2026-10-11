"""Offline reproduction harness re-running recorded experiments and verifying metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.engine import BacktestEngine
from backtesting.engine.event_loop.execution import SimulatedExecutionHandler
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.metrics.drawdown_analysis.drawdown import max_drawdown
from backtesting.metrics.risk_measures.measures import (
    annualized_return,
    annualized_vol,
    calmar,
    sharpe,
    sortino,
)
from config.trading.risk_limits import RiskLimits
from integrity.ledger import Ledger
from research.costs import build_fill_model
from research.data_access import ResearchData
from research.experiment import ExperimentSpec
from research.rebuild import rebuild_all_stores
from research.strategies_registry import build_strategy

logger = logging.getLogger("research.reproduce")

TOLERANCE = 1e-9

# Canonical 3 recorded experiments chosen across roles: baseline, trial, selected_eval
TARGET_SPECS = [
    {"role": "baseline", "family": "buy_and_hold_spy", "target_seq": 24},
    {"role": "trial", "family": "pairs", "target_seq": 28},
    {"role": "selected_eval", "family": "mean_reversion", "target_seq": 185},
]


def run_experiment_isolated(
    spec: ExperimentSpec,
    protocol_payload: dict[str, Any],
    data: ResearchData,
) -> dict[str, float]:
    """Execute backtest simulation in isolation without ledger writes or caching."""
    cost_cfg = protocol_payload.get("cost_model", {})
    universe = protocol_payload.get("universe", [])
    is_crypto = "BTCUSDT" in universe or "ETHUSDT" in universe

    bars_dict: dict[str, pd.DataFrame] = {}
    for sym in universe:
        bars_dict[sym] = data.load_partition(sym, spec.split)

    dev_panel = None
    if spec.family == "pairs" and spec.split == "dev":
        dev_panel = pd.DataFrame(
            {sym: bars_dict[sym]["close"] for sym in universe if "close" in bars_dict[sym]}
        ).dropna()

    strategy = build_strategy(
        family=spec.family,
        params=spec.params,
        universe=universe,
        dev_panel=dev_panel,
    )

    fill_model = build_fill_model(
        cost_model=cost_cfg,
        is_crypto=is_crypto,
        cost_multiplier=1.0,
    )

    data_handler = HistoricBarDataHandler(bars_dict)
    portfolio = Portfolio(initial_cash=100_000.0, risk_limits=RiskLimits(max_position_pct=1.0))
    exec_handler = SimulatedExecutionHandler(data_handler, fill_model=fill_model)

    engine = BacktestEngine(data_handler, strategy, portfolio, exec_handler)
    res = engine.run()

    eq_curve = res.equity_curve
    rets = eq_curve["equity"].pct_change().dropna() if len(eq_curve) > 1 else pd.Series([0.0])

    ann_ret = float(annualized_return(rets)) if len(rets) > 0 else 0.0
    vol = float(annualized_vol(rets)) if len(rets) > 0 else 0.0
    shrp = float(sharpe(rets)) if len(rets) > 0 else 0.0
    srt = float(sortino(rets)) if len(rets) > 0 else 0.0
    mdd = float(max_drawdown(eq_curve["equity"])) if len(eq_curve) > 0 else 0.0
    clmr = float(calmar(rets, max_dd=mdd)) if len(rets) > 0 else 0.0

    avg_equity = float(eq_curve["equity"].mean()) if len(eq_curve) > 0 else 100_000.0
    total_traded_value = sum(abs(float(f.quantity) * float(f.fill_price)) for f in res.fills)
    turnover = float(total_traded_value / avg_equity) if avg_equity > 0 else 0.0

    total_costs = sum(float(f.commission) + float(f.slippage) for f in res.fills)
    cost_bp = float((total_costs / avg_equity) * 10_000.0) if avg_equity > 0 else 0.0
    trade_count_val = len(res.fills)
    final_eq = float(eq_curve["equity"].iloc[-1]) if len(eq_curve) > 0 else 100_000.0

    return {
        "annualized_return": ann_ret,
        "vol": vol,
        "sharpe": shrp,
        "sortino": srt,
        "max_drawdown": mdd,
        "calmar": clmr,
        "turnover": turnover,
        "total_cost_bp_equity": cost_bp,
        "trade_count": float(trade_count_val),
        "final_equity": final_eq,
    }


def reproduce_all(
    ledger_path: Path | str = "research/ledger/ledger.jsonl",
    manifest_dir: Path | str = "research/manifests",
    archive_dir: Path | str = "research/raw_archive",
    store_devval_path: Path | str = "research/store_devval",
    vault_path: Path | str = "research/vault",
) -> tuple[bool, list[dict[str, Any]]]:
    """Rebuild stores from archive and re-run recorded experiments comparing metrics within 1e-9."""
    # 1. Rebuild stores offline from raw archive
    print("=== STEP 1: Rebuilding stores offline from raw archive ===")
    rebuild_res = rebuild_all_stores(
        manifest_dir=manifest_dir,
        archive_dir=archive_dir,
        store_devval_path=store_devval_path,
        vault_path=vault_path,
        write_stores=True,
    )
    print(f"Rebuilt {rebuild_res['total_rebuilt']} datasets from archive.\n")

    # 2. Load protocol configuration and recorded experiments from ledger
    ledger = Ledger(ledger_path)
    records = ledger._read_all_records()

    proto_rec = next(
        (r for r in records if r.get("kind") == "protocol" and r.get("status") == "PASS"),
        None,
    )
    if not proto_rec:
        raise ValueError("No active protocol record found in ledger")
    proto_payload = proto_rec["payload"]["protocol"]

    # ResearchData with ledger=None to prevent audit appends during reproduction
    data = ResearchData(
        store_path=store_devval_path,
        ledger=None,
        ledger_path="",
    )

    print("=== STEP 2: Re-running 3 recorded experiments (baseline, trial, selected_eval) ===")
    all_matched = True
    experiment_reports: list[dict[str, Any]] = []

    for target in TARGET_SPECS:
        role = target["role"]
        family = target["family"]
        target_seq = target["target_seq"]

        # Locate recorded experiment in ledger
        rec = next(
            (
                r for r in records
                if r.get("seq") == target_seq
                and r.get("kind") == "experiment"
                and r.get("status") == "PASS"
            ),
            None,
        )
        if not rec:
            # Fallback search by role and family
            rec = next(
                (
                    r for r in records
                    if r.get("kind") == "experiment"
                    and r.get("status") == "PASS"
                    and r.get("payload", {}).get("role") == role
                    and r.get("payload", {}).get("spec", {}).get("family") == family
                    and r.get("payload", {}).get("metrics") is not None
                ),
                None,
            )
        if not rec:
            print(f"ERROR: Could not find recorded experiment for role={role}, family={family}")
            all_matched = False
            continue

        seq = rec["seq"]
        p = rec["payload"]
        saved_metrics = p["metrics"]
        s_dict = p["spec"]

        spec = ExperimentSpec(
            family=s_dict["family"],
            params=s_dict.get("params", {}),
            universe_name=s_dict.get("universe_name", "etfs"),
            split=s_dict["split"],
            role=s_dict["role"],
            execution=s_dict.get("execution", "realistic"),
            sizing=s_dict.get("sizing", "half_kelly"),
            flags=s_dict.get("flags", {}),
            seeds=s_dict.get("seeds", [42]),
        )

        print(f"\n--- Experiment: {role.upper()} ({family}, seq={seq}, split={spec.split}) ---")
        reproduced_metrics = run_experiment_isolated(
            spec=spec,
            protocol_payload=proto_payload,
            data=data,
        )

        metric_diffs: dict[str, float] = {}
        experiment_ok = True

        for m_name, saved_val in saved_metrics.items():
            repro_val = reproduced_metrics.get(m_name, 0.0)
            diff = abs(float(saved_val) - float(repro_val))
            metric_diffs[m_name] = diff
            status_str = "MATCH" if diff < TOLERANCE else "MISMATCH"
            if diff >= TOLERANCE:
                experiment_ok = False
                all_matched = False

            print(
                f"  {m_name:<22}: saved={saved_val:<14.6f} "
                f"reproduced={repro_val:<14.6f} diff={diff:<11.2e} -> {status_str}"
            )

        experiment_reports.append({
            "seq": seq,
            "role": role,
            "family": family,
            "ok": experiment_ok,
            "diffs": metric_diffs,
        })

    print("\n" + "=" * 80)
    if all_matched:
        print("REPRODUCTION RESULT: ALL EXPERIMENTS MATCH RECORDED METRICS WITHIN 1e-9")
    else:
        print("REPRODUCTION RESULT: MISMATCH DETECTED")
    print("=" * 80)

    return all_matched, experiment_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run recorded experiments and verify reproduction to 1e-9")
    parser.add_argument("--ledger", default="research/ledger/ledger.jsonl", help="Ledger path")
    parser.add_argument("--manifest-dir", default="research/manifests", help="Manifest directory")
    parser.add_argument("--archive-dir", default="research/raw_archive", help="Raw archive directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    success, _ = reproduce_all(
        ledger_path=args.ledger,
        manifest_dir=args.manifest_dir,
        archive_dir=args.archive_dir,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
