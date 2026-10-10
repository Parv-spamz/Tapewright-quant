"""Experiment execution harness and honest trial counter.

Enforces experimental discipline across strategy search:
- Immutable ExperimentSpec with canonical hash excluding role
- Gated data access strictly through ResearchData
- Automatic idempotency and duplicate deduplication for trial counting
- Cryptographic recording of execution results and metrics into the provenance ledger
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import uuid

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
from integrity.ledger import Ledger, canonical_json
from research.costs import build_fill_model
from research.data_access import ResearchData, get_experiment_result_path, save_experiment_results
from research.protocol import protocol_chain
from research.strategies_registry import build_strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable specification defining a strategy experiment."""

    family: str
    params: dict[str, Any]
    universe_name: str
    split: str  # in {"dev", "val"}
    role: str   # in {"trial", "baseline", "selected_eval", "ablation", "sensitivity"}
    execution: str = "realistic"
    sizing: str = "half_kelly"
    flags: dict[str, Any] = field(default_factory=lambda: {"vol_filter": False, "causal_gate": False, "meta_label": False})
    seeds: list[int] = field(default_factory=lambda: [42])

    def __post_init__(self) -> None:
        if self.split not in {"dev", "val"}:
            raise ValueError(f"split must be in {{'dev', 'val'}}, got {self.split}")
        valid_roles = {"trial", "baseline", "selected_eval", "ablation", "sensitivity"}
        if self.role not in valid_roles:
            raise ValueError(f"role must be in {valid_roles}, got {self.role}")

    @property
    def spec_hash(self) -> str:
        """Calculate deterministic SHA-256 digest of everything except role."""
        d = asdict(self)
        d.pop("role", None)
        return hashlib.sha256(canonical_json(d).encode("utf-8")).hexdigest()


def compute_code_hash() -> str:
    """Compute combined SHA-256 digest of experiment execution source modules."""
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for fname in ["experiment.py", "strategies_registry.py", "costs.py"]:
        p = root / fname
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def trial_count(protocol_id: str, ledger: Ledger) -> int:
    """Count distinct spec_hash with role 'trial' and split 'dev' across protocol chain."""
    chain = protocol_chain(protocol_id, ledger)
    chain_set = set(chain)

    records = ledger._read_all_records()
    distinct_specs: set[str] = set()

    for r in records:
        if r.get("kind") != "experiment":
            continue
        payload = r.get("payload", {})
        p_id = payload.get("protocol_id")
        if p_id not in chain_set:
            continue
        if payload.get("role") == "trial" and payload.get("split") == "dev":
            spec_h = payload.get("spec_hash")
            if spec_h:
                distinct_specs.add(spec_h)

    return len(distinct_specs)


def run_experiment(
    spec: ExperimentSpec,
    protocol_id: str,
    ledger: Ledger,
    data: ResearchData,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Execute a single backtest experiment against partitioned research data.

    Enforces:
    - Registered protocol requirement
    - Duplicate detection for 'trial' role (returns stored record without re-running)
    - Full metric calculation and recording in tamper-evident ledger
    - Net daily returns saved to research/results/<run_id>
    """
    # 1. Require registered protocol
    try:
        chain = protocol_chain(protocol_id, ledger)
    except Exception as e:
        raise ValueError(f"Protocol ID {protocol_id} is not registered in the active ledger: {e}") from e
    if not chain or chain[-1] != protocol_id:
        raise ValueError(f"Protocol ID {protocol_id} is not registered in the active ledger.")

    # Fetch protocol configuration from ledger
    proto_record = next(
        (r for r in ledger._read_all_records() if r.get("kind") == "protocol" and r.get("payload", {}).get("protocol_id") == protocol_id),
        None,
    )
    if not proto_record:
        raise ValueError(f"Protocol record not found for {protocol_id}")
    proto_payload = proto_record["payload"]["protocol"]
    cost_cfg = proto_payload.get("cost_model", {})
    universe = proto_payload.get("universe", [])
    dataset_ids = proto_payload.get("dataset_ids", [])

    # 2. Check for duplicate 'trial' run
    if spec.role == "trial":
        for r in ledger._read_all_records():
            if r.get("kind") == "experiment" and r.get("status") == "PASS":
                p = r.get("payload", {})
                if (
                    p.get("protocol_id") == protocol_id
                    and p.get("role") == "trial"
                    and p.get("split") == spec.split
                    and p.get("spec_hash") == spec.spec_hash
                ):
                    logger.info("Found existing trial record for %s; returning stored result.", spec.spec_hash[:12])
                    return p

    # 3. Load partition data through ResearchData
    bars_dict: dict[str, pd.DataFrame] = {}
    is_crypto = "BTCUSDT" in universe or "ETHUSDT" in universe

    try:
        # Load data for each asset in universe
        for sym in universe:
            bars_dict[sym] = data.load_partition(sym, spec.split)

        dev_panel = None
        if spec.family == "pairs" and spec.split == "dev":
            dev_panel = pd.DataFrame({sym: bars_dict[sym]["close"] for sym in universe if "close" in bars_dict[sym]}).dropna()

        # Build strategy
        strategy = build_strategy(
            family=spec.family,
            params=spec.params,
            universe=universe,
            dev_panel=dev_panel,
        )

        # Build fill model
        fill_model = build_fill_model(
            cost_model=cost_cfg,
            is_crypto=is_crypto,
            cost_multiplier=cost_multiplier,
        )

        # 4. Execute BacktestEngine
        data_handler = HistoricBarDataHandler(bars_dict)
        portfolio = Portfolio(initial_cash=100_000.0, risk_limits=RiskLimits(max_position_pct=1.0))
        exec_handler = SimulatedExecutionHandler(data_handler, fill_model=fill_model)

        engine = BacktestEngine(data_handler, strategy, portfolio, exec_handler)
        res = engine.run()

        # 5. Extract equity curve and compute metrics
        eq_curve = res.equity_curve
        if len(eq_curve) > 1:
            rets = eq_curve["equity"].pct_change().dropna()
        else:
            rets = pd.Series([0.0])

        ann_ret = float(annualized_return(rets)) if len(rets) > 0 else 0.0
        vol = float(annualized_vol(rets)) if len(rets) > 0 else 0.0
        shrp = float(sharpe(rets)) if len(rets) > 0 else 0.0
        srt = float(sortino(rets)) if len(rets) > 0 else 0.0
        mdd = float(max_drawdown(eq_curve["equity"])) if len(eq_curve) > 0 else 0.0
        clmr = float(calmar(rets, max_dd=mdd)) if len(rets) > 0 else 0.0

        # Turnover and costs
        avg_equity = float(eq_curve["equity"].mean()) if len(eq_curve) > 0 else 100_000.0
        total_traded_value = sum(abs(float(f.quantity) * float(f.fill_price)) for f in res.fills)
        turnover = float(total_traded_value / avg_equity) if avg_equity > 0 else 0.0

        total_costs = sum(float(f.commission) + float(f.slippage) for f in res.fills)
        cost_bp = float((total_costs / avg_equity) * 10_000.0) if avg_equity > 0 else 0.0
        trade_count_val = len(res.fills)

        metrics = {
            "annualized_return": ann_ret,
            "vol": vol,
            "sharpe": shrp,
            "sortino": srt,
            "max_drawdown": mdd,
            "calmar": clmr,
            "turnover": turnover,
            "total_cost_bp_equity": cost_bp,
            "trade_count": trade_count_val,
            "final_equity": float(eq_curve["equity"].iloc[-1]) if len(eq_curve) > 0 else 100_000.0,
        }

        # 6. Save results
        run_id = f"{spec.family}_{spec.spec_hash[:12]}_{spec.split}_{uuid.uuid4().hex[:8]}"
        results_dir = Path("research/results")
        results_path = get_experiment_result_path(results_dir, run_id)

        results_df = pd.DataFrame({
            "timestamp": eq_curve.index if isinstance(eq_curve.index, pd.DatetimeIndex) else pd.date_range("2020-01-01", periods=len(eq_curve)),
            "net_return": rets.reindex(eq_curve.index, fill_value=0.0).values,
            "equity": eq_curve["equity"].values,
        })
        save_experiment_results(results_df, results_path)

        res_bytes = results_path.read_bytes()
        results_sha = hashlib.sha256(res_bytes).hexdigest()

        # 7. Record in ledger
        payload = {
            "protocol_id": protocol_id,
            "run_id": run_id,
            "spec": asdict(spec),
            "spec_hash": spec.spec_hash,
            "split": spec.split,
            "role": spec.role,
            "metrics": metrics,
            "results_sha256": results_sha,
            "results_path": str(results_path),
            "dataset_ids": dataset_ids,
            "code_hash": compute_code_hash(),
            "seeds": spec.seeds,
            "status": "PASS",
        }
        ledger.append(kind="experiment", status="PASS", tag=f"experiment.{spec.family}", payload=payload)
        return payload

    except Exception as e:
        logger.warning("Experiment failed for spec %s: %s", spec.spec_hash[:12], e)
        fail_payload = {
            "protocol_id": protocol_id,
            "spec": asdict(spec),
            "spec_hash": spec.spec_hash,
            "split": spec.split,
            "role": spec.role,
            "status": "FAIL",
            "error": str(e),
        }
        try:
            ledger.append(kind="experiment", status="FAIL", tag=f"experiment.{spec.family}", payload=fail_payload)
        except Exception as err:
            logger.warning("Failed to record experiment failure to ledger: %s", err)
        raise
