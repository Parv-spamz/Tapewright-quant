"""Strategy candidate selection, deflated Sharpe evaluation, validation, and PBO.

Enforces experimental discipline across candidate selection:
- Selection strictly follows objective pre-registered filters (>= 30 trades, MDD >= -40%)
- Deflated Sharpe Ratio (DSR) discounts selection bias across all dev trials
- Validation evaluation runs selected candidates once on pure validation split
- Protocol survival rule objectively labels candidate survival as boolean
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from backtesting.metrics.risk_measures.measures import sharpe
from backtesting.metrics.risk_measures.overfit import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from integrity.ledger import Ledger
from research.data_access import ResearchData, get_experiment_result_path, load_experiment_results
from research.experiment import ExperimentSpec, run_experiment, trial_count
from research.protocol import protocol_chain

logger = logging.getLogger(__name__)


def trial_matrix(
    protocol_id: str,
    ledger: Ledger | None = None,
    results_dir: Path | str = "research/results",
) -> pd.DataFrame:
    """Build dates x trials matrix of dev net returns from verified results files.

    Each results file is cryptographically checked against its recorded SHA-256 hash.
    """
    led = ledger or Ledger()
    chain = protocol_chain(protocol_id, led)
    chain_set = set(chain)

    records = led._read_all_records()
    trial_records = [
        r for r in records
        if r.get("kind") == "experiment"
        and r.get("status") == "PASS"
        and r.get("payload", {}).get("role") == "trial"
        and r.get("payload", {}).get("split") == "dev"
        and r.get("payload", {}).get("protocol_id") in chain_set
    ]

    series_map: dict[str, pd.Series] = {}
    r_dir = Path(results_dir)

    for r in trial_records:
        p = r["payload"]
        run_id = p["run_id"]
        res_path = Path(p.get("results_path", get_experiment_result_path(r_dir, run_id)))
        if not res_path.exists():
            continue

        # Hash check
        file_sha = hashlib.sha256(res_path.read_bytes()).hexdigest()
        expected_sha = p.get("results_sha256")
        if expected_sha and file_sha != expected_sha:
            raise ValueError(f"Hash mismatch for results file {res_path}: expected {expected_sha}, got {file_sha}")

        df = load_experiment_results(res_path)
        if "timestamp" in df.columns and "net_return" in df.columns:
            s = pd.Series(df["net_return"].values, index=pd.to_datetime(df["timestamp"]))
            series_map[run_id] = s

    if not series_map:
        return pd.DataFrame()

    mat = pd.DataFrame(series_map).sort_index().fillna(0.0)
    return mat


def dev_pbo(protocol_id: str, ledger: Ledger | None = None) -> float:
    """Compute Probability of Backtest Overfitting (PBO) across all dev trials and log to ledger."""
    led = ledger or Ledger()
    mat = trial_matrix(protocol_id, led)
    if mat.empty or mat.shape[1] < 2:
        pbo_val = 0.0
    else:
        pbo_val, _ = probability_of_backtest_overfitting(mat, n_blocks=min(16, max(2, len(mat) // 10)))

    payload = {
        "protocol_id": protocol_id,
        "pbo": float(pbo_val),
        "n_trials": mat.shape[1],
        "n_bars": len(mat),
    }
    led.append(
        kind="experiment",
        status="PASS",
        tag="dev.pbo",
        payload=payload,
    )
    return float(pbo_val)


def compute_per_period_sharpe(rets: np.ndarray | pd.Series) -> float:
    """Calculate sample per-period Sharpe ratio."""
    arr = np.asarray(rets, dtype=float)
    if len(arr) <= 1:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(arr) / std)


def dev_dsr(
    run_id: str,
    ledger: Ledger | None = None,
    results_dir: Path | str = "research/results",
) -> float:
    """Calculate Deflated Sharpe Ratio for a dev experiment run.

    Uses per-period Sharpe of the run, variance of per-period Sharpe across ALL dev
    trials in the protocol chain, and total trial count.
    """
    led = ledger or Ledger()
    records = led._read_all_records()

    target_record = next(
        (r for r in records if r.get("kind") == "experiment" and r.get("payload", {}).get("run_id") == run_id),
        None,
    )
    if not target_record:
        raise ValueError(f"Run ID {run_id} not found in ledger.")

    target_p = target_record["payload"]
    protocol_id = target_p["protocol_id"]
    chain = protocol_chain(protocol_id, led)
    chain_set = set(chain)

    n_trials = trial_count(protocol_id, led)
    if n_trials <= 0:
        n_trials = 1

    # Load returns for target run
    r_dir = Path(results_dir)
    target_path = Path(target_p.get("results_path", get_experiment_result_path(r_dir, run_id)))
    if not target_path.exists():
        raise FileNotFoundError(f"Results file not found: {target_path}")

    target_df = load_experiment_results(target_path)
    rets = target_df["net_return"].values
    T = len(rets)
    sr_hat = compute_per_period_sharpe(rets)
    skew = float(stats.skew(rets)) if T > 2 else 0.0
    kurt = float(stats.kurtosis(rets, fisher=False)) if T > 3 else 3.0

    # Collect per-period Sharpes across ALL dev trials in protocol chain
    trial_records = [
        r for r in records
        if r.get("kind") == "experiment"
        and r.get("status") == "PASS"
        and r.get("payload", {}).get("role") == "trial"
        and r.get("payload", {}).get("split") == "dev"
        and r.get("payload", {}).get("protocol_id") in chain_set
    ]

    all_sharpes: list[float] = []
    for r in trial_records:
        rp = r["payload"]
        p_file = Path(rp.get("results_path", get_experiment_result_path(r_dir, rp["run_id"])))
        if p_file.exists():
            df_t = load_experiment_results(p_file)
            if "net_return" in df_t:
                all_sharpes.append(compute_per_period_sharpe(df_t["net_return"].values))

    var_sr = float(np.var(all_sharpes, ddof=1)) if len(all_sharpes) > 1 else 0.0

    dsr = deflated_sharpe_ratio(
        sr_hat=sr_hat,
        T=T,
        skew=skew,
        kurt=kurt,
        n_trials=n_trials,
        var_sr=var_sr,
        annualized=False,
    )
    return float(dsr)


def select_candidates(
    protocol_id: str,
    ledger: Ledger | None = None,
) -> list[dict[str, Any]]:
    """Select the best candidate trial for each family on dev split under strict rules.

    Filters:
    - trade_count >= 30
    - max_drawdown >= -0.40 (magnitude <= 40%)
    - Highest dev net Sharpe
    Deterministic with no manual overrides. Appends 'experiment' record with role 'selected'.
    """
    led = ledger or Ledger()
    chain = protocol_chain(protocol_id, led)
    chain_set = set(chain)

    records = led._read_all_records()
    trial_records = [
        r for r in records
        if r.get("kind") == "experiment"
        and r.get("status") == "PASS"
        and r.get("payload", {}).get("role") == "trial"
        and r.get("payload", {}).get("split") == "dev"
        and r.get("payload", {}).get("protocol_id") in chain_set
    ]

    # Group eligible candidates by family
    candidates_by_family: dict[str, list[dict[str, Any]]] = {}
    for r in trial_records:
        p = r["payload"]
        fam = p.get("spec", {}).get("family")
        metrics = p.get("metrics", {})
        tc = metrics.get("trade_count", 0)
        mdd = metrics.get("max_drawdown", -1.0)

        # Enforce >= 30 trades and max drawdown >= -0.40
        if tc >= 30 and mdd >= -0.40:
            candidates_by_family.setdefault(fam, []).append(p)

    selected_results: list[dict[str, Any]] = []

    for fam, cand_list in sorted(candidates_by_family.items()):
        # Deterministic sorting: descending Sharpe, descending MDD, ascending spec_hash
        cand_list.sort(
            key=lambda p: (
                -float(p["metrics"].get("sharpe", -999.0)),
                -float(p["metrics"].get("max_drawdown", -1.0)),
                p.get("spec_hash", ""),
            )
        )
        best = cand_list[0]
        spec_hash = best["spec_hash"]

        # Check if already recorded as selected
        existing_selected = next(
            (
                r for r in records
                if r.get("kind") == "experiment"
                and r.get("payload", {}).get("role") == "selected"
                and r.get("payload", {}).get("protocol_id") == protocol_id
                and r.get("payload", {}).get("spec_hash") == spec_hash
            ),
            None,
        )

        if existing_selected:
            selected_results.append(existing_selected["payload"])
            continue

        payload = {
            "protocol_id": protocol_id,
            "role": "selected",
            "family": fam,
            "spec_hash": spec_hash,
            "run_id": best["run_id"],
            "spec": best["spec"],
            "justification": {
                "dev_net_sharpe": best["metrics"].get("sharpe"),
                "trade_count": best["metrics"].get("trade_count"),
                "max_drawdown": best["metrics"].get("max_drawdown"),
                "annualized_return": best["metrics"].get("annualized_return"),
                "vol": best["metrics"].get("vol"),
            },
        }

        led.append(
            kind="experiment",
            status="PASS",
            tag=f"selected.{fam}",
            payload=payload,
        )
        selected_results.append(payload)

    return selected_results


def block_bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray,
    block_size: int = 10,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate block-bootstrap 95% confidence interval for annualized Sharpe ratio."""
    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    if n <= 1:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(1, n - block_size + 1)
    boot_sharpes: list[float] = []

    for _ in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        sampled = np.concatenate([arr[s : s + block_size] for s in starts])[:n]
        std = np.std(sampled, ddof=1)
        sr = (np.mean(sampled) / std) * np.sqrt(252) if std > 1e-12 else 0.0
        boot_sharpes.append(sr)

    lo = float(np.percentile(boot_sharpes, 100 * (alpha / 2.0)))
    hi = float(np.percentile(boot_sharpes, 100 * (1.0 - alpha / 2.0)))
    return lo, hi


def evaluate_on_validation(
    protocol_id: str,
    ledger: Ledger | None = None,
    data: ResearchData | None = None,
) -> list[dict[str, Any]]:
    """Evaluate each selected candidate on pure validation split exactly once.

    Computes:
    - Val net Sharpe
    - PSR against 0
    - PSR against best baseline
    - Block-bootstrap Sharpe CI
    - Protocol survival rule (dev_dsr >= 0.90, val_net_sharpe > 0, val_psr_zero >= 0.90)

    Appends ledger 'experiment' record tagged 'val.<family>' with boolean 'survives'.
    Refuses to evaluate specs that were not selected.
    """
    led = ledger or Ledger()
    rd = data or ResearchData(protocol_id=protocol_id, ledger=led)
    chain = protocol_chain(protocol_id, led)
    chain_set = set(chain)

    records = led._read_all_records()

    # Find protocol record to read survival rule
    proto_rec = next(
        (r for r in records if r.get("kind") == "protocol" and r.get("payload", {}).get("protocol_id") == protocol_id),
        None,
    )
    if not proto_rec:
        raise ValueError(f"Protocol {protocol_id} record not found in ledger")

    survival_rule = proto_rec["payload"]["protocol"].get("survival_rule", {})
    dev_dsr_min = float(survival_rule.get("dev_dsr_min", 0.90))
    val_sharpe_min = float(survival_rule.get("val_net_sharpe_min", 0.0))
    val_psr_min = float(survival_rule.get("val_psr_against_zero_min", 0.90))

    # Retrieve selected records
    selected_records = [
        r for r in records
        if r.get("kind") == "experiment"
        and r.get("payload", {}).get("role") == "selected"
        and r.get("payload", {}).get("protocol_id") in chain_set
    ]

    if not selected_records:
        logger.warning("No candidate selected for validation evaluation.")
        return []

    # Run / retrieve baselines on val to find best benchmark
    baseline_specs = [
        ("buy_and_hold_spy", {}, "etfs"),
        ("equal_weight_all_etfs", {}, "etfs"),
        ("sixty_forty_spy_ief", {}, "etfs"),
    ]
    baseline_sharpes: list[float] = []
    for b_fam, b_p, u_n in baseline_specs:
        b_spec = ExperimentSpec(
            family=b_fam,
            params=b_p,
            universe_name=u_n,
            split="val",
            role="baseline",
        )
        try:
            b_res = run_experiment(b_spec, protocol_id, led, rd)
            baseline_sharpes.append(float(b_res["metrics"].get("sharpe", 0.0)))
        except Exception as e:
            logger.warning("Baseline on val %s failed: %s", b_fam, e)

    best_baseline_sharpe = max(baseline_sharpes) if baseline_sharpes else 0.0

    eval_results: list[dict[str, Any]] = []

    for sel in selected_records:
        sp = sel["payload"]
        fam = sp["family"]
        spec_hash = sp["spec_hash"]
        saved_spec_dict = sp["spec"]

        # Check idempotency: if already evaluated on val with tag val.<family>
        existing_eval = next(
            (
                r for r in records
                if r.get("kind") == "experiment"
                and r.get("tag") == f"val.{fam}"
                and r.get("payload", {}).get("protocol_id") == protocol_id
                and r.get("payload", {}).get("spec_hash") == spec_hash
            ),
            None,
        )
        if existing_eval:
            eval_results.append(existing_eval["payload"])
            continue

        # Build validation spec
        val_spec = ExperimentSpec(
            family=fam,
            params=saved_spec_dict.get("params", {}),
            universe_name=saved_spec_dict.get("universe_name", "etfs"),
            split="val",
            role="selected_eval",
            execution=saved_spec_dict.get("execution", "realistic"),
            sizing=saved_spec_dict.get("sizing", "half_kelly"),
            flags=saved_spec_dict.get("flags", {}),
            seeds=saved_spec_dict.get("seeds", [42]),
        )

        val_res = run_experiment(val_spec, protocol_id, led, rd)
        val_path = Path(val_res["results_path"])
        val_df = load_experiment_results(val_path)
        val_rets = val_df["net_return"].values
        T = len(val_rets)

        val_net_sharpe = float(val_res["metrics"].get("sharpe", 0.0))
        skew = float(stats.skew(val_rets)) if T > 2 else 0.0
        kurt = float(stats.kurtosis(val_rets, fisher=False)) if T > 3 else 3.0

        psr_zero = probabilistic_sharpe_ratio(
            sr_hat=val_net_sharpe,
            sr_benchmark=0.0,
            T=T,
            skew=skew,
            kurt=kurt,
            annualized=True,
        )

        psr_baseline = probabilistic_sharpe_ratio(
            sr_hat=val_net_sharpe,
            sr_benchmark=best_baseline_sharpe,
            T=T,
            skew=skew,
            kurt=kurt,
            annualized=True,
        )

        ci_lo, ci_hi = block_bootstrap_sharpe_ci(val_rets)
        candidate_dev_dsr = dev_dsr(sp["run_id"], led)

        # Apply survival rule objectively
        survives = bool(
            candidate_dev_dsr >= dev_dsr_min
            and val_net_sharpe > val_sharpe_min
            and psr_zero >= val_psr_min
        )

        eval_payload = {
            "protocol_id": protocol_id,
            "role": "selected_eval",
            "family": fam,
            "spec_hash": spec_hash,
            "val_run_id": val_res["run_id"],
            "val_net_sharpe": val_net_sharpe,
            "val_psr_zero": psr_zero,
            "val_psr_best_baseline": psr_baseline,
            "bootstrap_ci": [ci_lo, ci_hi],
            "dev_dsr": candidate_dev_dsr,
            "best_baseline_sharpe": best_baseline_sharpe,
            "survives": survives,
            "metrics": val_res["metrics"],
        }

        led.append(
            kind="experiment",
            status="PASS",
            tag=f"val.{fam}",
            payload=eval_payload,
        )
        eval_results.append(eval_payload)

    return eval_results
