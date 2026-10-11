"""Ablation analysis and component decomposition harness.

Decomposes the highest-performing strategy candidate across six pipeline pillars:
- V0: Signal only (fixed fraction, zero cost)
- V1: Add realistic costs
- V2: Add volatility filter
- V3: Add Half-Kelly plus CVaR sizing
- V4: Add causal gate
- V5: Add meta-labeling (NOT_RUN if family lacks side signal)
Also evaluates Leave-One-Out (LOO) variants and paired block-bootstrap CIs.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtesting.metrics.risk_measures.measures import sharpe
from integrity.ledger import Ledger
from research.data_access import ResearchData, load_experiment_results
from research.experiment import ExperimentSpec, run_experiment
from research.protocol import protocol_chain
from research.selection import dev_dsr

logger = logging.getLogger(__name__)


def paired_bootstrap_sharpe_diff_ci(
    returns_v_new: np.ndarray | pd.Series,
    returns_v_old: np.ndarray | pd.Series,
    block_size: int = 10,
    n_boot: int = 2000,
    seed: int = 1,
    alpha: float = 0.05,
) -> tuple[float, float, str]:
    """Calculate paired block-bootstrap confidence interval of Sharpe ratio difference.

    Synchronously samples identical time blocks from both return series to preserve
    cross-sectional dependencies across components.
    """
    r_new = np.asarray(returns_v_new, dtype=float)
    r_old = np.asarray(returns_v_old, dtype=float)
    n = min(len(r_new), len(r_old))
    if n <= 1:
        return 0.0, 0.0, "no demonstrated benefit"

    r_new = r_new[:n]
    r_old = r_old[:n]

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(1, n - block_size + 1)

    diffs: list[float] = []

    for _ in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(n, s + block_size)) for s in starts])[:n]

        sample_new = r_new[idx]
        sample_old = r_old[idx]

        std_new = np.std(sample_new, ddof=1)
        std_old = np.std(sample_old, ddof=1)

        sr_new = (np.mean(sample_new) / std_new) * np.sqrt(252) if std_new > 1e-12 else 0.0
        sr_old = (np.mean(sample_old) / std_old) * np.sqrt(252) if std_old > 1e-12 else 0.0

        diffs.append(sr_new - sr_old)

    lo = float(np.percentile(diffs, 100 * (alpha / 2.0)))
    hi = float(np.percentile(diffs, 100 * (1.0 - alpha / 2.0)))

    # A component whose interval contains 0 is labeled "no demonstrated benefit"
    label = "no demonstrated benefit" if (lo <= 0.0 <= hi) else "demonstrated benefit"
    return lo, hi, label


def get_ablation_variants(
    base_spec: ExperimentSpec,
    split: str = "dev",
) -> list[tuple[str, ExperimentSpec, float]]:
    """Generate ordered ablation variants V0 through V5 and LOO variants.

    Each progressive variant differs from the previous by exactly one flag or configuration.
    """
    variants: list[tuple[str, ExperimentSpec, float]] = []

    # V0: Signal only, fixed fraction, zero cost
    spec_v0 = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="zero_cost",
        sizing="fixed_fraction",
        flags={"vol_filter": False, "causal_gate": False, "meta_label": False},
        seeds=list(base_spec.seeds),
    )
    variants.append(("V0", spec_v0, 0.0))

    # V1: Add realistic costs
    spec_v1 = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="realistic",
        sizing="fixed_fraction",
        flags={"vol_filter": False, "causal_gate": False, "meta_label": False},
        seeds=list(base_spec.seeds),
    )
    variants.append(("V1", spec_v1, 1.0))

    # V2: Add volatility filter
    spec_v2 = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="realistic",
        sizing="fixed_fraction",
        flags={"vol_filter": True, "causal_gate": False, "meta_label": False},
        seeds=list(base_spec.seeds),
    )
    variants.append(("V2", spec_v2, 1.0))

    # V3: Add Half-Kelly plus CVaR sizing
    spec_v3 = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="realistic",
        sizing="half_kelly",
        flags={"vol_filter": True, "causal_gate": False, "meta_label": False},
        seeds=list(base_spec.seeds),
    )
    variants.append(("V3", spec_v3, 1.0))

    # V4: Add causal gate
    spec_v4 = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="realistic",
        sizing="half_kelly",
        flags={"vol_filter": True, "causal_gate": True, "meta_label": False},
        seeds=list(base_spec.seeds),
    )
    variants.append(("V4", spec_v4, 1.0))

    # V5: Add meta-labeling (only if model family supports side signals)
    spec_v5 = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="realistic",
        sizing="half_kelly",
        flags={"vol_filter": True, "causal_gate": True, "meta_label": True},
        seeds=list(base_spec.seeds),
    )
    variants.append(("V5", spec_v5, 1.0))

    # Leave-One-Out (LOO) variants from full pipeline
    # LOO_costs: full minus realistic costs (zero cost)
    spec_loo_costs = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="zero_cost",
        sizing="half_kelly",
        flags={"vol_filter": True, "causal_gate": True, "meta_label": False},
        seeds=list(base_spec.seeds),
    )
    variants.append(("LOO_costs", spec_loo_costs, 0.0))

    # LOO_vol_filter: full minus volatility filter
    spec_loo_vol = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="realistic",
        sizing="half_kelly",
        flags={"vol_filter": False, "causal_gate": True, "meta_label": False},
        seeds=list(base_spec.seeds),
    )
    variants.append(("LOO_vol_filter", spec_loo_vol, 1.0))

    # LOO_causal_gate: full minus causal gate
    spec_loo_causal = ExperimentSpec(
        family=base_spec.family,
        params=dict(base_spec.params),
        universe_name=base_spec.universe_name,
        split=split,
        role="ablation",
        execution="realistic",
        sizing="half_kelly",
        flags={"vol_filter": True, "causal_gate": False, "meta_label": False},
        seeds=list(base_spec.seeds),
    )
    variants.append(("LOO_causal_gate", spec_loo_causal, 1.0))

    return variants


def run_ablation_study(
    protocol_id: str,
    ledger: Ledger | None = None,
    data: ResearchData | None = None,
) -> list[dict[str, Any]]:
    """Execute complete ablation study on the selected candidate with highest dev DSR."""
    led = ledger or Ledger()
    rd = data or ResearchData(protocol_id=protocol_id, ledger=led)
    chain = protocol_chain(protocol_id, led)
    chain_set = set(chain)

    records = led._read_all_records()
    selected_records = [
        r for r in records
        if r.get("kind") == "experiment"
        and r.get("payload", {}).get("role") == "selected"
        and r.get("payload", {}).get("protocol_id") in chain_set
    ]

    if not selected_records:
        logger.warning("No candidate selected for ablation study.")
        return []

    # Identify ablation target: candidate with highest dev DSR
    candidates_with_dsr: list[tuple[float, dict[str, Any]]] = []
    for sel in selected_records:
        sp = sel["payload"]
        run_id = sp["run_id"]
        try:
            dsr_val = dev_dsr(run_id, led)
        except Exception:
            dsr_val = 0.0
        candidates_with_dsr.append((dsr_val, sp))

    candidates_with_dsr.sort(key=lambda x: x[0], reverse=True)
    target_dsr, target_cand = candidates_with_dsr[0]
    base_spec_dict = target_cand["spec"]
    base_spec = ExperimentSpec(
        family=base_spec_dict["family"],
        params=base_spec_dict.get("params", {}),
        universe_name=base_spec_dict.get("universe_name", "etfs"),
        split="dev",
        role="ablation",
        execution=base_spec_dict.get("execution", "realistic"),
        sizing=base_spec_dict.get("sizing", "half_kelly"),
        flags=base_spec_dict.get("flags", {}),
        seeds=base_spec_dict.get("seeds", [42]),
    )

    logger.info("Ablation target: %s (%s) with dev DSR=%.4f", base_spec.family, base_spec.spec_hash[:12], target_dsr)

    results_table: list[dict[str, Any]] = []

    # Run variants on dev and val
    for split in ["dev", "val"]:
        variants = get_ablation_variants(base_spec, split=split)
        prev_rets: np.ndarray | None = None

        for name, spec, cost_mult in variants:
            # V5 meta-labeling skipped with NOT_RUN if family is not an ML/side-signal family
            if name == "V5" and spec.family != "ml_meta":
                not_run_payload = {
                    "protocol_id": protocol_id,
                    "target_family": base_spec.family,
                    "variant": name,
                    "split": split,
                    "status": "NOT_RUN",
                    "reason": f"Family {base_spec.family} does not support side meta-label signals",
                }
                led.append(
                    kind="experiment",
                    status="NOT_RUN",
                    tag=f"ablation.{name}.{split}",
                    payload=not_run_payload,
                )
                results_table.append(not_run_payload)
                continue

            # Run experiment variant
            exp_res = run_experiment(spec, protocol_id, led, rd, cost_multiplier=cost_mult)
            p_file = Path(exp_res["results_path"])
            df = load_experiment_results(p_file)
            rets = df["net_return"].values
            shrp = float(exp_res["metrics"].get("sharpe", 0.0))

            # Paired block-bootstrap vs previous variant
            ci_lo, ci_hi, label = 0.0, 0.0, "baseline"
            if prev_rets is not None and len(prev_rets) == len(rets):
                ci_lo, ci_hi, label = paired_bootstrap_sharpe_diff_ci(rets, prev_rets, seed=1)

            prev_rets = rets

            payload = {
                "protocol_id": protocol_id,
                "target_family": base_spec.family,
                "variant": name,
                "split": split,
                "sharpe": shrp,
                "paired_ci": [ci_lo, ci_hi],
                "label": label,
                "metrics": exp_res["metrics"],
                "status": "PASS",
            }

            led.append(
                kind="experiment",
                status="PASS",
                tag=f"ablation.{name}.{split}",
                payload=payload,
            )
            results_table.append(payload)

    return results_table
