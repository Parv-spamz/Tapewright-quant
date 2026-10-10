"""Strategy parameter grid expansion and dev execution harness.

Runs all strategy family parameter combinations on the dev data split,
maintains multi-asset equal capital sleeve execution, and records trials
honestly in the tamper-evident provenance ledger.
"""

from __future__ import annotations

import logging
from typing import Any

from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.experiment import ExperimentSpec, run_experiment
from research.strategies_registry import get_family_grid

logger = logging.getLogger(__name__)

ALL_FAMILIES = [
    "crossover",
    "breakout",
    "mean_reversion",
    "xs_momentum",
    "pairs",
    "ml_meta",
]

BASELINES = [
    ("buy_and_hold_spy", {}, "etfs"),
    ("equal_weight_all_etfs", {}, "etfs"),
    ("sixty_forty_spy_ief", {}, "etfs"),
    ("crypto_buy_and_hold_btc", {}, "crypto"),
    ("crypto_equal_weight_btc_eth", {}, "crypto"),
]


def run_grid(
    family: str,
    protocol_id: str,
    ledger: Ledger | None = None,
    data: ResearchData | None = None,
) -> list[dict[str, Any]]:
    """Run all pre-registered parameter combinations for a family on dev data.

    Args:
        family: Name of the strategy family.
        protocol_id: Active pre-registered protocol identifier.
        ledger: Integrity ledger (default research/ledger/ledger.jsonl).
        data: ResearchData partition reader.

    Returns:
        List of experiment result dictionaries.
    """
    led = ledger or Ledger()
    rd = data or ResearchData()
    grid = get_family_grid(family)
    results: list[dict[str, Any]] = []

    for params in grid:
        seeds = [1, 2, 3, 4, 5] if family == "ml_meta" else [42]
        spec = ExperimentSpec(
            family=family,
            params=params,
            universe_name="etfs",
            split="dev",
            role="trial",
            seeds=seeds,
        )
        res = run_experiment(spec, protocol_id, led, rd)
        results.append(res)

    return results


def run_all_dev(
    protocol_id: str,
    ledger: Ledger | None = None,
    data: ResearchData | None = None,
) -> list[dict[str, Any]]:
    """Run every family grid and baseline on dev split.

    Executes all combinations across crossover, breakout, mean_reversion,
    xs_momentum, pairs, and ml_meta, followed by standard equity and crypto
    benchmarks with role 'baseline'.
    """
    led = ledger or Ledger()
    rd = data or ResearchData()
    all_results: list[dict[str, Any]] = []

    for fam in ALL_FAMILIES:
        logger.info("Executing grid for family %s on dev...", fam)
        fam_results = run_grid(fam, protocol_id, led, rd)
        all_results.extend(fam_results)

    # Baselines on dev
    for b_fam, b_params, u_name in BASELINES:
        b_spec = ExperimentSpec(
            family=b_fam,
            params=b_params,
            universe_name=u_name,
            split="dev",
            role="baseline",
        )
        try:
            res_b = run_experiment(b_spec, protocol_id, led, rd)
            all_results.append(res_b)
        except Exception as e:
            logger.warning("Baseline %s failed or skipped: %s", b_fam, e)

    return all_results
