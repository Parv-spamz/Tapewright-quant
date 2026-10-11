"""Parameter sensitivity surfaces, plateau scores, and cost break-even analysis.

Evaluates:
- Sharpe surface across parameter grids for selected strategy families
- Plateau score: fraction of neighboring parameter cells retaining >= 50% of peak Sharpe
- Cost sensitivity: performance under 0x, 1x, 2x, and 4x transaction costs
- Break-even multiplier: linear interpolation of the cost multiplier where net Sharpe reaches 0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.experiment import ExperimentSpec, run_experiment
from research.protocol import protocol_chain
from research.strategies_registry import get_family_grid

logger = logging.getLogger(__name__)


def plateau_score(
    grid_dict: dict[Any, float],
    best_key: Any,
) -> float:
    """Calculate the plateau score: share of neighboring grid cells whose Sharpe is >= 50% of peak.

    Args:
        grid_dict: Mapping of parameter keys (tuples or ints) to Sharpe ratios.
        best_key: Key corresponding to the peak Sharpe ratio.

    Returns:
        float: Share of adjacent neighbors with Sharpe >= 0.5 * best_sharpe.
    """
    if best_key not in grid_dict:
        return 0.0

    peak_sharpe = float(grid_dict[best_key])
    if peak_sharpe <= 0.0:
        return 0.0

    threshold = 0.5 * peak_sharpe

    # Helper to check adjacency
    def is_neighbor(k1: Any, k2: Any) -> bool:
        if k1 == k2:
            return False
        if isinstance(k1, (int, float)) and isinstance(k2, (int, float)):
            return abs(k1 - k2) <= 1  # Integer or unit neighbor
        if isinstance(k1, tuple) and isinstance(k2, tuple):
            if len(k1) != len(k2):
                return False
            # Neighbors if Chebyshev distance <= 1 across all dimension indices
            diffs = [abs(a - b) for a, b in zip(k1, k2)]
            return max(diffs) <= 1 and sum(diffs) > 0
        return False

    neighbors = [k for k in grid_dict if is_neighbor(best_key, k)]
    if not neighbors:
        return 1.0  # Isolated peak with no neighbors

    retained = sum(1 for k in neighbors if grid_dict[k] >= threshold)
    return float(retained / len(neighbors))


def break_even_cost_multiplier(
    multipliers: list[float],
    sharpes: list[float],
) -> float | str:
    """Calculate the cost multiplier where net Sharpe crosses zero using linear interpolation.

    Args:
        multipliers: Ordered list of non-negative cost multipliers (e.g. [0.0, 1.0, 2.0, 4.0]).
        sharpes: Net Sharpe ratio at each corresponding multiplier.

    Returns:
        float: Estimated multiplier where Sharpe crosses 0.0.
        'NOT_FOUND': If Sharpe remains strictly positive or strictly negative across all multipliers.
    """
    if len(multipliers) != len(sharpes) or len(multipliers) < 2:
        return "NOT_FOUND"

    for i in range(len(multipliers) - 1):
        m1, m2 = multipliers[i], multipliers[i + 1]
        s1, s2 = sharpes[i], sharpes[i + 1]

        # Check if 0 is crossed between s1 and s2
        if (s1 >= 0.0 >= s2) or (s1 <= 0.0 <= s2):
            if abs(s2 - s1) < 1e-12:
                return float(m1)
            # Linear interpolation: s(m) = s1 + (m - m1)/(m2 - m1) * (s2 - s1) = 0
            # m = m1 - s1 * (m2 - m1) / (s2 - s1)
            m_cross = m1 - s1 * (m2 - m1) / (s2 - s1)
            return float(m_cross)

    return "NOT_FOUND"


def run_sensitivity_study(
    protocol_id: str,
    ledger: Ledger | None = None,
    data: ResearchData | None = None,
) -> list[dict[str, Any]]:
    """Run sensitivity analysis for all selected candidates across parameter surfaces and costs."""
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

    results_table: list[dict[str, Any]] = []

    for sel in selected_records:
        sp = sel["payload"]
        fam = sp["family"]
        base_spec_dict = sp["spec"]
        base_spec = ExperimentSpec(
            family=fam,
            params=base_spec_dict.get("params", {}),
            universe_name=base_spec_dict.get("universe_name", "etfs"),
            split="dev",
            role="sensitivity",
            execution=base_spec_dict.get("execution", "realistic"),
            sizing=base_spec_dict.get("sizing", "half_kelly"),
            flags=base_spec_dict.get("flags", {}),
            seeds=base_spec_dict.get("seeds", [42]),
        )

        # 1. Parameter surface and plateau score
        grid_combos = get_family_grid(fam)
        surface_sharpes: dict[int, float] = {}
        best_idx = 0
        best_sr = -999.0

        for idx, p in enumerate(grid_combos):
            spec_p = ExperimentSpec(
                family=fam,
                params=p,
                universe_name=base_spec.universe_name,
                split="dev",
                role="trial",
                seeds=base_spec.seeds,
            )
            # Find trial in ledger
            match_rec = next(
                (
                    r for r in records
                    if r.get("kind") == "experiment"
                    and r.get("payload", {}).get("spec_hash") == spec_p.spec_hash
                    and r.get("payload", {}).get("split") == "dev"
                    and r.get("status") == "PASS"
                ),
                None,
            )
            sr = float(match_rec["payload"]["metrics"]["sharpe"]) if match_rec else 0.0
            surface_sharpes[idx] = sr
            if sr > best_sr:
                best_sr = sr
                best_idx = idx

        plateau = plateau_score(surface_sharpes, best_idx)

        # 2. Cost sensitivity: 0x, 1x, 2x, 4x multipliers
        cost_multipliers = [0.0, 1.0, 2.0, 4.0]
        cost_sharpes: list[float] = []

        for mult in cost_multipliers:
            c_res = run_experiment(base_spec, protocol_id, led, rd, cost_multiplier=mult)
            cost_sharpes.append(float(c_res["metrics"].get("sharpe", 0.0)))

        break_even = break_even_cost_multiplier(cost_multipliers, cost_sharpes)

        payload = {
            "protocol_id": protocol_id,
            "family": fam,
            "spec_hash": base_spec.spec_hash,
            "plateau_score": plateau,
            "cost_multipliers": cost_multipliers,
            "cost_sharpes": cost_sharpes,
            "break_even_multiplier": break_even,
        }

        led.append(
            kind="experiment",
            status="PASS",
            tag=f"sensitivity.{fam}",
            payload=payload,
        )
        results_table.append(payload)

    return results_table
