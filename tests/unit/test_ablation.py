"""Unit tests for ablation variants, paired bootstrap CI, sensitivity plateau, and stress drawdown.

All tests run offline on deterministic hand-constructed inputs and stated seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.metrics.risk_measures.measures import sharpe
from research.ablation import get_ablation_variants, paired_bootstrap_sharpe_diff_ci
from research.experiment import ExperimentSpec
from research.real_stress import calculate_peak_to_trough_drawdown
from research.sensitivity import break_even_cost_multiplier, plateau_score


def test_ablation_variants_produced_in_order_differ_by_one_flag() -> None:
    """Ablation variants V0 to V5 are produced in strict order and each differs from the previous by 1 attribute."""
    base_spec = ExperimentSpec(
        family="crossover",
        params={"fast": 20, "slow": 50},
        universe_name="etfs",
        split="dev",
        role="ablation",
    )
    variants = get_ablation_variants(base_spec, split="dev")

    names = [v[0] for v in variants[:6]]
    assert names == ["V0", "V1", "V2", "V3", "V4", "V5"]

    v0_spec = variants[0][1]
    v1_spec = variants[1][1]
    v2_spec = variants[2][1]
    v3_spec = variants[3][1]
    v4_spec = variants[4][1]
    v5_spec = variants[5][1]

    # V0 -> V1: zero_cost to realistic
    assert v0_spec.execution == "zero_cost"
    assert v1_spec.execution == "realistic"
    assert v0_spec.flags == v1_spec.flags
    assert v0_spec.sizing == v1_spec.sizing

    # V1 -> V2: vol_filter toggled to True
    assert v1_spec.flags["vol_filter"] is False
    assert v2_spec.flags["vol_filter"] is True
    assert v1_spec.sizing == v2_spec.sizing

    # V2 -> V3: sizing toggled from fixed_fraction to half_kelly
    assert v2_spec.sizing == "fixed_fraction"
    assert v3_spec.sizing == "half_kelly"
    assert v2_spec.flags == v3_spec.flags

    # V3 -> V4: causal_gate toggled to True
    assert v3_spec.flags["causal_gate"] is False
    assert v4_spec.flags["causal_gate"] is True

    # V4 -> V5: meta_label toggled to True
    assert v4_spec.flags["meta_label"] is False
    assert v5_spec.flags["meta_label"] is True


def test_paired_bootstrap_ci_contains_true_difference_planted_case() -> None:
    """Paired block-bootstrap CI contains the true empirical Sharpe difference in a planted test case."""
    rng = np.random.default_rng(42)
    n = 1000
    r_old = rng.normal(0.0005, 0.01, n)
    # Plant a small alpha boost
    r_new = r_old + 0.0003

    true_diff = float(sharpe(r_new) - sharpe(r_old))

    lo, hi, label = paired_bootstrap_sharpe_diff_ci(
        r_new,
        r_old,
        block_size=10,
        n_boot=2000,
        seed=1,
    )

    assert lo <= true_diff <= hi
    assert label == "demonstrated benefit"


def test_plateau_score_on_hand_surface() -> None:
    """plateau_score correctly calculates the share of adjacent neighbors retaining >= 50% peak Sharpe."""
    # Peak at (1, 1) with Sharpe 2.0 -> threshold is 1.0
    surface = {
        (1, 1): 2.0,  # Peak
        (0, 1): 1.5,  # Neighbor 1: >= 1.0 (retained)
        (2, 1): 1.2,  # Neighbor 2: >= 1.0 (retained)
        (1, 0): 0.8,  # Neighbor 3: < 1.0 (decayed)
        (1, 2): 0.5,  # Neighbor 4: < 1.0 (decayed)
    }

    score = plateau_score(surface, (1, 1))
    assert score == pytest.approx(0.50, abs=1e-6)


def test_break_even_cost_multiplier_on_hand_example() -> None:
    """Linear interpolation accurately identifies the cost multiplier where net Sharpe crosses 0."""
    mults = [0.0, 1.0, 2.0, 4.0]
    sharpes = [2.0, 1.0, -0.5, -3.0]

    # Between 1.0 (Sharpe 1.0) and 2.0 (Sharpe -0.5):
    # m = 1.0 - 1.0 * (2.0 - 1.0) / (-0.5 - 1.0) = 1.0 + 1.0 / 1.5 = 1.6666667
    expected_m = 1.0 + (1.0 / 1.5)
    be_m = break_even_cost_multiplier(mults, sharpes)

    assert be_m == pytest.approx(expected_m, abs=1e-6)

    # Strictly positive returns NOT_FOUND
    assert break_even_cost_multiplier(mults, [2.0, 1.5, 1.0, 0.5]) == "NOT_FOUND"


def test_calculate_peak_to_trough_drawdown_to_1e12() -> None:
    """calculate_peak_to_trough_drawdown returns exact peak-to-trough drawdown within 1e-12 precision."""
    # Peak is 120.0, trough is 90.0 -> (90 - 120) / 120 = -30 / 120 = -0.25
    prices = pd.Series([100.0, 120.0, 90.0, 110.0])
    dd = calculate_peak_to_trough_drawdown(prices)

    assert dd == pytest.approx(-0.25, abs=1e-12)
