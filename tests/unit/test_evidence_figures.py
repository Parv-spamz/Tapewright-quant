"""Unit tests for evidence figure generation functions in backtesting/reports/tearsheet/figures.py.

Verification criteria:
1. Figure point counts equal input lengths.
2. NOT_RUN input yields a text figure containing 'NOT_RUN'.
3. Inputs are not mutated.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from backtesting.reports.tearsheet.figures import (
    ablation_benefit_figure,
    calibration_reliability_figure,
    evidence_metric_figure,
    forward_test_figure,
    not_run_figure,
    validation_survives_figure,
)


def _has_not_run_text(fig) -> bool:
    """Helper checking if a figure contains 'NOT_RUN' in any layout annotation."""
    return any("NOT_RUN" in str(ann.text) for ann in fig.layout.annotations)


# ==============================================================================
# 1. not_run_figure Tests
# ==============================================================================


def test_not_run_figure_text_and_provenance() -> None:
    fig = not_run_figure("No validation data recorded", title="Test Status", provenance="SYNTHETIC")
    assert _has_not_run_text(fig)
    assert any("PROVENANCE: SYNTHETIC" in str(ann.text) for ann in fig.layout.annotations)


# ==============================================================================
# 2. evidence_metric_figure Tests
# ==============================================================================


def test_evidence_metric_figure_point_count_and_no_mutation() -> None:
    raw_data = [1.2, 2.4, 3.1, 4.5, 5.0]
    input_copy = list(raw_data)

    fig = evidence_metric_figure(raw_data, title="Dev Metrics", provenance="REAL api polygon a1b2c3d4")
    # Point count equals input length
    assert len(fig.data[0].x) == len(raw_data)
    assert len(fig.data[0].y) == len(raw_data)

    # Input not mutated
    assert raw_data == input_copy


def test_evidence_metric_figure_not_run() -> None:
    fig_str = evidence_metric_figure("NOT_RUN: missing benchmark")
    assert _has_not_run_text(fig_str)

    fig_dict = evidence_metric_figure({"status": "NOT_RUN", "reason": "data store offline"})
    assert _has_not_run_text(fig_dict)

    fig_empty = evidence_metric_figure([])
    assert _has_not_run_text(fig_empty)


# ==============================================================================
# 3. validation_survives_figure Tests
# ==============================================================================


def test_validation_survives_figure_point_count_and_no_mutation() -> None:
    records = [
        {"name": "Trial 1", "score": 1.45, "survives": True},
        {"name": "Trial 2", "score": 0.85, "survives": False},
        {"name": "Trial 3", "score": 1.92, "survives": True},
    ]
    records_copy = copy.deepcopy(records)

    fig = validation_survives_figure(records, title="Validation Results", provenance="SYNTHETIC")
    # Point count equals input length
    assert len(fig.data[0].x) == len(records)
    assert len(fig.data[0].y) == len(records)

    # Input not mutated
    assert records == records_copy


def test_validation_survives_figure_not_run() -> None:
    fig_str = validation_survives_figure("NOT_RUN: stage unexecuted")
    assert _has_not_run_text(fig_str)

    fig_empty = validation_survives_figure([])
    assert _has_not_run_text(fig_empty)


# ==============================================================================
# 4. ablation_benefit_figure Tests
# ==============================================================================


def test_ablation_benefit_figure_point_count_and_no_mutation() -> None:
    records = [
        {"component": "vpin_filter", "delta": 0.25, "demonstrated_benefit": True},
        {"component": "fractional_diff", "delta": -0.05, "demonstrated_benefit": False},
        {"component": "kalman_hedge", "delta": 0.12, "demonstrated_benefit": True},
        {"component": "causal_gate", "delta": -0.01, "demonstrated_benefit": False},
    ]
    records_copy = copy.deepcopy(records)

    fig = ablation_benefit_figure(records, title="Ablation Comparison", provenance="REAL api polygon deadbeef")
    # Point count equals input length
    assert len(fig.data[0].x) == len(records)
    assert len(fig.data[0].y) == len(records)

    # Input not mutated
    assert records == records_copy


def test_ablation_benefit_figure_not_run() -> None:
    fig_str = ablation_benefit_figure("NOT_RUN: dev baseline only")
    assert _has_not_run_text(fig_str)

    fig_empty = ablation_benefit_figure([])
    assert _has_not_run_text(fig_empty)


# ==============================================================================
# 5. calibration_reliability_figure Tests
# ==============================================================================


def test_calibration_reliability_figure_point_count_and_no_mutation() -> None:
    cal_pts = [(0.1, 0.12), (0.3, 0.28), (0.5, 0.52), (0.7, 0.68), (0.9, 0.89)]
    cal_copy = list(cal_pts)

    fig = calibration_reliability_figure(cal_pts, title="Calibration Curve", provenance="SYNTHETIC")
    # Empirical trace is trace 1 (trace 0 is reference line)
    assert len(fig.data[1].x) == len(cal_pts)
    assert len(fig.data[1].y) == len(cal_pts)

    # Input not mutated
    assert cal_pts == cal_copy


def test_calibration_reliability_figure_not_run() -> None:
    fig_str = calibration_reliability_figure("NOT_RUN: no probability output logged")
    assert _has_not_run_text(fig_str)

    fig_dict = calibration_reliability_figure({"status": "NOT_RUN", "reason": "uncalibrated"})
    assert _has_not_run_text(fig_dict)

    fig_empty = calibration_reliability_figure([])
    assert _has_not_run_text(fig_empty)


# ==============================================================================
# 6. forward_test_figure Tests
# ==============================================================================


def test_forward_test_figure_point_count_and_no_mutation() -> None:
    data_series = pd.Series([100.0, 101.2, 100.8, 102.5, 103.1], index=pd.date_range("2024-01-01", periods=5))
    series_copy = data_series.copy()

    fig = forward_test_figure(data_series, title="Forward Test Tracking", provenance="REAL api polygon 12345678")
    # Point count equals input length
    assert len(fig.data[0].x) == len(data_series)
    assert len(fig.data[0].y) == len(data_series)

    # Input not mutated
    assert (data_series == series_copy).all()


def test_forward_test_figure_not_run() -> None:
    fig_str = forward_test_figure("NOT_RUN: forward paper trading unopened")
    assert _has_not_run_text(fig_str)

    fig_dict = forward_test_figure({"status": "NOT_RUN", "reason": "cold start"})
    assert _has_not_run_text(fig_dict)

    fig_empty = forward_test_figure([])
    assert _has_not_run_text(fig_empty)
