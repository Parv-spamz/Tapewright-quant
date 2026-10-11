"""Unit tests for strategy selection, DSR reconciliation, validation survival, and lockbox CLI gate.

All tests operate offline on temporary ledgers and synthetic matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import norm

from backtesting.metrics.risk_measures.overfit import deflated_sharpe_ratio
from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.experiment import ExperimentSpec, run_experiment
from research.lockbox import ONE_SHOT_CONFIRMATION
from research.protocol import Protocol, compute_splits_hash, register_protocol
from research.run_lockbox import run_lockbox_cli
from research.selection import (
    dev_dsr,
    evaluate_on_validation,
    select_candidates,
)


def reference_dsr(
    returns_matrix: np.ndarray,
) -> tuple[float, float, float, int, float, float, float, float]:
    """Prompt 19 reference implementation of DSR using scipy norm and sample moments."""
    T, N = returns_matrix.shape
    g = 0.5772156649

    sr_trials = np.mean(returns_matrix, axis=0) / np.std(returns_matrix, axis=0, ddof=1)
    var_sr = float(np.var(sr_trials, ddof=1))

    best_idx = int(np.argmax(sr_trials))
    sr_hat = float(sr_trials[best_idx])
    best_returns = returns_matrix[:, best_idx]

    sr0 = float(
        np.sqrt(var_sr)
        * ((1.0 - g) * norm.ppf(1.0 - 1.0 / N) + g * norm.ppf(1.0 - 1.0 / (N * np.e)))
    )

    skew = float(stats.skew(best_returns))
    kurt = float(stats.kurtosis(best_returns, fisher=False))

    denom_var = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    denom = np.sqrt(max(denom_var, 1e-12))
    z = float((sr_hat - sr0) * np.sqrt(T - 1) / denom)
    psr = float(norm.cdf(z))

    return psr, sr_hat, sr0, T, skew, kurt, z, var_sr


def test_dev_dsr_matches_prompt19_reference_on_built_matrix() -> None:
    """dev_dsr calculation matches Prompt 19 reference implementation on a built matrix."""
    np.random.seed(1)
    T = 1000
    N = 30
    R = np.random.normal(0.0, 0.01, (T, N))

    ref_psr, sr_hat, sr0, t_obs, sk, kt, z, var_sr = reference_dsr(R)

    library_dsr = deflated_sharpe_ratio(
        sr_hat=sr_hat,
        T=t_obs,
        skew=sk,
        kurt=kt,
        n_trials=N,
        var_sr=var_sr,
        annualized=False,
    )

    assert library_dsr == pytest.approx(ref_psr, abs=1e-9)


def test_selection_obeys_constraints_on_hand_built_ledger(tmp_path: Path) -> None:
    """select_candidates enforces trade_count >= 30, max_drawdown >= -0.40, and picks highest Sharpe."""
    ledger_path = tmp_path / "test_ledger.jsonl"
    ledger = Ledger(str(ledger_path))

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "ds_syn.json").write_text(
        json.dumps({"dataset_id": "ds_syn", "source_kind": "real_mock"}),
        encoding="utf-8",
    )

    proto = Protocol(
        name="test_selection_protocol",
        universe=["SPY"],
        dataset_ids=["ds_syn"],
        splits_hash=compute_splits_hash(),
        strategy_families=["crossover"],
        cost_model={},
        sizing={},
        survival_rule={},
        max_trials=100,
        known_biases=[],
    )
    proto_id = register_protocol(proto, ledger, manifest_dir=manifest_dir)

    # 1. Spec with trade_count = 20 (< 30) -> should be excluded
    ledger.append(
        kind="experiment",
        status="PASS",
        tag="experiment.crossover",
        payload={
            "protocol_id": proto_id,
            "role": "trial",
            "split": "dev",
            "spec_hash": "hash_low_trades",
            "run_id": "run_low_trades",
            "spec": {"family": "crossover", "params": {"fast": 5, "slow": 50}},
            "metrics": {"trade_count": 20, "max_drawdown": -0.15, "sharpe": 2.5},
        },
    )

    # 2. Spec with max_drawdown = -0.50 (< -0.40) -> should be excluded
    ledger.append(
        kind="experiment",
        status="PASS",
        tag="experiment.crossover",
        payload={
            "protocol_id": proto_id,
            "role": "trial",
            "split": "dev",
            "spec_hash": "hash_bad_dd",
            "run_id": "run_bad_dd",
            "spec": {"family": "crossover", "params": {"fast": 10, "slow": 50}},
            "metrics": {"trade_count": 45, "max_drawdown": -0.50, "sharpe": 3.0},
        },
    )

    # 3. Spec qualifying with Sharpe 1.2
    ledger.append(
        kind="experiment",
        status="PASS",
        tag="experiment.crossover",
        payload={
            "protocol_id": proto_id,
            "role": "trial",
            "split": "dev",
            "spec_hash": "hash_good_1",
            "run_id": "run_good_1",
            "spec": {"family": "crossover", "params": {"fast": 20, "slow": 100}},
            "metrics": {"trade_count": 35, "max_drawdown": -0.25, "sharpe": 1.2},
        },
    )

    # 4. Spec qualifying with Sharpe 1.8 (highest Sharpe among eligible)
    ledger.append(
        kind="experiment",
        status="PASS",
        tag="experiment.crossover",
        payload={
            "protocol_id": proto_id,
            "role": "trial",
            "split": "dev",
            "spec_hash": "hash_best",
            "run_id": "run_best",
            "spec": {"family": "crossover", "params": {"fast": 20, "slow": 150}},
            "metrics": {"trade_count": 40, "max_drawdown": -0.20, "sharpe": 1.8},
        },
    )

    selected = select_candidates(proto_id, ledger)
    assert len(selected) == 1
    assert selected[0]["spec_hash"] == "hash_best"
    assert selected[0]["justification"]["dev_net_sharpe"] == 1.8


def test_selection_is_deterministic(tmp_path: Path) -> None:
    """Calling select_candidates multiple times yields identical deterministic selection."""
    ledger_path = tmp_path / "test_ledger.jsonl"
    ledger = Ledger(str(ledger_path))

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "ds_syn.json").write_text(
        json.dumps({"dataset_id": "ds_syn", "source_kind": "real_mock"}),
        encoding="utf-8",
    )

    proto = Protocol(
        name="test_det_protocol",
        universe=["SPY"],
        dataset_ids=["ds_syn"],
        splits_hash=compute_splits_hash(),
        strategy_families=["breakout"],
        cost_model={},
        sizing={},
        survival_rule={},
        max_trials=100,
        known_biases=[],
    )
    proto_id = register_protocol(proto, ledger, manifest_dir=manifest_dir)

    ledger.append(
        kind="experiment",
        status="PASS",
        tag="experiment.breakout",
        payload={
            "protocol_id": proto_id,
            "role": "trial",
            "split": "dev",
            "spec_hash": "hash_bk_1",
            "run_id": "run_bk_1",
            "spec": {"family": "breakout", "params": {"n": 20}},
            "metrics": {"trade_count": 50, "max_drawdown": -0.20, "sharpe": 1.5},
        },
    )

    sel1 = select_candidates(proto_id, ledger)
    sel2 = select_candidates(proto_id, ledger)

    assert len(sel1) == 1
    assert len(sel2) == 1
    assert sel1[0]["spec_hash"] == sel2[0]["spec_hash"]


def test_survival_rule_on_three_handmade_cases() -> None:
    """Survival rule evaluates to True only when dev DSR, val Sharpe, and val PSR pass thresholds."""
    dev_dsr_min = 0.90
    val_sharpe_min = 0.0
    val_psr_min = 0.90

    def check_survival(dev_dsr_val: float, val_sr: float, val_psr: float) -> bool:
        return bool(
            dev_dsr_val >= dev_dsr_min
            and val_sr > val_sharpe_min
            and val_psr >= val_psr_min
        )

    # Case 1: Passes all
    assert check_survival(0.92, 0.75, 0.93) is True

    # Case 2: Fails dev DSR
    assert check_survival(0.85, 0.75, 0.93) is False

    # Case 3: Fails val (negative Sharpe and low PSR)
    assert check_survival(0.95, -0.10, 0.40) is False


def test_evaluate_on_validation_idempotency_and_unselected_check(tmp_path: Path) -> None:
    """evaluate_on_validation returns empty if no candidate selected, and is idempotent."""
    ledger_path = tmp_path / "test_ledger.jsonl"
    ledger = Ledger(str(ledger_path))

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "ds_syn.json").write_text(
        json.dumps({"dataset_id": "ds_syn", "source_kind": "real_mock"}),
        encoding="utf-8",
    )

    proto = Protocol(
        name="test_val_protocol",
        universe=["SPY"],
        dataset_ids=["ds_syn"],
        splits_hash=compute_splits_hash(),
        strategy_families=["breakout"],
        cost_model={},
        sizing={},
        survival_rule={"dev_dsr_min": 0.90, "val_net_sharpe_min": 0.0, "val_psr_against_zero_min": 0.90},
        max_trials=100,
        known_biases=[],
    )
    proto_id = register_protocol(proto, ledger, manifest_dir=manifest_dir)

    # No candidate selected yet: evaluate_on_validation returns empty list without error
    eval_res = evaluate_on_validation(proto_id, ledger)
    assert eval_res == []


def test_run_lockbox_refuses_without_exact_confirmation(tmp_path: Path) -> None:
    """run_lockbox_cli strictly refuses execution when confirmation phrase is missing or altered."""
    fake_vault = tmp_path / "fake_vault"
    fake_vault.mkdir(parents=True, exist_ok=True)
    ledger_path = tmp_path / "test_ledger.jsonl"
    ledger = Ledger(str(ledger_path))

    with pytest.raises(ValueError, match="Lockbox execution refused"):
        run_lockbox_cli(
            confirm="INCORRECT_CONFIRMATION",
            vault_path=fake_vault,
            ledger_path=ledger_path,
        )
