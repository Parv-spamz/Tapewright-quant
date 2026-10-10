"""Unit tests for research experiment harness, cost models, and trial counting.

All test fixtures use synthetic offline data and temporary protocols marked explicitly
as synthetic test fixtures to prevent confusion with real research manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from integrity.ledger import Ledger
from research.data_access import ResearchData
from research.experiment import ExperimentSpec, run_experiment, trial_count
from research.protocol import Protocol, compute_splits_hash, register_protocol


def create_synthetic_test_bar_data(n_bars: int = 100) -> pd.DataFrame:
    """Generate deterministic synthetic bar data for offline test isolation."""
    dates = pd.date_range("2010-01-01", periods=n_bars, freq="B")
    # Synthetic upward ramp: prices from 100 to 150
    close = np.linspace(100.0, 150.0, n_bars)
    return pd.DataFrame({
        "open": close * 0.995,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 100_000.0,
    }, index=dates)


class MockResearchData(ResearchData):
    """Mock ResearchData holding synthetic in-memory partition frames for tests."""

    def __init__(self, synthetic_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
        self.synthetic_frames = synthetic_frames
        self.is_synthetic = True

    def load_partition(self, symbol: str, split: str) -> pd.DataFrame:
        if symbol not in self.synthetic_frames:
            raise KeyError(f"Symbol {symbol} not in synthetic test data")
        if split not in self.synthetic_frames[symbol]:
            raise KeyError(f"Split {split} not in synthetic test data")
        return self.synthetic_frames[symbol][split].copy()


@pytest.fixture
def test_environment(tmp_path: Path):
    """Fixture providing a temporary ledger, protocol, and synthetic ResearchData."""
    ledger_path = tmp_path / "test_ledger.jsonl"
    ledger = Ledger(str(ledger_path))

    # Construct and register a temporary synthetic protocol
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for ds_id in ["syn_spy", "syn_xlk"]:
        (manifest_dir / f"{ds_id}.json").write_text(
            json.dumps({"dataset_id": ds_id, "source_kind": "real_mock"}),
            encoding="utf-8",
        )

    proto = Protocol(
        name="synthetic_test_protocol",
        universe=["SPY", "XLK"],
        dataset_ids=["syn_spy", "syn_xlk"],
        splits_hash=compute_splits_hash(),
        strategy_families=["crossover", "breakout", "buy_and_hold_spy"],
        cost_model={
            "equities": {
                "half_spread_bps": 2.0,
                "commission": 1.0,
                "sqrt_impact_adv_coef": 0.1,
            }
        },
        sizing={"method": "Half-Kelly"},
        survival_rule={"dev_dsr_min": 0.90},
        max_trials=100,
        known_biases=["synthetic test fixture"],
    )
    proto_id = register_protocol(proto, ledger, manifest_dir=manifest_dir)

    df_spy = create_synthetic_test_bar_data(100)
    df_xlk = create_synthetic_test_bar_data(100)

    data = MockResearchData({
        "SPY": {"dev": df_spy, "val": df_spy},
        "XLK": {"dev": df_xlk, "val": df_xlk},
    })

    return {
        "ledger": ledger,
        "protocol_id": proto_id,
        "data": data,
        "tmp_path": tmp_path,
    }


def test_running_without_registered_protocol_raises(test_environment: dict[str, Any]) -> None:
    """Attempting to run an experiment with an unregistered protocol ID raises ValueError."""
    spec = ExperimentSpec(
        family="crossover",
        params={"fast": 5, "slow": 50},
        universe_name="etfs",
        split="dev",
        role="trial",
    )
    with pytest.raises(ValueError, match="not registered"):
        run_experiment(
            spec=spec,
            protocol_id="unregistered_protocol_deadbeef",
            ledger=test_environment["ledger"],
            data=test_environment["data"],
        )


def test_duplicate_trial_returns_stored_record(test_environment: dict[str, Any]) -> None:
    """Duplicate trial run returns stored record and leaves ledger record count unchanged."""
    ledger: Ledger = test_environment["ledger"]
    proto_id: str = test_environment["protocol_id"]
    data: ResearchData = test_environment["data"]

    spec = ExperimentSpec(
        family="crossover",
        params={"fast": 5, "slow": 50},
        universe_name="etfs",
        split="dev",
        role="trial",
    )

    # First run
    res1 = run_experiment(spec, proto_id, ledger, data)
    records_after_first = len(ledger._read_all_records())

    # Second run with exact same spec
    res2 = run_experiment(spec, proto_id, ledger, data)
    records_after_second = len(ledger._read_all_records())

    assert records_after_first == records_after_second
    assert res1["results_sha256"] == res2["results_sha256"]
    assert res1["run_id"] == res2["run_id"]


def test_trial_count_equals_distinct_specs(test_environment: dict[str, Any]) -> None:
    """trial_count strictly measures distinct spec_hash with role 'trial' and split 'dev'."""
    ledger: Ledger = test_environment["ledger"]
    proto_id: str = test_environment["protocol_id"]
    data: ResearchData = test_environment["data"]

    spec1 = ExperimentSpec(
        family="crossover",
        params={"fast": 5, "slow": 50},
        universe_name="etfs",
        split="dev",
        role="trial",
    )
    spec2 = ExperimentSpec(
        family="crossover",
        params={"fast": 10, "slow": 50},
        universe_name="etfs",
        split="dev",
        role="trial",
    )
    spec_baseline = ExperimentSpec(
        family="buy_and_hold_spy",
        params={},
        universe_name="etfs",
        split="dev",
        role="baseline",
    )

    # Run spec1 (trial)
    run_experiment(spec1, proto_id, ledger, data)
    assert trial_count(proto_id, ledger) == 1

    # Run spec1 duplicate -> trial count stays 1
    run_experiment(spec1, proto_id, ledger, data)
    assert trial_count(proto_id, ledger) == 1

    # Run spec2 (trial) -> trial count increments to 2
    run_experiment(spec2, proto_id, ledger, data)
    assert trial_count(proto_id, ledger) == 2

    # Run baseline -> role 'baseline' does NOT count towards trial_count
    run_experiment(spec_baseline, proto_id, ledger, data)
    assert trial_count(proto_id, ledger) == 2


def test_buy_and_hold_zero_cost_reproduces_hand_computed_equity(test_environment: dict[str, Any]) -> None:
    """Zero-cost buy-and-hold matches hand-computed mathematical final equity."""
    ledger: Ledger = test_environment["ledger"]
    proto_id: str = test_environment["protocol_id"]
    data: ResearchData = test_environment["data"]

    spec = ExperimentSpec(
        family="buy_and_hold_spy",
        params={},
        universe_name="etfs",
        split="dev",
        role="baseline",
    )

    res = run_experiment(spec, proto_id, ledger, data, cost_multiplier=0.0)
    final_eq = res["metrics"]["final_equity"]

    # In synthetic data: SPY goes from 100 to 150 on close
    df_spy = data.load_partition("SPY", "dev")
    p_first = float(df_spy["close"].iloc[0])
    p_last = float(df_spy["close"].iloc[-1])

    expected_return_ratio = p_last / p_first
    # With 100k cash and zero cost, equity equals 100k * (p_last / p_first)
    expected_equity = 100_000.0 * expected_return_ratio

    assert final_eq == pytest.approx(expected_equity, rel=1e-3)


def test_cost_increases_monotonically_reduce_final_equity(test_environment: dict[str, Any]) -> None:
    """Higher execution costs monotonically reduce final portfolio equity."""
    ledger: Ledger = test_environment["ledger"]
    proto_id: str = test_environment["protocol_id"]
    data: ResearchData = test_environment["data"]

    spec = ExperimentSpec(
        family="crossover",
        params={"fast": 5, "slow": 50},
        universe_name="etfs",
        split="dev",
        role="sensitivity",
    )

    res_0x = run_experiment(spec, proto_id, ledger, data, cost_multiplier=0.0)
    res_1x = run_experiment(spec, proto_id, ledger, data, cost_multiplier=1.0)
    res_2x = run_experiment(spec, proto_id, ledger, data, cost_multiplier=2.0)
    res_4x = run_experiment(spec, proto_id, ledger, data, cost_multiplier=4.0)

    eq_0x = res_0x["metrics"]["final_equity"]
    eq_1x = res_1x["metrics"]["final_equity"]
    eq_2x = res_2x["metrics"]["final_equity"]
    eq_4x = res_4x["metrics"]["final_equity"]

    assert eq_0x >= eq_1x >= eq_2x >= eq_4x


def test_fail_record_written_when_strategy_raises(test_environment: dict[str, Any]) -> None:
    """Strategy failure appends a record with status FAIL and re-raises exception."""
    ledger: Ledger = test_environment["ledger"]
    proto_id: str = test_environment["protocol_id"]
    data: ResearchData = test_environment["data"]

    spec = ExperimentSpec(
        family="crossover",
        params={"fast": 100, "slow": 10},  # invalid: fast > slow raises ValueError
        universe_name="etfs",
        split="dev",
        role="trial",
    )

    with pytest.raises(ValueError):
        run_experiment(spec, proto_id, ledger, data)

    # Check ledger for FAIL record
    records = ledger._read_all_records()
    fail_records = [
        r for r in records
        if r.get("kind") == "experiment" and r.get("status") == "FAIL"
    ]
    assert len(fail_records) >= 1
    assert "error" in fail_records[-1]["payload"]


def test_results_sha256_matches_saved_file(test_environment: dict[str, Any]) -> None:
    """results_sha256 in payload matches the SHA-256 of the saved parquet file on disk."""
    ledger: Ledger = test_environment["ledger"]
    proto_id: str = test_environment["protocol_id"]
    data: ResearchData = test_environment["data"]

    spec = ExperimentSpec(
        family="breakout",
        params={"n": 20},
        universe_name="etfs",
        split="dev",
        role="trial",
    )

    res = run_experiment(spec, proto_id, ledger, data)
    parquet_path = Path(res["results_path"])
    assert parquet_path.exists()

    actual_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    assert res["results_sha256"] == actual_hash
