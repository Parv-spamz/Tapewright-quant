"""Unit tests for forward testing harness, time-barrier enforcement, and ledger anchoring."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from integrity.anchor import anchor_ledger
from integrity.ledger import Ledger
from integrity.stages import stage_forward
from research.forward import emit_signals, evaluate_forward, forward_report


def _create_mock_bars(end_date: str = "2026-10-10") -> pd.DataFrame:
    dates = pd.date_range(end=end_date, periods=5, freq="D", tz="UTC")
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [102.0, 103.0, 104.0, 105.0, 106.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    return pd.DataFrame(data, index=dates)


def test_signal_after_target_bar_open_is_refused(tmp_path: Path) -> None:
    ledger_file = tmp_path / "ledger.jsonl"
    ledger = Ledger(str(ledger_file))
    spec_hash = "mock_spec_12345"

    ledger.append(
        kind="experiment",
        status="PASS",
        tag="test.selected",
        payload={"spec_hash": spec_hash, "role": "selected", "family": "crossover"},
    )

    bars_df = _create_mock_bars("2026-10-10")
    # Last bar is 2026-10-10. Target bar open is 2026-10-11.
    # Set clock to 2026-10-11 12:00:00 (after target bar open)
    fake_clock = lambda: datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="Current time .* is later than or equal to target bar open"):
        emit_signals(
            protocol_id="proto_test",
            spec_hash=spec_hash,
            ledger=ledger,
            data_fetcher=lambda sym: bars_df,
            clock_fn=fake_clock,
        )


def test_outcome_without_signal_is_impossible(tmp_path: Path) -> None:
    ledger_file = tmp_path / "ledger.jsonl"
    ledger = Ledger(str(ledger_file))
    spec_hash = "mock_spec_orphan"

    # Outcome appended directly without prior signal
    ledger.append(
        kind="forward_outcome",
        status="PASS",
        tag="forward.outcome.orphan",
        payload={
            "protocol_id": "proto_test",
            "spec_hash": spec_hash,
            "target_bar_open": "2026-10-11 00:00:00+00:00",
            "net_return": 0.015,
        },
    )

    import argparse
    args = argparse.Namespace(ledger=str(ledger_file))
    status, msg = stage_forward.run(args)
    assert status == "FAIL"
    assert "has no preceding forward_signal" in msg


def test_forward_equity_matches_hand_compounding(tmp_path: Path) -> None:
    ledger_file = tmp_path / "ledger.jsonl"
    ledger = Ledger(str(ledger_file))

    r1, r2, r3 = 0.01, -0.005, 0.02
    initial_eq = 100_000.0
    expected_final_eq = initial_eq * (1.0 + r1) * (1.0 + r2) * (1.0 + r3)
    expected_cum_ret = (expected_final_eq - initial_eq) / initial_eq

    dates = ["2026-10-11", "2026-10-12", "2026-10-13"]
    for d, r in zip(dates, [r1, r2, r3]):
        ledger.append(
            kind="forward_outcome",
            status="PASS",
            tag=f"outcome.{d}",
            payload={
                "spec_hash": "test_spec",
                "target_bar_open": d,
                "net_return": r,
            },
        )

    report = forward_report(ledger=ledger, initial_equity=initial_eq)
    assert report["n_outcomes"] == 3
    assert np.isclose(report["final_equity"], expected_final_eq, atol=1e-6)
    assert np.isclose(report["cum_return"], expected_cum_ret, atol=1e-6)


def test_emit_twice_idempotency(tmp_path: Path) -> None:
    ledger_file = tmp_path / "ledger.jsonl"
    ledger = Ledger(str(ledger_file))
    spec_hash = "mock_spec_idem"

    ledger.append(
        kind="experiment",
        status="PASS",
        tag="test.selected",
        payload={"spec_hash": spec_hash, "role": "selected", "family": "crossover"},
    )

    bars_df = _create_mock_bars("2026-10-10")
    # Clock is 2026-10-10 20:00:00 (before 2026-10-11 open)
    fake_clock = lambda: datetime(2026, 10, 10, 20, 0, 0, tzinfo=timezone.utc)

    res1 = emit_signals(
        protocol_id="proto_idem",
        spec_hash=spec_hash,
        ledger=ledger,
        data_fetcher=lambda sym: bars_df,
        clock_fn=fake_clock,
    )
    res2 = emit_signals(
        protocol_id="proto_idem",
        spec_hash=spec_hash,
        ledger=ledger,
        data_fetcher=lambda sym: bars_df,
        clock_fn=fake_clock,
    )

    assert res1 == res2

    signal_records = [r for r in ledger.query() if r.get("kind") == "forward_signal"]
    assert len(signal_records) == 1


def test_anchor_appends_one_line_per_call(tmp_path: Path) -> None:
    ledger_file = tmp_path / "ledger.jsonl"
    anchors_file = tmp_path / "ANCHORS.txt"
    ledger = Ledger(str(ledger_file))
    ledger.append(kind="audit_note", status="PASS", tag="test.entry", payload={"v": 1.0})

    anchor_ledger(ledger_path=ledger_file, anchors_path=anchors_file)
    with open(anchors_file, "r", encoding="utf-8") as f:
        lines1 = f.readlines()
    assert len(lines1) == 1
    assert "seq=1" in lines1[0]

    ledger.append(kind="audit_note", status="PASS", tag="test.entry2", payload={"v": 2.0})
    anchor_ledger(ledger_path=ledger_file, anchors_path=anchors_file)
    with open(anchors_file, "r", encoding="utf-8") as f:
        lines2 = f.readlines()
    assert len(lines2) == 2
    assert "seq=2" in lines2[1]
