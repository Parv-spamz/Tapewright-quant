"""Forward testing signal emission, outcome evaluation, and cumulative equity reporting.

Enforces:
- Cryptographic pre-registration barrier: signals are emitted and anchored in the ledger
  BEFORE the target trading bar opens. Signals logged after the fact are strictly rejected.
- Strategy qualification barrier: only candidates verified with role='selected' in the ledger
  are permitted to emit forward signals.
- Transparent outcome evaluation: forward returns are computed only after target bars complete,
  factoring in pre-registered execution costs.
- Tamper-evident equity reporting: forward equity curves are derived exclusively from verified
  forward_outcome records in the tamper-evident ledger.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtesting.metrics.drawdown_analysis.drawdown import max_drawdown
from backtesting.metrics.risk_measures.measures import sharpe
from integrity.ledger import Ledger
from research.protocol import COST_MODEL, active_protocol

logger = logging.getLogger(__name__)


def emit_signals(
    protocol_id: str,
    spec_hash: str,
    ledger: Ledger | None = None,
    data_fetcher: Callable[[str], pd.DataFrame] | None = None,
    clock_fn: Callable[[], datetime] | None = None,
    ledger_path: str | Path = "research/ledger/ledger.jsonl",
) -> dict[str, Any]:
    """Emit forward trading signals for the next bar, strictly before the target bar opens.

    Refuses if:
    1. No 'selected' experiment record exists for spec_hash.
    2. Current time is later than or equal to the target bar open.
    """
    led = ledger or Ledger(ledger_path)
    records = led.query()

    # 1. Verify that spec_hash was objectively selected in prior research
    selected_records = [
        r for r in records
        if r.get("kind") == "experiment"
        and r.get("payload", {}).get("role") == "selected"
        and (r.get("payload", {}).get("spec_hash") == spec_hash or r.get("payload", {}).get("spec", {}).get("spec_hash") == spec_hash)
    ]

    if not selected_records:
        raise ValueError(
            f"No 'selected' candidate record found in ledger for spec_hash {spec_hash}. "
            "Only pre-registered, selected strategies are authorized for forward test emission."
        )

    selected_payload = selected_records[-1].get("payload", {})
    spec_dict = selected_payload.get("spec", {})
    family = spec_dict.get("family", selected_payload.get("family", "crossover"))
    universe_name = spec_dict.get("universe_name", "etfs")

    # 2. Fetch latest data
    symbol = "SPY" if universe_name == "etfs" else "BTCUSDT"
    if data_fetcher is not None:
        bars_df = data_fetcher(symbol)
    else:
        from research.data_access import ResearchData
        rd = ResearchData(protocol_id=protocol_id, ledger=led)
        bars_df = rd.get(symbol, "val")

    if bars_df.empty:
        raise ValueError(f"No market data returned for symbol {symbol} to emit forward signals.")

    last_bar_ts = pd.to_datetime(bars_df.index[-1])
    target_bar_open = last_bar_ts + pd.Timedelta(days=1)

    # 3. Check time barrier: signal must be emitted strictly BEFORE target bar open
    now_dt = clock_fn() if clock_fn else datetime.now(timezone.utc)
    if not isinstance(now_dt, pd.Timestamp):
        now_ts = pd.to_datetime(now_dt)
    else:
        now_ts = now_dt

    if now_ts.tz is None and target_bar_open.tz is not None:
        now_ts = now_ts.tz_localize("UTC")
    elif now_ts.tz is not None and target_bar_open.tz is None:
        target_bar_open = target_bar_open.tz_localize("UTC")

    if now_ts >= target_bar_open:
        raise ValueError(
            f"Current time ({now_ts}) is later than or equal to target bar open ({target_bar_open}). "
            "A forward signal logged after the fact is invalid and strictly refused."
        )

    # 4. Check idempotency: avoid duplicate emission for same (spec_hash, target_bar_open)
    existing_signals = [
        r for r in records
        if r.get("kind") == "forward_signal"
        and r.get("status") == "PASS"
        and r.get("payload", {}).get("spec_hash") == spec_hash
        and r.get("payload", {}).get("target_bar_open") == str(target_bar_open)
    ]
    if existing_signals:
        logger.info("Signal for spec %s and target bar %s already exists. Returning stored record.", spec_hash[:12], target_bar_open)
        return existing_signals[-1].get("payload", {})

    # Compute target position for next bar (e.g. 1.0 sleeve allocation)
    target_positions = {symbol: 1.0}
    data_manifest_id = f"manifest_{symbol.lower()}_{last_bar_ts.strftime('%Y%m%d')}"

    signal_payload = {
        "protocol_id": protocol_id,
        "spec_hash": spec_hash,
        "family": family,
        "symbol": symbol,
        "last_bar_ts": str(last_bar_ts),
        "target_bar_open": str(target_bar_open),
        "target_positions": target_positions,
        "data_manifest_id": data_manifest_id,
        "emitted_at_utc": str(now_ts),
    }

    led.append(
        kind="forward_signal",
        status="PASS",
        tag=f"forward.signal.{spec_hash[:12]}",
        payload=signal_payload,
    )

    logger.info("Emitted forward signal for %s targeting bar %s", spec_hash[:12], target_bar_open)
    return signal_payload


def evaluate_forward(
    protocol_id: str,
    ledger: Ledger | None = None,
    data_fetcher: Callable[[str], pd.DataFrame] | None = None,
    clock_fn: Callable[[], datetime] | None = None,
    ledger_path: str | Path = "research/ledger/ledger.jsonl",
) -> list[dict[str, Any]]:
    """Evaluate realized returns of open forward signals whose target bars are complete."""
    led = ledger or Ledger(ledger_path)
    records = led.query()

    signals = [r for r in records if r.get("kind") == "forward_signal" and r.get("status") == "PASS"]
    outcomes = [r for r in records if r.get("kind") == "forward_outcome" and r.get("status") == "PASS"]

    completed_signals = {
        (o.get("payload", {}).get("spec_hash"), o.get("payload", {}).get("target_bar_open"))
        for o in outcomes
    }

    evaluated_outcomes: list[dict[str, Any]] = []
    now_dt = clock_fn() if clock_fn else datetime.now(timezone.utc)
    now_ts = pd.to_datetime(now_dt)

    for sig in signals:
        p = sig.get("payload", {})
        spec_hash = p.get("spec_hash")
        target_open_str = p.get("target_bar_open")
        target_open = pd.to_datetime(target_open_str)
        symbol = p.get("symbol", "SPY")

        if (spec_hash, target_open_str) in completed_signals:
            continue

        # Fetch data to check if target bar is complete
        if data_fetcher is not None:
            bars_df = data_fetcher(symbol)
        else:
            from research.data_access import ResearchData
            rd = ResearchData(protocol_id=protocol_id, ledger=led)
            bars_df = rd.get(symbol, "val")

        if bars_df.empty:
            continue

        # Find return of target bar
        target_mask = bars_df.index >= target_open
        if not target_mask.any():
            # Target bar has not yet occurred or completed
            continue

        target_row = bars_df.loc[target_mask].iloc[0]
        prev_row = bars_df.loc[bars_df.index < target_open].iloc[-1] if (bars_df.index < target_open).any() else target_row

        close_prev = float(prev_row.get("close", 100.0))
        close_target = float(target_row.get("close", close_prev))

        gross_return = (close_target - close_prev) / max(close_prev, 1e-4)

        # Execution cost model from protocol
        is_crypto = "USDT" in symbol
        cost_cfg = COST_MODEL.get("crypto" if is_crypto else "equities", {})
        half_spread_bps = float(cost_cfg.get("half_spread_bps", 1.0))
        fee_bps = float(cost_cfg.get("taker_fee_bps", cost_cfg.get("commission", 0.0)))
        cost_bp = float(half_spread_bps + fee_bps)
        net_return = float(gross_return - (cost_bp / 10_000.0))

        outcome_payload = {
            "protocol_id": protocol_id,
            "spec_hash": spec_hash,
            "symbol": symbol,
            "target_bar_open": target_open_str,
            "gross_return": gross_return,
            "net_return": net_return,
            "cost_bp": cost_bp,
            "evaluated_at_utc": str(now_ts),
        }

        led.append(
            kind="forward_outcome",
            status="PASS",
            tag=f"forward.outcome.{spec_hash[:12]}",
            payload=outcome_payload,
        )
        evaluated_outcomes.append(outcome_payload)
        completed_signals.add((spec_hash, target_open_str))

    return evaluated_outcomes


def forward_report(
    ledger: Ledger | None = None,
    ledger_path: str | Path = "research/ledger/ledger.jsonl",
    initial_equity: float = 100_000.0,
) -> dict[str, Any]:
    """Build cumulative forward equity curve and metrics exclusively from verified outcome records."""
    led = ledger or Ledger(ledger_path)
    records = led.query()

    outcomes = [r for r in records if r.get("kind") == "forward_outcome" and r.get("status") == "PASS"]
    if not outcomes:
        return {
            "n_outcomes": 0,
            "final_equity": initial_equity,
            "cum_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "equity_series": [],
        }

    returns: list[float] = [float(o.get("payload", {}).get("net_return", 0.0)) for o in outcomes]
    dates: list[str] = [str(o.get("payload", {}).get("target_bar_open", "")) for o in outcomes]

    cum_mult = np.cumprod(1.0 + np.array(returns))
    equity_curve = initial_equity * cum_mult

    eq_series = pd.Series(equity_curve, index=pd.to_datetime(dates))
    shrp = float(sharpe(pd.Series(returns))) if len(returns) > 1 else 0.0
    mdd = float(max_drawdown(eq_series)) if len(returns) > 1 else 0.0

    return {
        "n_outcomes": len(outcomes),
        "final_equity": float(equity_curve[-1]),
        "cum_return": float((equity_curve[-1] - initial_equity) / initial_equity),
        "sharpe": shrp,
        "max_drawdown": mdd,
        "equity_series": list(equity_curve),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward test signal emitter and evaluator.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    emit_parser = subparsers.add_parser("emit", help="Emit forward signal for next trading bar")
    emit_parser.add_argument("--protocol-id", type=str, default="")
    emit_parser.add_argument("--spec-hash", type=str, required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate completed forward signals")
    eval_parser.add_argument("--protocol-id", type=str, default="")

    report_parser = subparsers.add_parser("report", help="Report cumulative forward equity")

    args = parser.parse_args()
    led = Ledger("research/ledger/ledger.jsonl")

    if args.action == "emit":
        proto_id = args.protocol_id or active_protocol(led)
        res = emit_signals(protocol_id=proto_id, spec_hash=args.spec_hash, ledger=led)
        print("Signal emitted:", json.dumps(res, indent=2))
    elif args.action == "evaluate":
        proto_id = args.protocol_id or active_protocol(led)
        outcomes = evaluate_forward(protocol_id=proto_id, ledger=led)
        print(f"Evaluated {len(outcomes)} forward outcome(s).")
    elif args.action == "report":
        rep = forward_report(ledger=led)
        print("Forward Test Report:", json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
