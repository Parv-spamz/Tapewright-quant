from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from backtesting.engine.event_loop.events import SignalEvent
from database.repositories.signal_repo.repo import SignalRepo


def persist_signals(
    signals: list[SignalEvent] | list[dict[str, Any]],
    signal_repo: SignalRepo,
) -> int:
    """Audit and persist generated trading signals to MySQL / SQLite repository.

    Maps SignalEvent domain attributes to the relational `trade_signals` schema:
    - asset_id: signal.symbol
    - model_id: signal.strategy_id
    - signal_type: BUY / SELL / EXIT
    - confidence_score: signal.strength
    - target_weight: signal.target_pct
    - timestamp: signal.timestamp

    Args:
        signals: List of SignalEvent instances or signal dictionaries.
        signal_repo: SignalRepo repository instance.

    Returns:
        int: Number of signals successfully written to the database.
    """
    if not signals:
        return 0

    records: list[dict[str, Any]] = []
    for s in signals:
        if isinstance(s, SignalEvent):
            if s.direction == "LONG":
                sig_type = "BUY"
            elif s.direction == "SHORT":
                sig_type = "SELL"
            else:
                sig_type = "EXIT"

            ts = str(s.timestamp) if isinstance(s.timestamp, (pd.Timestamp, np.datetime64)) else s.timestamp
            record = {
                "asset_id": s.symbol,
                "model_id": s.strategy_id,
                "signal_type": sig_type,
                "confidence_score": float(s.strength),
                "target_weight": float(s.target_pct) if s.target_pct is not None else None,
                "metadata": {"direction": s.direction},
                "timestamp": ts,
            }
            records.append(record)
        elif isinstance(s, dict):
            records.append(s)

    return signal_repo.insert_signals_bulk(records)
