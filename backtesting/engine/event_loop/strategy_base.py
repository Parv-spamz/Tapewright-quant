from __future__ import annotations

from typing import Any
from backtesting.engine.event_loop.events import SignalEvent, TickEvent


class Strategy:
    """Base class for event-driven strategies."""

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        """Generate trading signals on incoming tick event."""
        return []


class BuyAndHoldStrategy(Strategy):
    """Simple strategy that enters long on the very first bar for the designated symbol."""

    def __init__(self, symbol: str, target_pct: float = 1.0) -> None:
        self.symbol: str = symbol
        self.target_pct: float = float(target_pct)
        self.bought: bool = False

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        if tick.symbol == self.symbol and not self.bought:
            self.bought = True
            return [
                SignalEvent(
                    timestamp=tick.timestamp,
                    symbol=self.symbol,
                    direction="LONG",
                    strength=1.0,
                    strategy_id="buy_and_hold",
                    target_pct=self.target_pct,
                )
            ]
        return []


class FixedScheduleStrategy(Strategy):
    """Strategy that executes a deterministic schedule of signals keyed by timestamp or bar count."""

    def __init__(
        self,
        schedule: list[SignalEvent] | dict[Any, list[SignalEvent]],
    ) -> None:
        if isinstance(schedule, list):
            self.schedule_by_time: dict[Any, list[SignalEvent]] = {}
            for sig in schedule:
                self.schedule_by_time.setdefault(sig.timestamp, []).append(sig)
        else:
            self.schedule_by_time = schedule

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        signals = self.schedule_by_time.get(tick.timestamp, [])
        # Return signals for this symbol
        return [s for s in signals if s.symbol == tick.symbol]
