from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid
import pandas as pd


@dataclass
class Event:
    """Base event containing a strict timestamp."""
    timestamp: Any

    def __post_init__(self) -> None:
        if self.timestamp is None:
            raise ValueError("Event timestamp cannot be None.")
        if not isinstance(self.timestamp, pd.Timestamp):
            self.timestamp = pd.to_datetime(self.timestamp)


@dataclass
class TickEvent(Event):
    """Market tick / OHLCV bar event for a single symbol."""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("TickEvent symbol must be a non-empty string.")
        if self.open < 0.0 or self.high < 0.0 or self.low < 0.0 or self.close < 0.0:
            raise ValueError("TickEvent price fields must be non-negative.")
        if self.high < self.low:
            raise ValueError(f"TickEvent high ({self.high}) cannot be less than low ({self.low}).")
        if self.volume < 0.0:
            raise ValueError("TickEvent volume cannot be negative.")


@dataclass
class SignalEvent(Event):
    """Trading signal emitted by a strategy."""
    symbol: str = ""
    direction: str = "LONG"  # "LONG", "SHORT", "EXIT"
    strength: float = 1.0
    strategy_id: str = "default_strategy"
    target_pct: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.symbol:
            raise ValueError("SignalEvent symbol must be a non-empty string.")
        if self.direction not in {"LONG", "SHORT", "EXIT"}:
            raise ValueError(f"SignalEvent direction must be LONG, SHORT, or EXIT; got '{self.direction}'.")
        if self.strength < 0.0:
            raise ValueError(f"SignalEvent strength must be non-negative, got {self.strength}.")
        if self.target_pct is not None and not (-1.0 <= self.target_pct <= 1.0):
            raise ValueError(f"SignalEvent target_pct must be between -1.0 and 1.0, got {self.target_pct}.")


@dataclass
class OrderEvent(Event):
    """Order submitted to an execution handler."""
    symbol: str = ""
    side: str = "BUY"  # "BUY", "SELL"
    quantity: float = 0.0
    order_type: str = "MKT"  # "MKT", "LMT"
    limit_price: float | None = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ttl_bars: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.symbol:
            raise ValueError("OrderEvent symbol must be a non-empty string.")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError(f"OrderEvent side must be BUY or SELL; got '{self.side}'.")
        if self.quantity <= 0.0:
            raise ValueError(f"OrderEvent quantity must be strictly > 0, got {self.quantity}.")
        if self.order_type not in {"MKT", "LMT"}:
            raise ValueError(f"OrderEvent order_type must be MKT or LMT; got '{self.order_type}'.")
        if self.order_type == "LMT":
            if self.limit_price is None or self.limit_price <= 0.0:
                raise ValueError("Limit orders require a strictly positive limit_price.")
        if self.ttl_bars is not None and self.ttl_bars <= 0:
            raise ValueError(f"ttl_bars must be positive, got {self.ttl_bars}.")
        if not self.order_id:
            self.order_id = str(uuid.uuid4())


@dataclass
class FillEvent(Event):
    """Transaction fill report returned by an execution handler."""
    symbol: str = ""
    side: str = "BUY"  # "BUY", "SELL"
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    order_id: str = ""
    partial: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.symbol:
            raise ValueError("FillEvent symbol must be a non-empty string.")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError(f"FillEvent side must be BUY or SELL; got '{self.side}'.")
        if self.quantity <= 0.0:
            raise ValueError(f"FillEvent quantity must be strictly > 0, got {self.quantity}.")
        if self.fill_price < 0.0:
            raise ValueError(f"FillEvent fill_price cannot be negative, got {self.fill_price}.")
        if self.commission < 0.0:
            raise ValueError(f"FillEvent commission cannot be negative, got {self.commission}.")
