from __future__ import annotations

from typing import Any
import pandas as pd

from backtesting.engine.event_loop.events import FillEvent, OrderEvent, SignalEvent, TickEvent
from config.trading.risk_limits import RiskLimits, check_order_allowed
from utils.logging.logger import get_logger


class Portfolio:
    """Portfolio accounting, risk limits enforcement, and position tracking."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        risk_limits: RiskLimits | None = None,
    ) -> None:
        if initial_cash <= 0.0:
            raise ValueError(f"initial_cash must be strictly positive, got {initial_cash}")

        self.initial_cash: float = float(initial_cash)
        self.cash: float = float(initial_cash)
        self.risk_limits: RiskLimits = risk_limits or RiskLimits()

        # Signed quantities per symbol: positive=long, negative=short
        self.positions: dict[str, float] = {}
        # Average cost basis per unit of current position
        self.cost_basis: dict[str, float] = {}
        # Accumulated commissions allocated to currently open position
        self.open_commissions: dict[str, float] = {}

        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.current_prices: dict[str, float] = {}
        self.equity: float = float(initial_cash)

        # Equity history marks
        self.equity_curve: list[dict[str, Any]] = []
        # Trade execution audit logs
        self.trades: list[dict[str, Any]] = []

        self.logger = get_logger("Portfolio")

    def on_signal(self, signal: SignalEvent) -> list[OrderEvent]:
        """Convert a strategy SignalEvent into sized OrderEvents gated by risk limits."""
        symbol = signal.symbol
        px = self.current_prices.get(symbol)
        if px is None or px <= 0.0:
            self.logger.warning("No valid market price recorded for %s to size signal", symbol)
            return []

        # Target portfolio allocation weight
        if signal.direction == "EXIT":
            target_weight = 0.0
        else:
            pct = signal.target_pct if signal.target_pct is not None else self.risk_limits.max_position_pct
            if signal.direction == "LONG":
                target_weight = abs(float(pct))
            elif signal.direction == "SHORT":
                target_weight = -abs(float(pct))
            else:
                target_weight = 0.0

        target_value = target_weight * self.equity
        current_qty = self.positions.get(symbol, 0.0)
        current_value = current_qty * px

        delta_value = target_value - current_value
        if abs(delta_value) < 1e-4:
            return []

        delta_qty = delta_value / px
        if abs(delta_qty) < 1e-6:
            return []

        # Risk limits check
        is_allowed, reason = check_order_allowed(
            limits=self.risk_limits,
            current_position_value=current_value,
            order_value=delta_value,
            equity=self.equity,
        )

        if not is_allowed:
            self.logger.warning("Order rejected by risk limits for %s: %s", symbol, reason)
            return []

        side = "BUY" if delta_qty > 0.0 else "SELL"
        order_qty = abs(delta_qty)

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            side=side,
            quantity=order_qty,
            order_type="MKT",
        )
        return [order]

    def on_fill(self, fill: FillEvent) -> None:
        """Process fill execution and update cash, positions, cost basis, and realized PnL."""
        symbol = fill.symbol
        qty = float(fill.quantity)
        px = float(fill.fill_price)
        comm = float(fill.commission)
        side = fill.side

        # 1. Cash accounting: BUY cash -= qty*price + comm; SELL cash += qty*price - comm
        if side == "BUY":
            self.cash -= (qty * px + comm)
            signed_delta = qty
        else:
            self.cash += (qty * px - comm)
            signed_delta = -qty

        curr_pos = self.positions.get(symbol, 0.0)
        curr_cost = self.cost_basis.get(symbol, 0.0)
        curr_open_comm = self.open_commissions.get(symbol, 0.0)

        # 2. Position updates and realized PnL calculation
        if abs(curr_pos) < 1e-9:
            # Opening new position from flat
            self.positions[symbol] = signed_delta
            self.cost_basis[symbol] = px
            self.open_commissions[symbol] = comm
        elif (curr_pos > 0 and signed_delta > 0) or (curr_pos < 0 and signed_delta < 0):
            # Expanding position in same direction
            new_pos = curr_pos + signed_delta
            new_cost = (abs(curr_pos) * curr_cost + qty * px) / abs(new_pos)
            self.positions[symbol] = new_pos
            self.cost_basis[symbol] = new_cost
            self.open_commissions[symbol] = curr_open_comm + comm
        else:
            # Opposite direction: closing, partial closing, or position flip
            if qty <= abs(curr_pos):
                # Partial or full close without reversal
                close_qty = qty
                fraction_closed = close_qty / abs(curr_pos)
                allocated_open_comm = curr_open_comm * fraction_closed

                if curr_pos > 0:
                    gross_pnl = (px - curr_cost) * close_qty
                else:
                    gross_pnl = (curr_cost - px) * close_qty

                net_trade_pnl = gross_pnl - comm - allocated_open_comm
                self.realized_pnl += net_trade_pnl

                self.trades.append({
                    "symbol": symbol,
                    "timestamp": fill.timestamp,
                    "qty": close_qty,
                    "entry_price": curr_cost,
                    "exit_price": px,
                    "pnl": net_trade_pnl,
                    "commission": comm + allocated_open_comm,
                })

                new_pos = curr_pos + signed_delta
                if abs(new_pos) < 1e-9:
                    self.positions[symbol] = 0.0
                    self.cost_basis[symbol] = 0.0
                    self.open_commissions[symbol] = 0.0
                else:
                    self.positions[symbol] = new_pos
                    self.open_commissions[symbol] = curr_open_comm * (1.0 - fraction_closed)
            else:
                # Position reversal (flipping from long to short or short to long)
                close_qty = abs(curr_pos)
                reversal_qty = qty - close_qty
                close_comm = comm * (close_qty / qty)
                reversal_comm = comm * (reversal_qty / qty)

                if curr_pos > 0:
                    gross_pnl = (px - curr_cost) * close_qty
                else:
                    gross_pnl = (curr_cost - px) * close_qty

                net_trade_pnl = gross_pnl - close_comm - curr_open_comm
                self.realized_pnl += net_trade_pnl

                self.trades.append({
                    "symbol": symbol,
                    "timestamp": fill.timestamp,
                    "qty": close_qty,
                    "entry_price": curr_cost,
                    "exit_price": px,
                    "pnl": net_trade_pnl,
                    "commission": close_comm + curr_open_comm,
                })

                new_pos = signed_delta + curr_pos
                self.positions[symbol] = new_pos
                self.cost_basis[symbol] = px
                self.open_commissions[symbol] = reversal_comm

    def mark_to_market(self, tick: TickEvent) -> None:
        """Mark portfolio to market at bar close."""
        self.current_prices[tick.symbol] = float(tick.close)

        unrealized = 0.0
        positions_value = 0.0

        for sym, pos in self.positions.items():
            if abs(pos) > 1e-9:
                px = self.current_prices.get(sym, self.cost_basis.get(sym, 0.0))
                cb = self.cost_basis.get(sym, 0.0)
                if pos > 0.0:
                    unrealized += (px - cb) * pos
                    positions_value += pos * px
                else:
                    unrealized += (cb - px) * abs(pos)
                    positions_value += pos * px

        self.unrealized_pnl = unrealized
        self.equity = self.cash + positions_value

        record = {
            "timestamp": tick.timestamp,
            "equity": self.equity,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
        }

        # Update or append mark for this timestamp
        if self.equity_curve and self.equity_curve[-1]["timestamp"] == tick.timestamp:
            self.equity_curve[-1] = record
        else:
            self.equity_curve.append(record)
