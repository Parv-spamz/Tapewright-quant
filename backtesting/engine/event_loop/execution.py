from __future__ import annotations

from typing import Any
from backtesting.engine.event_loop.events import FillEvent, OrderEvent, TickEvent


class FillModel:
    """Protocol for order fill pricing, slippage, and fee modeling."""

    def fill(self, order: OrderEvent, bar: TickEvent) -> FillEvent | None:
        """Evaluate order fill feasibility against the incoming bar."""
        return None


class ZeroCostFillModel(FillModel):
    """Default zero-cost fill model executing market orders at open, and limit orders on touch."""

    def fill(self, order: OrderEvent, bar: TickEvent) -> FillEvent | None:
        if order.symbol != bar.symbol:
            return None

        if order.order_type == "MKT":
            return FillEvent(
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=float(bar.open),
                commission=0.0,
                slippage=0.0,
                order_id=order.order_id,
                partial=False,
            )

        elif order.order_type == "LMT":
            if order.limit_price is None:
                return None

            limit = float(order.limit_price)
            b_open = float(bar.open)
            b_low = float(bar.low)
            b_high = float(bar.high)

            if order.side == "BUY":
                # Fills if bar low <= limit price
                if b_low <= limit:
                    fill_price = min(b_open, limit)
                    return FillEvent(
                        timestamp=bar.timestamp,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        fill_price=fill_price,
                        commission=0.0,
                        slippage=0.0,
                        order_id=order.order_id,
                        partial=False,
                    )
            elif order.side == "SELL":
                # Fills if bar high >= limit price
                if b_high >= limit:
                    fill_price = max(b_open, limit)
                    return FillEvent(
                        timestamp=bar.timestamp,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        fill_price=fill_price,
                        commission=0.0,
                        slippage=0.0,
                        order_id=order.order_id,
                        partial=False,
                    )

        return None


class ExecutionHandler:
    """Base execution handler protocol."""

    def on_order(self, order: OrderEvent) -> None:
        """Submit an order to the execution queue."""
        return None

    def on_bar(self, bar: TickEvent) -> list[FillEvent]:
        """Process price bar against pending orders and return generated fills."""
        return []


class SimulatedExecutionHandler(ExecutionHandler):
    """Simulated execution maintaining pending queues with strict causal delay and TTL."""

    def __init__(
        self,
        data_handler: Any = None,
        fill_model: FillModel | None = None,
    ) -> None:
        self.data_handler = data_handler
        self.fill_model: FillModel = fill_model or ZeroCostFillModel()
        self.pending_orders: list[dict[str, Any]] = []

    def on_order(self, order: OrderEvent) -> None:
        """Enqueue order for execution on subsequent bars."""
        self.pending_orders.append({
            "order": order,
            "bars_active": 0,
        })

    def on_bar(self, bar: TickEvent) -> list[FillEvent]:
        """Attempt to fill pending orders against the current bar.

        Enforces strict causal separation: orders generated at or after bar.timestamp
        cannot fill on the same bar.
        """
        fills: list[FillEvent] = []
        remaining: list[dict[str, Any]] = []

        for item in self.pending_orders:
            order: OrderEvent = item["order"]

            # Filter for orders matching this bar's symbol
            if order.symbol != bar.symbol:
                remaining.append(item)
                continue

            # Strict causal guard: order submitted at time t cannot execute until time > t
            if order.timestamp >= bar.timestamp:
                remaining.append(item)
                continue

            # Attempt fill via fill model
            fill = self.fill_model.fill(order, bar)
            if fill is not None:
                fills.append(fill)
            else:
                # Order did not fill on this bar (e.g. limit order not touched)
                item["bars_active"] += 1
                # Check TTL expiration
                if order.ttl_bars is not None and item["bars_active"] >= order.ttl_bars:
                    # Expired order dropped from pending queue
                    continue
                else:
                    remaining.append(item)

        self.pending_orders = remaining
        return fills
