from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from backtesting.engine.event_loop.events import FillEvent, OrderEvent, TickEvent
from backtesting.engine.event_loop.execution import ExecutionHandler


def bars_from_tape(
    book_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    rule: str = "1min",
) -> pd.DataFrame:
    """Produce OHLCV TickEvent-ready bars from Level-2 book and trades tape."""
    from data.transformation.resamplers.bars import ticks_to_bars

    if not trades_df.empty:
        bars = ticks_to_bars(trades_df, rule=rule)
    else:
        bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # If book_df has mid prices, fill any empty gaps
    if not book_df.empty and "mid" in book_df.columns:
        df_book = book_df.copy()
        if not isinstance(df_book.index, pd.DatetimeIndex):
            df_book.index = pd.to_datetime(df_book.index)
        mid_bars = df_book["mid"].resample(rule).ohlc()
        mid_bars["volume"] = 0.0
        if bars.empty:
            bars = mid_bars
        else:
            bars = bars.combine_first(mid_bars)

    return bars.dropna(subset=["close"])


class LOBSimulator:
    """Level-2 limit order book simulator with queue priority, trade walking, and cancellations."""

    def __init__(
        self,
        book_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        cancel_rate: float = 0.0,
    ) -> None:
        self.book_df = book_df.copy()
        if not self.book_df.empty and not isinstance(self.book_df.index, pd.DatetimeIndex):
            self.book_df.index = pd.to_datetime(self.book_df.index)
        self.book_df.sort_index(inplace=True)

        self.trades_df = trades_df.copy()
        if not self.trades_df.empty:
            if "timestamp" in self.trades_df.columns:
                self.trades_df["timestamp"] = pd.to_datetime(self.trades_df["timestamp"])
                self.trades_df.sort_values(by="timestamp", inplace=True)
            elif isinstance(self.trades_df.index, pd.DatetimeIndex):
                self.trades_df["timestamp"] = self.trades_df.index
                self.trades_df.sort_values(by="timestamp", inplace=True)

        self.cancel_rate = float(cancel_rate)

        # Active limit orders: {order_id: dict_state}
        self.active_lmt_orders: dict[str, dict[str, Any]] = {}
        # Pending market orders waiting to execute on next tape step
        self.pending_orders: list[OrderEvent] = []

    def queue_position(self, order_id: str) -> float:
        """Return the current remaining queue size ahead for an active limit order."""
        if order_id in self.active_lmt_orders:
            return float(self.active_lmt_orders[order_id]["queue_ahead"])
        return 0.0

    def submit_order(self, order: OrderEvent, submission_time: Any) -> list[FillEvent]:
        """Submit an order into the LOB simulator.

        Market orders walk the visible book at submission_time.
        Limit orders join the back of the queue at submission_time.
        """
        sub_ts = pd.to_datetime(submission_time)

        if order.order_type == "MKT":
            return self._execute_market_order(order, sub_ts)
        elif order.order_type == "LMT":
            self._enqueue_limit_order(order, sub_ts)
            return []
        return []

    def _get_book_row(self, timestamp: pd.Timestamp) -> pd.Series | None:
        if self.book_df.empty:
            return None
        # Strict past-only book row at or before timestamp
        past_book = self.book_df.loc[self.book_df.index <= timestamp]
        if past_book.empty:
            return self.book_df.iloc[0]
        return past_book.iloc[-1]

    def _execute_market_order(self, order: OrderEvent, timestamp: pd.Timestamp) -> list[FillEvent]:
        """Execute market order by walking the visible Level-2 book levels."""
        book_row = self._get_book_row(timestamp)
        if book_row is None:
            return []

        qty = float(order.quantity)
        rem = qty
        total_dollars = 0.0

        if order.side.upper() == "BUY":
            # Walk ask levels: ask_px_1, ask_sz_1 ...
            levels: list[tuple[float, float]] = []
            for k in range(1, 6):
                px_col = f"ask_px_{k}"
                sz_col = f"ask_sz_{k}"
                if px_col in book_row and sz_col in book_row:
                    levels.append((float(book_row[px_col]), float(book_row[sz_col])))

            if not levels:
                levels = [(float(book_row.get("mid", 100.0)), 1000000.0)]

            # Consume levels best-to-worst
            for px, sz in levels:
                take = min(rem, sz)
                total_dollars += take * px
                rem -= take
                if rem <= 1e-9:
                    break

            # Any quantity beyond visible levels fills at last level + sqrt impact
            if rem > 1e-9:
                last_px, _ = levels[-1]
                total_visible = sum(sz for _, sz in levels) or 1000.0
                impact_slip = 0.005 * last_px * np.sqrt(rem / total_visible)
                extra_px = last_px + impact_slip
                total_dollars += rem * extra_px
                rem = 0.0

            vwap_price = total_dollars / qty
            best_ask = levels[0][0]
            slippage_cost = abs(vwap_price - best_ask) * qty

            fill = FillEvent(
                timestamp=timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=qty,
                fill_price=vwap_price,
                commission=0.0,
                slippage=slippage_cost,
                order_id=order.order_id,
                partial=False,
            )
            return [fill]

        else:  # SELL
            # Walk bid levels: bid_px_1, bid_sz_1 ...
            levels = []
            for k in range(1, 6):
                px_col = f"bid_px_{k}"
                sz_col = f"bid_sz_{k}"
                if px_col in book_row and sz_col in book_row:
                    levels.append((float(book_row[px_col]), float(book_row[sz_col])))

            if not levels:
                levels = [(float(book_row.get("mid", 100.0)), 1000000.0)]

            for px, sz in levels:
                take = min(rem, sz)
                total_dollars += take * px
                rem -= take
                if rem <= 1e-9:
                    break

            if rem > 1e-9:
                last_px, _ = levels[-1]
                total_visible = sum(sz for _, sz in levels) or 1000.0
                impact_slip = 0.005 * last_px * np.sqrt(rem / total_visible)
                extra_px = max(0.0001, last_px - impact_slip)
                total_dollars += rem * extra_px
                rem = 0.0

            vwap_price = total_dollars / qty
            best_bid = levels[0][0]
            slippage_cost = abs(best_bid - vwap_price) * qty

            fill = FillEvent(
                timestamp=timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=qty,
                fill_price=vwap_price,
                commission=0.0,
                slippage=slippage_cost,
                order_id=order.order_id,
                partial=False,
            )
            return [fill]

    def _enqueue_limit_order(self, order: OrderEvent, submission_time: pd.Timestamp) -> None:
        """Calculate queue priority and register active limit order."""
        book_row = self._get_book_row(submission_time)
        limit = float(order.limit_price if order.limit_price is not None else 0.0)

        ahead = 0.0
        if book_row is not None:
            best_bid = float(book_row.get("bid_px_1", 0.0))
            best_ask = float(book_row.get("ask_px_1", float("inf")))

            # Check if order is inside the spread
            if best_bid < limit < best_ask:
                ahead = 0.0
            else:
                if order.side.upper() == "BUY":
                    # Find matching bid level
                    found = False
                    for k in range(1, 6):
                        px_col = f"bid_px_{k}"
                        sz_col = f"bid_sz_{k}"
                        if px_col in book_row and abs(float(book_row[px_col]) - limit) < 1e-4:
                            ahead = float(book_row[sz_col])
                            found = True
                            break
                    if not found and limit <= best_bid:
                        ahead = float(book_row.get("bid_sz_1", 0.0))
                else:  # SELL
                    found = False
                    for k in range(1, 6):
                        px_col = f"ask_px_{k}"
                        sz_col = f"ask_sz_{k}"
                        if px_col in book_row and abs(float(book_row[px_col]) - limit) < 1e-4:
                            ahead = float(book_row[sz_col])
                            found = True
                            break
                    if not found and limit >= best_ask:
                        ahead = float(book_row.get("ask_sz_1", 0.0))

        self.active_lmt_orders[order.order_id] = {
            "order": order,
            "remaining_qty": float(order.quantity),
            "queue_ahead": float(ahead),
            "submitted_time": submission_time,
        }

    def process_tape_window(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> list[FillEvent]:
        """Step active limit orders through the tape window between start_time and end_time."""
        fills: list[FillEvent] = []
        if not self.active_lmt_orders:
            return fills

        # Slice trades strictly in (start_time, end_time]
        if not self.trades_df.empty:
            trades_slice = self.trades_df.loc[
                (self.trades_df["timestamp"] > start_time) & (self.trades_df["timestamp"] <= end_time)
            ]
        else:
            trades_slice = pd.DataFrame()

        current_book = self._get_book_row(end_time)

        order_ids_to_remove: list[str] = []

        for order_id, item in list(self.active_lmt_orders.items()):
            order: OrderEvent = item["order"]
            limit = float(order.limit_price if order.limit_price is not None else 0.0)

            # Strict causal guard: order submitted at t only processes trades after t
            if order.timestamp >= end_time:
                continue

            # 1. Taker check against current book
            if current_book is not None:
                if order.side.upper() == "BUY":
                    best_ask = float(current_book.get("ask_px_1", float("inf")))
                    if best_ask <= limit:
                        fills.append(
                            FillEvent(
                                timestamp=end_time,
                                symbol=order.symbol,
                                side=order.side,
                                quantity=item["remaining_qty"],
                                fill_price=best_ask,
                                commission=0.0,
                                slippage=0.0,
                                order_id=order_id,
                                partial=False,
                            )
                        )
                        order_ids_to_remove.append(order_id)
                        continue
                else:  # SELL
                    best_bid = float(current_book.get("bid_px_1", -float("inf")))
                    if best_bid >= limit:
                        fills.append(
                            FillEvent(
                                timestamp=end_time,
                                symbol=order.symbol,
                                side=order.side,
                                quantity=item["remaining_qty"],
                                fill_price=best_bid,
                                commission=0.0,
                                slippage=0.0,
                                order_id=order_id,
                                partial=False,
                            )
                        )
                        order_ids_to_remove.append(order_id)
                        continue

            # 2. Apply queue cancellation
            if item["queue_ahead"] > 0.0 and self.cancel_rate > 0.0:
                item["queue_ahead"] -= self.cancel_rate * item["queue_ahead"]

            # 3. Match against trades slice
            if not trades_slice.empty:
                for _, trade in trades_slice.iterrows():
                    trd_time = trade["timestamp"]
                    # Ensure trade occurred strictly after order submission
                    if trd_time <= order.timestamp:
                        continue

                    trd_px = float(trade["price"])
                    trd_sz = float(trade["size"])
                    trd_side = trade.get("side", 0)

                    if order.side.upper() == "BUY":
                        # Sell trade at price <= limit fills buy limit order
                        is_match = (trd_side in (-1, "SELL")) or (trd_px <= limit + 1e-5)
                        if is_match and trd_px <= limit + 1e-5:
                            if item["queue_ahead"] > 0.0:
                                if trd_sz >= item["queue_ahead"]:
                                    spillover = trd_sz - item["queue_ahead"]
                                    item["queue_ahead"] = 0.0
                                    if spillover > 0.0 and item["remaining_qty"] > 0.0:
                                        fill_qty = min(item["remaining_qty"], spillover)
                                        item["remaining_qty"] -= fill_qty
                                        is_partial = item["remaining_qty"] > 1e-6
                                        fills.append(
                                            FillEvent(
                                                timestamp=trd_time,
                                                symbol=order.symbol,
                                                side=order.side,
                                                quantity=fill_qty,
                                                fill_price=limit,
                                                commission=0.0,
                                                slippage=0.0,
                                                order_id=order_id,
                                                partial=is_partial,
                                            )
                                        )
                                        if not is_partial:
                                            order_ids_to_remove.append(order_id)
                                            break
                                else:
                                    item["queue_ahead"] -= trd_sz
                            else:
                                if item["remaining_qty"] > 0.0:
                                    fill_qty = min(item["remaining_qty"], trd_sz)
                                    item["remaining_qty"] -= fill_qty
                                    is_partial = item["remaining_qty"] > 1e-6
                                    fills.append(
                                        FillEvent(
                                            timestamp=trd_time,
                                            symbol=order.symbol,
                                            side=order.side,
                                            quantity=fill_qty,
                                            fill_price=limit,
                                            commission=0.0,
                                            slippage=0.0,
                                            order_id=order_id,
                                            partial=is_partial,
                                        )
                                    )
                                    if not is_partial:
                                        order_ids_to_remove.append(order_id)
                                        break

                    else:  # SELL
                        # Buy trade at price >= limit fills sell limit order
                        is_match = (trd_side in (1, "BUY")) or (trd_px >= limit - 1e-5)
                        if is_match and trd_px >= limit - 1e-5:
                            if item["queue_ahead"] > 0.0:
                                if trd_sz >= item["queue_ahead"]:
                                    spillover = trd_sz - item["queue_ahead"]
                                    item["queue_ahead"] = 0.0
                                    if spillover > 0.0 and item["remaining_qty"] > 0.0:
                                        fill_qty = min(item["remaining_qty"], spillover)
                                        item["remaining_qty"] -= fill_qty
                                        is_partial = item["remaining_qty"] > 1e-6
                                        fills.append(
                                            FillEvent(
                                                timestamp=trd_time,
                                                symbol=order.symbol,
                                                side=order.side,
                                                quantity=fill_qty,
                                                fill_price=limit,
                                                commission=0.0,
                                                slippage=0.0,
                                                order_id=order_id,
                                                partial=is_partial,
                                            )
                                        )
                                        if not is_partial:
                                            order_ids_to_remove.append(order_id)
                                            break
                                else:
                                    item["queue_ahead"] -= trd_sz
                            else:
                                if item["remaining_qty"] > 0.0:
                                    fill_qty = min(item["remaining_qty"], trd_sz)
                                    item["remaining_qty"] -= fill_qty
                                    is_partial = item["remaining_qty"] > 1e-6
                                    fills.append(
                                        FillEvent(
                                            timestamp=trd_time,
                                            symbol=order.symbol,
                                            side=order.side,
                                            quantity=fill_qty,
                                            fill_price=limit,
                                            commission=0.0,
                                            slippage=0.0,
                                            order_id=order_id,
                                            partial=is_partial,
                                        )
                                    )
                                    if not is_partial:
                                        order_ids_to_remove.append(order_id)
                                        break

        for o_id in order_ids_to_remove:
            self.active_lmt_orders.pop(o_id, None)

        return fills


class LOBExecutionHandler(ExecutionHandler):
    """Execution handler bridging the BacktestEngine with the Level-2 order book simulator."""

    def __init__(
        self,
        simulator: LOBSimulator,
    ) -> None:
        self.simulator = simulator
        self.pending_orders: list[OrderEvent] = []
        self.last_bar_time: pd.Timestamp | None = None

    def queue_position(self, order_id: str) -> float:
        """Return remaining queue size ahead for an active order."""
        return self.simulator.queue_position(order_id)

    def on_order(self, order: OrderEvent) -> None:
        """Enqueue order for execution on incoming tape intervals."""
        self.pending_orders.append(order)

    def on_bar(self, bar: TickEvent) -> list[FillEvent]:
        """Advance simulator through tape up to bar.timestamp and process pending orders."""
        fills: list[FillEvent] = []
        bar_ts = pd.to_datetime(bar.timestamp)

        prev_time = self.last_bar_time if self.last_bar_time is not None else bar_ts - pd.Timedelta(seconds=1)

        # 1. First process any pending orders that were submitted BEFORE this bar
        unprocessed: list[OrderEvent] = []
        for order in self.pending_orders:
            if order.timestamp < bar_ts:
                # Submit into simulator at the start of this bar interval
                sub_fills = self.simulator.submit_order(order, bar_ts)
                fills.extend(sub_fills)
            else:
                unprocessed.append(order)

        self.pending_orders = unprocessed

        # 2. Advance active limit orders through the tape window
        tape_fills = self.simulator.process_tape_window(start_time=prev_time, end_time=bar_ts)
        fills.extend(tape_fills)

        self.last_bar_time = bar_ts
        return fills
