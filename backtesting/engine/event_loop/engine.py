from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

from backtesting.engine.event_loop.data_handler import HistoricBarDataHandler
from backtesting.engine.event_loop.events import FillEvent, OrderEvent, SignalEvent
from backtesting.engine.event_loop.execution import ExecutionHandler
from backtesting.engine.event_loop.portfolio import Portfolio
from backtesting.engine.event_loop.strategy_base import Strategy


@dataclass
class BacktestResult:
    """Consolidated results of an event-driven backtest simulation run."""
    equity_curve: pd.DataFrame
    trades: list[dict[str, Any]]
    fills: list[FillEvent]
    signals: list[SignalEvent]
    orders: list[OrderEvent]


class BacktestEngine:
    """Discrete event-driven backtesting execution engine."""

    def __init__(
        self,
        data_handler: HistoricBarDataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
        persist_to_db: bool = False,
        connection_manager: Any = None,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.persist_to_db = persist_to_db
        self.connection_manager = connection_manager

        self.signals_history: list[SignalEvent] = []
        self.orders_history: list[OrderEvent] = []
        self.fills_history: list[FillEvent] = []

        self.events_queue: deque[Any] = deque()

        if self.persist_to_db:
            self._init_db_prerequisites()

    def _init_db_prerequisites(self) -> None:
        """Seed assets and model_registry entries to satisfy foreign key constraints."""
        try:
            from database.repositories.market_data_repo.repo import AssetRepo
            from database.repositories.market_data_repo.model_registry_repo import ModelRegistryRepo

            asset_repo = AssetRepo(self.connection_manager)
            model_repo = ModelRegistryRepo(self.connection_manager)

            for sym in self.data_handler.bars.keys():
                asset_repo.upsert_asset(
                    asset_id=sym,
                    symbol=sym,
                    name=sym,
                    asset_class="EQUITY",
                    exchange="SIM",
                )

            for m_id in ["default_strategy", "buy_and_hold", "fixed_schedule"]:
                try:
                    model_repo.register(
                        model_id=m_id,
                        model_name=m_id,
                        model_family="TIME_SERIES",
                        version="1.0.0",
                    )
                except Exception as e:
                    logger.warning("Failed to register model %s: %s", m_id, e)
        except Exception as e:
            logger.warning("Error in demo initialization: %s", e)

    def _persist_signal(self, signal: SignalEvent) -> int | None:
        try:
            from database.repositories.signal_repo.repo import SignalRepo
            repo = SignalRepo(self.connection_manager)

            direction_map = {
                "LONG": "BUY",
                "SHORT": "SELL",
                "EXIT": "HOLD",
            }
            signal_type = direction_map.get(signal.direction, "BUY")
            model_id = signal.strategy_id if signal.strategy_id in ["buy_and_hold", "fixed_schedule"] else "default_strategy"
            ts_val = (
                signal.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if hasattr(signal.timestamp, "strftime")
                else str(signal.timestamp)
            )
            sig_id = repo.insert_signal(
                asset_id=signal.symbol,
                model_id=model_id,
                signal_type=signal_type,
                confidence_score=float(min(1.0, max(0.0, signal.strength))),
                target_weight=float(signal.target_pct) if signal.target_pct is not None else None,
                timestamp=ts_val,
            )
            return sig_id
        except Exception as e:
            logger.warning("Failed to persist signal: %s", e)
            return None

    def _persist_fill(self, fill: FillEvent, signal_id: int | None = None) -> None:
        try:
            from database.repositories.order_repo.repo import OrderRepo
            repo = OrderRepo(self.connection_manager)

            ts_val = (
                fill.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if hasattr(fill.timestamp, "strftime")
                else str(fill.timestamp)
            )

            repo.insert_order(
                order_id=fill.order_id,
                asset_id=fill.symbol,
                side=fill.side,
                order_type="MARKET",
                quantity=fill.quantity,
                executed_price=fill.fill_price,
                signal_id=signal_id,
                slippage_cost=fill.slippage,
                fee=fill.commission,
                executed_at=ts_val,
            )
        except Exception as e:
            logger.warning("Failed to persist fill: %s", e)

    def run(self) -> BacktestResult:
        """Run the event-driven simulation loop across all historical bars."""
        last_signal_id: int | None = None

        while self.data_handler.continue_backtest:
            # Advance bar stream
            ticks = self.data_handler.update_bars()
            if not ticks:
                continue

            for tick in ticks:
                # Step 1: Execution fills pending orders against THIS bar's open/range first
                fills = self.execution_handler.on_bar(tick)
                for fill in fills:
                    self.portfolio.on_fill(fill)
                    self.fills_history.append(fill)
                    if self.persist_to_db:
                        self._persist_fill(fill, signal_id=last_signal_id)

                # Step 2: Portfolio mark to market at the close
                self.portfolio.mark_to_market(tick)

                # Step 3: Strategy on_bar -> signals -> portfolio on_signal -> orders queued for next bar
                signals = self.strategy.on_bar(tick, self.data_handler)
                for sig in signals:
                    self.signals_history.append(sig)
                    if self.persist_to_db:
                        last_signal_id = self._persist_signal(sig)

                    orders = self.portfolio.on_signal(sig)
                    for order in orders:
                        self.orders_history.append(order)
                        self.execution_handler.on_order(order)

        # Build final equity curve DataFrame
        if self.portfolio.equity_curve:
            df_equity = pd.DataFrame(self.portfolio.equity_curve)
            df_equity.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            df_equity.set_index("timestamp", inplace=True)
            df_equity.sort_index(inplace=True)
        else:
            df_equity = pd.DataFrame(columns=["equity", "cash", "realized_pnl", "unrealized_pnl"])

        return BacktestResult(
            equity_curve=df_equity,
            trades=list(self.portfolio.trades),
            fills=list(self.fills_history),
            signals=list(self.signals_history),
            orders=list(self.orders_history),
        )
