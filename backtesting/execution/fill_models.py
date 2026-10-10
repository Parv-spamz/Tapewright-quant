from __future__ import annotations

from typing import Any
import pandas as pd

from backtesting.engine.event_loop.events import FillEvent, OrderEvent, TickEvent
from backtesting.engine.event_loop.execution import FillModel
from backtesting.execution.slippage_models.models import (
    AdverseSelectionPenalty,
    SlippageModel,
    half_spread_cost,
)
from backtesting.execution.transaction_costs.costs import CostModel


class RealisticFillModel(FillModel):
    """Realistic fill simulator modeling half-spread crossing, market impact, adverse selection, and fees."""

    def __init__(
        self,
        slippage: SlippageModel,
        cost_model: CostModel,
        adverse_selection: AdverseSelectionPenalty | None = None,
        spread_estimator: Any | None = None,
        default_spread_frac: float = 0.0005,
    ) -> None:
        self.slippage = slippage
        self.cost_model = cost_model
        self.adverse_selection = adverse_selection
        self.spread_estimator = spread_estimator
        self.default_spread_frac = float(default_spread_frac)

        # Context containers for dynamic series lookups
        self.vpin_series: dict[str, pd.Series] = {}
        self.adv_series: dict[str, pd.Series] = {}
        self.sigma_series: dict[str, pd.Series] = {}

    def set_vpin_series(self, symbol: str, series: pd.Series) -> None:
        """Register trailing VPIN series for a symbol."""
        self.vpin_series[symbol] = series

    def set_adv_series(self, symbol: str, series: pd.Series) -> None:
        """Register trailing ADV series for a symbol."""
        self.adv_series[symbol] = series

    def set_sigma_series(self, symbol: str, series: pd.Series) -> None:
        """Register trailing volatility series for a symbol."""
        self.sigma_series[symbol] = series

    def _get_spread_frac(self, bar: TickEvent) -> float:
        if self.spread_estimator is None:
            return self.default_spread_frac
        if callable(self.spread_estimator):
            return float(self.spread_estimator(bar))
        if hasattr(self.spread_estimator, "estimate"):
            return float(self.spread_estimator.estimate(bar))
        if isinstance(self.spread_estimator, (int, float)):
            return float(self.spread_estimator)
        return self.default_spread_frac

    def _get_past_value(self, registry: dict[str, pd.Series], symbol: str, timestamp: Any) -> float | None:
        s = registry.get(symbol)
        if s is None or s.empty:
            return None
        # Strict past-only lookup: <= timestamp
        past = s.loc[s.index <= timestamp]
        if past.empty:
            return None
        return float(past.iloc[-1])

    def fill(self, order: OrderEvent, bar: TickEvent) -> FillEvent | None:
        """Simulate order execution with spread, impact, adverse selection, and commission."""
        if order.symbol != bar.symbol:
            return None

        b_open = float(bar.open)
        qty = float(order.quantity)

        if order.order_type == "MKT":
            spread_frac = self._get_spread_frac(bar)
            spread_charge = half_spread_cost(b_open, spread_frac)

            adv = self._get_past_value(self.adv_series, order.symbol, bar.timestamp)
            sigma = self._get_past_value(self.sigma_series, order.symbol, bar.timestamp)

            raw_impact = abs(
                self.slippage.calculate_impact(
                    side=order.side,
                    quantity=qty,
                    price=b_open,
                    adv=adv,
                    sigma=sigma,
                )
            )

            if self.adverse_selection is not None:
                vpin_val = self._get_past_value(self.vpin_series, order.symbol, bar.timestamp)
                vpin_safe = vpin_val if vpin_val is not None else 0.0
                adverse_mult = self.adverse_selection.calculate_multiplier(vpin_safe)
            else:
                adverse_mult = 1.0

            total_impact = raw_impact * adverse_mult
            per_share_slippage = spread_charge + total_impact

            if order.side.upper() == "BUY":
                fill_price = b_open + per_share_slippage
            else:
                fill_price = max(0.0001, b_open - per_share_slippage)

            slippage_cost = per_share_slippage * qty

            commission = self.cost_model.calculate(
                side=order.side,
                quantity=qty,
                price=fill_price,
                order_type="MKT",
                is_liquidity_provider=False,  # taker fee
            )

            return FillEvent(
                timestamp=bar.timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=qty,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage_cost,
                order_id=order.order_id,
                partial=False,
            )

        elif order.order_type == "LMT":
            if order.limit_price is None:
                return None

            limit = float(order.limit_price)
            b_low = float(bar.low)
            b_high = float(bar.high)

            touched = False
            fill_price = limit

            if order.side.upper() == "BUY":
                if b_low <= limit:
                    touched = True
                    fill_price = min(b_open, limit)
            elif order.side.upper() == "SELL":
                if b_high >= limit:
                    touched = True
                    fill_price = max(b_open, limit)

            if touched:
                commission = self.cost_model.calculate(
                    side=order.side,
                    quantity=qty,
                    price=fill_price,
                    order_type="LMT",
                    is_liquidity_provider=True,  # maker fee
                )
                return FillEvent(
                    timestamp=bar.timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=qty,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=0.0,
                    order_id=order.order_id,
                    partial=False,
                )

        return None
