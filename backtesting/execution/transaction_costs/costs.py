from __future__ import annotations

from typing import Any


class CostModel:
    """Base cost model protocol."""

    def calculate(
        self,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MKT",
        is_liquidity_provider: bool = False,
    ) -> float:
        """Calculate total transaction fee/commission in currency units."""
        return 0.0


class PerShareCommission(CostModel):
    """Fixed commission per share with a minimum ticket charge."""

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0) -> None:
        if per_share < 0.0 or minimum < 0.0:
            raise ValueError("Commission rates must be non-negative.")
        self.per_share = float(per_share)
        self.minimum = float(minimum)

    def calculate(
        self,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MKT",
        is_liquidity_provider: bool = False,
    ) -> float:
        if quantity <= 0.0:
            return 0.0
        return max(self.minimum, self.per_share * float(quantity))


class PercentCommission(CostModel):
    """Commission charged as a percentage of traded notional."""

    def __init__(self, pct: float = 0.001) -> None:
        if pct < 0.0:
            raise ValueError("Commission percentage must be non-negative.")
        self.pct = float(pct)

    def calculate(
        self,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MKT",
        is_liquidity_provider: bool = False,
    ) -> float:
        notional = float(quantity) * float(price)
        return notional * self.pct


class MakerTakerFee(CostModel):
    """Exchange maker/taker schedule (MKT=taker, resting LMT=maker)."""

    def __init__(self, maker_bps: float = 2.0, taker_bps: float = 5.0) -> None:
        self.maker_bps = float(maker_bps)
        self.taker_bps = float(taker_bps)

    def calculate(
        self,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MKT",
        is_liquidity_provider: bool = False,
    ) -> float:
        notional = float(quantity) * float(price)
        # Resting limit orders that provide liquidity incur maker fee; market orders incur taker fee
        rate_bps = self.maker_bps if is_liquidity_provider else self.taker_bps
        return notional * (rate_bps * 1e-4)


class SecFee(CostModel):
    """SEC Section 31 transaction fee levied exclusively on SELL notional transactions.

    Note: The rate_per_million parameter reflects the regulatory assessment rate.
    Users must verify the current official fee rate per million from SEC fee rate advisories.
    """

    def __init__(self, rate_per_million: float = 27.80) -> None:
        if rate_per_million < 0.0:
            raise ValueError("SEC fee rate must be non-negative.")
        self.rate_per_million = float(rate_per_million)

    def calculate(
        self,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MKT",
        is_liquidity_provider: bool = False,
    ) -> float:
        # SEC fee applies strictly to round-turn sales (SELL side), zero on BUY
        if side.upper() != "SELL":
            return 0.0
        notional = float(quantity) * float(price)
        return notional * (self.rate_per_million / 1_000_000.0)


class CompositeCostModel(CostModel):
    """Aggregates multiple individual fee and commission models."""

    def __init__(self, models: list[CostModel]) -> None:
        self.models = list(models)

    def calculate(
        self,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "MKT",
        is_liquidity_provider: bool = False,
    ) -> float:
        return sum(
            m.calculate(
                side=side,
                quantity=quantity,
                price=price,
                order_type=order_type,
                is_liquidity_provider=is_liquidity_provider,
            )
            for m in self.models
        )
