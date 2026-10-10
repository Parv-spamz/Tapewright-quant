from __future__ import annotations

import numpy as np


def half_spread_cost(price: float, spread_frac: float) -> float:
    """Calculate half-spread cost per share given current price and fractional spread."""
    return float(price) * (float(spread_frac) / 2.0)


class SlippageModel:
    """Base protocol class for price slippage and market impact models."""

    def calculate_impact(
        self,
        side: str,
        quantity: float,
        price: float,
        adv: float | None = None,
        sigma: float | None = None,
    ) -> float:
        """Calculate signed price change delta_P (positive for BUY, negative for SELL)."""
        return 0.0

    def get_execution_price(
        self,
        side: str,
        price: float,
        quantity: float = 1.0,
        adv: float | None = None,
        sigma: float | None = None,
    ) -> float:
        """Return execution price including price impact."""
        delta = self.calculate_impact(side=side, quantity=quantity, price=price, adv=adv, sigma=sigma)
        return float(price) + delta


class FixedBpsSlippage(SlippageModel):
    """Constant basis points slippage penalty."""

    def __init__(self, bps: float = 5.0) -> None:
        if bps < 0.0:
            raise ValueError("Slippage bps must be non-negative.")
        self.bps = float(bps)

    def calculate_impact(
        self,
        side: str,
        quantity: float,
        price: float,
        adv: float | None = None,
        sigma: float | None = None,
    ) -> float:
        sign = 1.0 if side.upper() == "BUY" else -1.0
        return sign * float(price) * (self.bps * 1e-4)


class LinearImpactSlippage(SlippageModel):
    """Linear market impact model: fractional impact = k * (qty / ADV)."""

    def __init__(self, k: float = 0.1, default_adv: float = 1_000_000.0) -> None:
        if k < 0.0:
            raise ValueError("Linear impact coefficient k must be non-negative.")
        self.k = float(k)
        self.default_adv = float(default_adv)

    def calculate_impact(
        self,
        side: str,
        quantity: float,
        price: float,
        adv: float | None = None,
        sigma: float | None = None,
    ) -> float:
        vol_adv = float(adv if adv is not None and adv > 0.0 else self.default_adv)
        frac = self.k * (float(quantity) / vol_adv)
        sign = 1.0 if side.upper() == "BUY" else -1.0
        return sign * float(price) * frac


class SquareRootImpactSlippage(SlippageModel):
    """Square-root market impact: delta_P = c * sign(Q) * sigma * sqrt(|Q| / ADV) * price."""

    def __init__(
        self,
        c: float = 0.5,
        default_adv: float = 1_000_000.0,
        default_sigma: float = 0.01,
    ) -> None:
        if c < 0.0:
            raise ValueError("Square-root impact coefficient c must be non-negative.")
        self.c = float(c)
        self.default_adv = float(default_adv)
        self.default_sigma = float(default_sigma)

    def calculate_impact(
        self,
        side: str,
        quantity: float,
        price: float,
        adv: float | None = None,
        sigma: float | None = None,
    ) -> float:
        vol_adv = float(adv if adv is not None and adv > 0.0 else self.default_adv)
        vol_sigma = float(sigma if sigma is not None and sigma > 0.0 else self.default_sigma)
        sign = 1.0 if side.upper() == "BUY" else -1.0
        participation = float(quantity) / vol_adv
        return self.c * sign * vol_sigma * np.sqrt(max(0.0, participation)) * float(price)


class AdverseSelectionPenalty:
    """Microstructure adverse selection multiplier based on flow toxicity (VPIN).

    multiplier = 1 + gamma * max(0, vpin - vpin_threshold) / (1 - vpin_threshold)
    """

    def __init__(self, gamma: float = 0.5, vpin_threshold: float = 0.5) -> None:
        if gamma < 0.0:
            raise ValueError("Adverse selection gamma must be non-negative.")
        if not (0.0 <= vpin_threshold < 1.0):
            raise ValueError("vpin_threshold must be in [0, 1).")
        self.gamma = float(gamma)
        self.vpin_threshold = float(vpin_threshold)

    def calculate_multiplier(self, vpin: float) -> float:
        """Calculate adverse impact scaling multiplier."""
        excess_vpin = max(0.0, float(vpin) - self.vpin_threshold)
        scale = excess_vpin / (1.0 - self.vpin_threshold)
        return 1.0 + self.gamma * scale
