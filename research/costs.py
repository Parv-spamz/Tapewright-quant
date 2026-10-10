"""Cost and execution model mapping for research backtests.

Maps pre-registered protocol cost parameters to backtest slippage and commission models.
Supports cost scaling multipliers for sensitivity analyses (e.g., 0x, 1x, 2x, 4x).
"""

from __future__ import annotations

from typing import Any

from backtesting.execution.fill_models import RealisticFillModel
from backtesting.execution.slippage_models.models import (
    FixedBpsSlippage,
    SquareRootImpactSlippage,
)
from backtesting.execution.transaction_costs.costs import (
    MakerTakerFee,
    PercentCommission,
)


def build_fill_model(
    cost_model: dict[str, Any],
    is_crypto: bool = False,
    cost_multiplier: float = 1.0,
) -> RealisticFillModel:
    """Build a RealisticFillModel matching the pre-registered protocol specification.

    Args:
        cost_model: The protocol's cost_model dictionary.
        is_crypto: True if the asset universe is cryptocurrency.
        cost_multiplier: Scalar multiplier applied to spread, fees, and impact.
            Setting to 0.0 results in a zero-cost execution model.

    Returns:
        RealisticFillModel configured with the specified parameters.
    """
    mult = max(0.0, float(cost_multiplier))

    if mult == 0.0:
        return RealisticFillModel(
            slippage=FixedBpsSlippage(bps=0.0),
            cost_model=PercentCommission(pct=0.0),
            default_spread_frac=0.0,
        )

    if is_crypto:
        crypto_cfg = cost_model.get("crypto", {})
        half_spread_bps = float(crypto_cfg.get("half_spread_bps", 1.0)) * mult
        taker_fee_bps = float(crypto_cfg.get("taker_fee_bps", 10.0)) * mult

        # Full spread is 2 * half_spread_bps in fractional terms
        default_spread_frac = (2.0 * half_spread_bps) / 10_000.0
        slippage = FixedBpsSlippage(bps=half_spread_bps)
        fee_model = MakerTakerFee(maker_bps=0.0, taker_bps=taker_fee_bps)

        return RealisticFillModel(
            slippage=slippage,
            cost_model=fee_model,
            default_spread_frac=default_spread_frac,
        )

    # Equities configuration
    eq_cfg = cost_model.get("equities", {})
    half_spread_bps = float(eq_cfg.get("half_spread_bps", 1.0)) * mult
    commission_bps = float(eq_cfg.get("commission", 0.0)) * mult
    sqrt_coef = float(eq_cfg.get("sqrt_impact_adv_coef", 0.1)) * mult

    default_spread_frac = (2.0 * half_spread_bps) / 10_000.0
    slippage = SquareRootImpactSlippage(c=sqrt_coef)
    fee_model = PercentCommission(pct=commission_bps / 10_000.0)

    return RealisticFillModel(
        slippage=slippage,
        cost_model=fee_model,
        default_spread_frac=default_spread_frac,
    )
