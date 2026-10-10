from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct: float = 0.20
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.05
    max_drawdown_halt: float = 0.25
    max_gross_exposure: float = 1.0
    kelly_fraction: float = 0.5
    kelly_cap: float = 1.0
    target_annual_vol: float = 0.15
    cvar_alpha: float = 0.99
    cvar_budget_pct: float = 0.03
    vol_scale_min: float = 0.25
    vol_scale_max: float = 2.0

    def __post_init__(self) -> None:
        # All parameters must be strictly positive
        fields_to_check = {
            "max_position_pct": self.max_position_pct,
            "max_leverage": self.max_leverage,
            "stop_loss_pct": self.stop_loss_pct,
            "max_drawdown_halt": self.max_drawdown_halt,
            "max_gross_exposure": self.max_gross_exposure,
            "kelly_fraction": self.kelly_fraction,
            "kelly_cap": self.kelly_cap,
            "target_annual_vol": self.target_annual_vol,
            "cvar_alpha": self.cvar_alpha,
            "cvar_budget_pct": self.cvar_budget_pct,
            "vol_scale_min": self.vol_scale_min,
            "vol_scale_max": self.vol_scale_max,
        }

        for name, val in fields_to_check.items():
            if val <= 0:
                raise ValueError(f"Risk parameter '{name}' must be positive, got {val}")

        # Fractions must be in (0, 1]
        fraction_fields = {
            "max_position_pct": self.max_position_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_drawdown_halt": self.max_drawdown_halt,
            "kelly_fraction": self.kelly_fraction,
            "kelly_cap": self.kelly_cap,
            "cvar_alpha": self.cvar_alpha,
            "cvar_budget_pct": self.cvar_budget_pct,
        }

        for name, val in fraction_fields.items():
            if not (0.0 < val <= 1.0):
                raise ValueError(f"Risk parameter '{name}' must be in (0, 1], got {val}")

        # Volatility scale bounds check
        if self.vol_scale_min >= self.vol_scale_max:
            raise ValueError(
                f"vol_scale_min ({self.vol_scale_min}) must be strictly less than "
                f"vol_scale_max ({self.vol_scale_max})"
            )


def check_order_allowed(
    limits: RiskLimits,
    current_position_value: float,
    order_value: float,
    equity: float,
) -> tuple[bool, str]:
    """Validate whether an order is allowed given portfolio equity and risk limits.

    Returns:
        tuple[bool, str]: (is_allowed, reason)
    """
    if equity <= 0:
        return False, f"Portfolio equity must be positive, got {equity}"

    new_position_value = current_position_value + order_value
    position_pct = abs(new_position_value) / equity

    if position_pct > limits.max_position_pct:
        return (
            False,
            f"Position size {position_pct:.2%} exceeds max limit of {limits.max_position_pct:.2%}",
        )

    if position_pct > limits.max_leverage:
        return (
            False,
            f"Leverage {position_pct:.2f} exceeds max leverage of {limits.max_leverage:.2f}",
        )

    return True, "Order within risk limits"


def vol_scaling_factor(limits: RiskLimits, current_vol: float) -> float:
    """Calculate the volatility scaling factor clipped to [vol_scale_min, vol_scale_max].

    Args:
        limits: RiskLimits configuration.
        current_vol: Current annualized volatility (must be > 0).

    Returns:
        float: Clipped volatility scaling factor.
    """
    if current_vol <= 0:
        raise ValueError(f"Current volatility must be positive, got {current_vol}")

    raw_factor = limits.target_annual_vol / current_vol
    return max(limits.vol_scale_min, min(raw_factor, limits.vol_scale_max))
