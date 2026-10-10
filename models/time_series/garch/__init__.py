"""
GARCH and volatility time-series models package.
"""

from models.time_series.garch.garch import (
    fit_egarch,
    fit_garch,
    forecast_vol,
    regime_probability,
    rolling_vol_forecast,
)

__all__ = [
    "fit_garch",
    "fit_egarch",
    "forecast_vol",
    "rolling_vol_forecast",
    "regime_probability",
]
