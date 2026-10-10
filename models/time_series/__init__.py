"""
Statistical and econometric time-series models package for Tapewright-Quant.
"""

from models.time_series.arima.arima import (
    AutoARIMAResults,
    ForecastResult,
    auto_arima,
    forecast,
    ljung_box_diagnosis,
)
from models.time_series.evaluation import (
    DMResult,
    diebold_mariano,
    directional_accuracy,
    mae,
    rmse,
    rolling_origin_evaluation,
)
from models.time_series.garch.garch import (
    fit_egarch,
    fit_garch,
    forecast_vol,
    regime_probability,
    rolling_vol_forecast,
)
from models.time_series.prophet.additive import AdditiveTrendSeasonal
from models.time_series.state_space.ucm import UCMResults, fit_local_linear_trend

__all__ = [
    # ARIMA
    "auto_arima",
    "ljung_box_diagnosis",
    "forecast",
    "AutoARIMAResults",
    "ForecastResult",
    # GARCH
    "fit_garch",
    "fit_egarch",
    "forecast_vol",
    "rolling_vol_forecast",
    "regime_probability",
    # Additive
    "AdditiveTrendSeasonal",
    # UCM
    "fit_local_linear_trend",
    "UCMResults",
    # Evaluation
    "rolling_origin_evaluation",
    "rmse",
    "mae",
    "directional_accuracy",
    "diebold_mariano",
    "DMResult",
]
