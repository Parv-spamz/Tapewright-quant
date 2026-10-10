"""
ARIMA modeling module with automated order selection, diagnostics, and out-of-sample forecasting.

Provides:
- auto_arima: Iterative stationarity testing (ADF + KPSS) for integration order d,
  followed by AIC grid search over (p, q) with non-convergence safeguards.
- ljung_box_diagnosis: Ljung-Box portmanteau test for residual autocorrelation.
- forecast: Out-of-sample point forecasts with confidence intervals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class ForecastResult:
    """Container for ARIMA point forecasts and confidence intervals."""

    predicted_mean: pd.Series
    conf_int: pd.DataFrame

    @property
    def forecast(self) -> pd.Series:
        return self.predicted_mean

    @property
    def lower(self) -> pd.Series:
        return self.conf_int.iloc[:, 0]

    @property
    def upper(self) -> pd.Series:
        return self.conf_int.iloc[:, 1]

    def __iter__(self):
        """Allows unpacking as (forecast, conf_int)."""
        return iter((self.predicted_mean, self.conf_int))

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "forecast": self.predicted_mean,
                "lower_ci": self.lower,
                "upper_ci": self.upper,
            }
        )
        return df


class AutoARIMAResults:
    """Wrapper around fitted statsmodels ARIMAResults providing rich metadata."""

    def __init__(
        self,
        fitted_model: ARIMAResults,
        order: Tuple[int, int, int],
        aic: float,
        bic: float,
        d_selected: int,
    ) -> None:
        self.model = fitted_model
        self.order = order
        self.p, self.d, self.q = order
        self.aic = aic
        self.bic = bic
        self.d_selected = d_selected

    @property
    def params(self) -> pd.Series:
        return self.model.params

    @property
    def arparams(self) -> np.ndarray:
        return self.model.arparams

    @property
    def maparams(self) -> np.ndarray:
        return self.model.maparams

    @property
    def resid(self) -> pd.Series:
        return self.model.resid

    def forecast(self, steps: int = 1, alpha: float = 0.05) -> ForecastResult:
        return forecast(self.model, steps=steps, alpha=alpha)

    def get_forecast(self, steps: int = 1, **kwargs) -> Any:
        return self.model.get_forecast(steps=steps, **kwargs)

    def summary(self) -> Any:
        return self.model.summary()

    def __getattr__(self, item: str) -> Any:
        return getattr(self.model, item)


def auto_arima(
    series: Union[pd.Series, np.ndarray],
    max_p: int = 3,
    max_q: int = 3,
    max_d: int = 2,
    alpha_adf: float = 0.05,
    alpha_kpss: float = 0.05,
) -> AutoARIMAResults:
    """Automatically selects integration order d and (p, q) parameters via AIC minimization.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Historical univariate time-series observations.
    max_p : int
        Maximum autoregressive lag order to evaluate.
    max_q : int
        Maximum moving-average lag order to evaluate.
    max_d : int, default 2
        Maximum differencing order. Stationarity is determined via repeated
        Augmented Dickey-Fuller (ADF) and KPSS unit-root tests.
    alpha_adf : float, default 0.05
        Significance threshold for ADF test (p-value < alpha rejects unit root).
    alpha_kpss : float, default 0.05
        Significance threshold for KPSS test (p-value > alpha fails to reject stationarity).

    Returns
    -------
    AutoARIMAResults
        Fitted model corresponding to the minimum AIC specification.
    """
    if isinstance(series, np.ndarray):
        s = pd.Series(series)
    else:
        s = series.dropna()

    # Step 1: Select differencing order d by repeated ADF (+ KPSS) testing
    d = 0
    diff_s = s.copy()

    while d < max_d:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(diff_s.dropna(), autolag="AIC")
            try:
                kpss_stat, kpss_pvalue, _, _ = kpss(
                    diff_s.dropna(), regression="c", nlags="auto"
                )
            except Exception:
                kpss_pvalue = 1.0

        is_stationary = adf_pvalue < alpha_adf
        if is_stationary:
            break

        d += 1
        diff_s = s.diff(d).dropna()

    # Step 2: Grid search over (p, q) combinations to minimize AIC
    best_aic = float("inf")
    best_bic = float("inf")
    best_order: Tuple[int, int, int] = (0, d, 0)
    best_fit: Union[ARIMAResults, None] = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(s, order=(p, d, q))
                    fitted = model.fit()

                if not np.isfinite(fitted.aic):
                    continue

                if fitted.aic < best_aic:
                    best_aic = float(fitted.aic)
                    best_bic = float(fitted.bic)
                    best_order = (p, d, q)
                    best_fit = fitted
            except Exception:
                # Catch non-convergence or invertibility matrix failure and skip candidate
                continue

    if best_fit is None:
        # Fallback to (0, d, 0) random walk / drift
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_fit = ARIMA(s, order=(0, d, 0)).fit()
            best_order = (0, d, 0)
            best_aic = float(best_fit.aic)
            best_bic = float(best_fit.bic)

    return AutoARIMAResults(
        fitted_model=best_fit,
        order=best_order,
        aic=best_aic,
        bic=best_bic,
        d_selected=d,
    )


def ljung_box_diagnosis(
    resid: Union[pd.Series, np.ndarray],
    lags: Union[int, list[int]] = 10,
) -> pd.DataFrame:
    """Performs the Ljung-Box portmanteau diagnostic test on model residuals.

    Tests the null hypothesis that residuals are white noise (no autocorrelation).
    p-value > 0.05 indicates failure to reject white noise (residuals are clean).

    Parameters
    ----------
    resid : pd.Series or np.ndarray
        Residual series from a fitted ARIMA model.
    lags : int or list of int, default 10
        Number of autocorrelation lags to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'lb_stat' and 'lb_pvalue'.
    """
    if isinstance(resid, np.ndarray):
        r = pd.Series(resid)
    else:
        r = resid

    r_clean = r.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_lb = acorr_ljungbox(r_clean, lags=lags, return_df=True)

    # Provide alias column 'pvalue' for convenient single-column access
    df_lb["pvalue"] = df_lb["lb_pvalue"]
    return df_lb


def forecast(
    model: Union[ARIMAResults, AutoARIMAResults],
    steps: int = 1,
    alpha: float = 0.05,
) -> ForecastResult:
    """Generates out-of-sample point forecasts and symmetric confidence intervals.

    Parameters
    ----------
    model : ARIMAResults or AutoARIMAResults
        Fitted ARIMA model object.
    steps : int, default 1
        Forecast horizon (number of out-of-sample steps).
    alpha : float, default 0.05
        Significance level for confidence interval (0.05 yields 95% intervals).

    Returns
    -------
    ForecastResult
        Dataclass containing .predicted_mean, .conf_int, unpackable as (mean, conf_int).
    """
    fitted = model.model if isinstance(model, AutoARIMAResults) else model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = fitted.get_forecast(steps=steps)
        pred_mean = fc.predicted_mean
        conf_int = fc.conf_int(alpha=alpha)

    if not isinstance(pred_mean, pd.Series):
        pred_mean = pd.Series(pred_mean)
    if not isinstance(conf_int, pd.DataFrame):
        conf_int = pd.DataFrame(conf_int, columns=["lower_ci", "upper_ci"])

    return ForecastResult(predicted_mean=pred_mean, conf_int=conf_int)
