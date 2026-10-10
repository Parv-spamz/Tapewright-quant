"""
State-space Unobserved Components Model (UCM) for structural time-series decomposition.

Provides:
- fit_local_linear_trend: Estimates local linear trend models with optional seasonal cycles,
  extracting smoothed level, slope/trend components, and finite multi-step forecasts.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import (
    UnobservedComponents,
    UnobservedComponentsResults,
)


class UCMResults:
    """Wrapper around statsmodels UnobservedComponentsResults providing convenient component extraction."""

    def __init__(
        self,
        results: UnobservedComponentsResults,
        series: Union[pd.Series, np.ndarray],
    ) -> None:
        self.results = results
        index = series.index if isinstance(series, pd.Series) else None

        # Extract smoothed state components
        try:
            lvl = results.level["smoothed"]
            self.smoothed_level = pd.Series(lvl, index=index, name="smoothed_level")
        except Exception:
            self.smoothed_level = pd.Series(results.smoothed_state[0], index=index, name="smoothed_level")

        try:
            tr = results.trend["smoothed"]
            self.smoothed_trend = pd.Series(tr, index=index, name="smoothed_trend")
        except Exception:
            if results.smoothed_state.shape[0] > 1:
                self.smoothed_trend = pd.Series(results.smoothed_state[1], index=index, name="smoothed_trend")
            else:
                self.smoothed_trend = pd.Series(0.0, index=index, name="smoothed_trend")

    def forecast(self, steps: int = 1) -> pd.Series:
        """Generates out-of-sample forward forecasts of length `steps`.

        Parameters
        ----------
        steps : int, default 1
            Forecast horizon length.

        Returns
        -------
        pd.Series
            Predicted mean series of length `steps`. Guaranteed to be finite.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.results.get_forecast(steps=steps)
            pred = fc.predicted_mean
            if not isinstance(pred, pd.Series):
                pred = pd.Series(pred)
        return pred

    def get_forecast(self, steps: int = 1, **kwargs) -> Any:
        return self.results.get_forecast(steps=steps, **kwargs)

    def summary(self) -> Any:
        return self.results.summary()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.results, name)


def fit_local_linear_trend(
    series: Union[pd.Series, np.ndarray],
    seasonal: Optional[int] = None,
) -> UCMResults:
    """Fits a state-space local linear trend model with optional harmonic seasonality.

    Decomposes the series into:
        y_t = mu_t + gamma_t + eps_t
        mu_{t+1} = mu_t + beta_t + eta_t    (stochastic level)
        beta_{t+1} = beta_t + zeta_t        (stochastic slope)

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Historical time-series observations.
    seasonal : int, optional
        Length of seasonal cycle (e.g. 7 for weekly, 12 for monthly). None for no seasonality.

    Returns
    -------
    UCMResults
        Wrapper containing smoothed level, smoothed trend, and forecast method.
    """
    if isinstance(series, np.ndarray):
        s = pd.Series(series)
    else:
        s = series.dropna()

    model = UnobservedComponents(
        s,
        level="local linear trend",
        seasonal=seasonal,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.fit(disp=False)

    return UCMResults(results=results, series=s)
