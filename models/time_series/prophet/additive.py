"""
Additive trend and seasonal decomposition forecasting model.

Note: inspired by Prophet, not a reimplementation.

This module implements a regularized piecewise-linear trend with configurable changepoints
combined with Fourier harmonic series for periodic seasonality (weekly, yearly, and custom cycles).
Estimation is performed via ridge least squares with analytic closed-form parameter recovery,
strictly avoiding third-party Stan/Prophet dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class AdditiveTrendSeasonal:
    """Additive piecewise-linear trend with Fourier seasonality.

    Inspired by Prophet, not a reimplementation.

    Parameters
    ----------
    n_changepoints : int, default 5
        Number of potential rate change points placed along the historical trend.
    changepoint_range : float, default 0.8
        Proportion of historical history over which changepoints are placed.
    weekly_order : int, default 3
        Fourier order for weekly (period=7) seasonality. 0 disables weekly seasonality.
    yearly_order : int, default 5
        Fourier order for yearly (period=365.25) seasonality. 0 disables yearly seasonality.
    seasonal_orders : dict, optional
        Custom seasonality components mapping name -> (period, order).
    ridge_alpha : float, default 1e-4
        L2 regularization penalty for ridge regression stability.
    """

    def __init__(
        self,
        n_changepoints: int = 5,
        changepoint_range: float = 0.8,
        weekly_order: int = 3,
        yearly_order: int = 5,
        seasonal_orders: Optional[Dict[str, Tuple[float, int]]] = None,
        ridge_alpha: float = 1e-4,
    ) -> None:
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.weekly_order = weekly_order
        self.yearly_order = yearly_order
        self.seasonal_orders = seasonal_orders
        self.ridge_alpha = ridge_alpha

        # Fitted parameters
        self.beta_: Optional[np.ndarray] = None
        self.changepoints_: Optional[np.ndarray] = None
        self.seasons_config_: Dict[str, Tuple[float, int]] = {}
        self.t_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.dt_step_: float = 1.0
        self.is_datetime_: bool = False
        self.last_timestamp_: Optional[pd.Timestamp] = None
        self.last_t_: float = 0.0
        self.r2_: float = 0.0

    def _extract_t(
        self,
        series: Union[pd.Series, np.ndarray],
        dates: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Optional[pd.Timestamp], float]:
        """Converts inputs into continuous time array t and target array y."""
        if isinstance(series, pd.Series):
            y = series.values.astype(float)
            idx = series.index
        else:
            y = np.asarray(series, dtype=float)
            idx = None

        is_dt = False
        last_ts = None
        dt_step = 1.0

        if dates is not None:
            dt_series = pd.to_datetime(dates)
            t = (dt_series - dt_series[0]).total_seconds().values / 86400.0
            is_dt = True
            last_ts = dt_series[-1]
            if len(dt_series) > 1:
                dt_step = np.median(np.diff(t))
        elif idx is not None and isinstance(idx, pd.DatetimeIndex):
            t = (idx - idx[0]).total_seconds().values / 86400.0
            is_dt = True
            last_ts = idx[-1]
            if len(idx) > 1:
                dt_step = float(np.median(np.diff(t)))
        else:
            t = np.arange(len(y), dtype=float)
            is_dt = False
            last_ts = None
            dt_step = 1.0

        return t, y, is_dt, last_ts, dt_step

    def _build_design_matrix(
        self,
        t_vec: np.ndarray,
        changepoints: np.ndarray,
        seasons: Dict[str, Tuple[float, int]],
    ) -> np.ndarray:
        """Constructs piecewise-linear and Fourier seasonal basis columns."""
        cols: List[np.ndarray] = []

        # 1. Intercept & Base linear trend
        cols.append(np.ones_like(t_vec, dtype=float))
        cols.append(t_vec.astype(float))

        # 2. Piecewise changepoints (hinge / ramp functions)
        for cp in changepoints:
            cols.append(np.maximum(0.0, t_vec - cp))

        # 3. Fourier seasonality series
        for _, (period, order) in seasons.items():
            if period <= 0 or order <= 0:
                continue
            for m in range(1, order + 1):
                omega = 2.0 * np.pi * m / period
                cols.append(np.cos(omega * t_vec))
                cols.append(np.sin(omega * t_vec))

        return np.column_stack(cols)

    def fit(
        self,
        series: Union[pd.Series, np.ndarray],
        dates: Optional[Any] = None,
    ) -> AdditiveTrendSeasonal:
        """Fits piecewise trend and seasonality parameters via regularized ridge regression.

        Parameters
        ----------
        series : pd.Series or np.ndarray
            Historical target observations.
        dates : array-like, optional
            Explicit timestamp sequence corresponding to series observations.

        Returns
        -------
        AdditiveTrendSeasonal
            Fitted estimator instance.
        """
        t, y, is_dt, last_ts, dt_step = self._extract_t(series, dates)
        self.t_train_ = t
        self.y_train_ = y
        self.is_datetime_ = is_dt
        self.last_timestamp_ = last_ts
        self.last_t_ = float(t[-1])
        self.dt_step_ = max(dt_step, 1e-6)

        # Configure changepoints
        if self.n_changepoints > 0 and len(t) > 2:
            cp_limit = t[0] + self.changepoint_range * (t[-1] - t[0])
            self.changepoints_ = np.linspace(t[0], cp_limit, self.n_changepoints + 2)[1:-1]
        else:
            self.changepoints_ = np.array([], dtype=float)

        # Configure seasonal components
        self.seasons_config_ = {}
        if self.seasonal_orders is not None:
            self.seasons_config_.update(self.seasonal_orders)
        else:
            if self.weekly_order > 0:
                self.seasons_config_["weekly"] = (7.0, self.weekly_order)
            if self.yearly_order > 0:
                self.seasons_config_["yearly"] = (365.25, self.yearly_order)

        # Build feature design matrix
        X = self._build_design_matrix(t, self.changepoints_, self.seasons_config_)

        # Ridge regularized regression (intercept unpenalized)
        p = X.shape[1]
        reg_diag = np.full(p, self.ridge_alpha)
        reg_diag[0] = 0.0  # Do not shrink the global intercept

        xtx = X.T @ X + np.diag(reg_diag)
        xty = X.T @ y
        self.beta_ = np.linalg.solve(xtx, xty)

        # Compute in-sample fit diagnostics
        y_pred = X @ self.beta_
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        self.r2_ = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return self

    def predict(
        self,
        steps: Optional[int] = None,
        future_dates: Optional[Any] = None,
        future_t: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, pd.Series]:
        """Generates predictions for in-sample or out-of-sample forward horizons.

        Parameters
        ----------
        steps : int, optional
            Number of forward steps to project. If None and no future dates are given,
            returns in-sample fitted values.
        future_dates : array-like, optional
            Specific future timestamps for evaluation.
        future_t : np.ndarray, optional
            Specific future time coordinates.

        Returns
        -------
        np.ndarray or pd.Series
            Predicted values.
        """
        if self.beta_ is None or self.changepoints_ is None:
            raise ValueError("Model must be fit before calling predict().")

        if steps is None and future_dates is None and future_t is None:
            # In-sample evaluation
            X_in = self._build_design_matrix(
                self.t_train_, self.changepoints_, self.seasons_config_
            )
            return X_in @ self.beta_

        if future_t is not None:
            t_eval = np.asarray(future_t, dtype=float)
            X_eval = self._build_design_matrix(
                t_eval, self.changepoints_, self.seasons_config_
            )
            return X_eval @ self.beta_

        if future_dates is not None:
            dt_series = pd.to_datetime(future_dates)
            t_eval = (dt_series - self.last_timestamp_).total_seconds().values / 86400.0 + self.last_t_
            X_eval = self._build_design_matrix(
                t_eval, self.changepoints_, self.seasons_config_
            )
            pred = X_eval @ self.beta_
            return pd.Series(pred, index=dt_series, name="forecast")

        # Steps provided
        num_steps = int(steps)
        t_eval = self.last_t_ + np.arange(1, num_steps + 1, dtype=float) * self.dt_step_
        X_eval = self._build_design_matrix(
            t_eval, self.changepoints_, self.seasons_config_
        )
        pred = X_eval @ self.beta_

        if self.is_datetime_ and self.last_timestamp_ is not None:
            forward_index = pd.date_range(
                start=self.last_timestamp_ + pd.Timedelta(days=self.dt_step_),
                periods=num_steps,
                freq=f"{self.dt_step_}D",
            )
            return pd.Series(pred, index=forward_index, name="forecast")

        return pred

    def forecast(self, steps: int = 1) -> Union[np.ndarray, pd.Series]:
        """Convenience method for generating multi-step ahead forecasts."""
        return self.predict(steps=steps)
