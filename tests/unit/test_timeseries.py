"""
Unit tests for time-series forecasting, GARCH volatility, AdditiveTrendSeasonal, UCM, and evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.time_series.arima import auto_arima, forecast, ljung_box_diagnosis
from models.time_series.evaluation import (
    diebold_mariano,
    directional_accuracy,
    mae,
    rmse,
    rolling_origin_evaluation,
)
from models.time_series.garch import (
    fit_egarch,
    fit_garch,
    forecast_vol,
    regime_probability,
    rolling_vol_forecast,
)
from models.time_series.prophet import AdditiveTrendSeasonal
from models.time_series.state_space import fit_local_linear_trend


class TestARIMAForecasting:
    """Tests for auto_arima order selection, Ljung-Box diagnosis, and forecasting."""

    def test_arima_ar1_simulation_and_random_walk_d_selection(self):
        # 1. Simulate AR(1) with phi=0.7 (n=1000, fixed seed)
        np.random.seed(42)
        n = 1000
        e = np.random.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + e[t]

        s_ar1 = pd.Series(y)
        res = auto_arima(s_ar1, max_p=2, max_q=1, max_d=1)

        # auto_arima selects p >= 1
        assert res.p >= 1, f"Expected p >= 1, got {res.p}"
        assert res.order[0] >= 1

        # Estimated AR coefficient within 0.1 of 0.7
        assert len(res.arparams) >= 1
        est_phi = res.arparams[0]
        assert abs(est_phi - 0.7) < 0.1, f"Estimated phi {est_phi} not within 0.1 of 0.7"

        # Ljung-Box p > 0.05 on residuals
        lb_df = ljung_box_diagnosis(res.resid, lags=10)
        assert lb_df["lb_pvalue"].min() > 0.05, f"Ljung-Box p-value rejected white noise: {lb_df}"

        # 2. Random walk simulation gets d == 1
        np.random.seed(123)
        rw = pd.Series(np.cumsum(np.random.randn(n)))
        rw_res = auto_arima(rw, max_p=1, max_q=1, max_d=2)
        assert rw_res.d == 1, f"Expected d==1 for random walk, got {rw_res.d}"

        # 3. Forecast output structure and confidence intervals
        fc = res.forecast(steps=5)
        assert len(fc.forecast) == 5
        assert len(fc.conf_int) == 5
        assert (fc.conf_int.iloc[:, 0] <= fc.conf_int.iloc[:, 1]).all()
        # Test tuple unpacking
        mean_fc, ci_df = fc
        assert len(mean_fc) == 5
        assert isinstance(ci_df, pd.DataFrame)


class TestGARCHVolatilityModels:
    """Tests for GARCH parameter estimation, rolling forecasting, and regime gating."""

    def test_garch_parameter_recovery_and_properties(self):
        # Simulate GARCH(1,1) with known parameters: omega=0.05, alpha=0.10, beta=0.85
        np.random.seed(42)
        n = 5000
        omega = 0.05
        alpha = 0.10
        beta = 0.85
        eps = np.random.randn(n)
        sig2 = np.zeros(n)
        r = np.zeros(n)
        sig2[0] = omega / (1.0 - alpha - beta)

        for t in range(1, n):
            sig2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sig2[t - 1]
            r[t] = np.sqrt(sig2[t]) * eps[t]

        r_ser = pd.Series(r)
        res = fit_garch(r_ser, p=1, q=1, dist="normal")

        est_alpha = float(res.params["alpha[1]"])
        est_beta = float(res.params["beta[1]"])
        est_sum = est_alpha + est_beta

        # alpha + beta within 0.1 of truth (0.95)
        assert abs(est_sum - (alpha + beta)) < 0.1, f"alpha+beta {est_sum} not within 0.1 of 0.95"

        # Multi-step and 1-step forecast_vol
        fc_1 = forecast_vol(res, horizon=1, annualize=False)
        assert isinstance(fc_1, float)
        assert fc_1 > 0.0

        fc_5 = forecast_vol(res, horizon=5, annualize=True)
        assert isinstance(fc_5, pd.Series)
        assert len(fc_5) == 5
        assert (fc_5 > 0.0).all()

        # EGARCH fit executes cleanly
        egarch_res = fit_egarch(r_ser.iloc[:500], p=1, q=1, o=1, dist="normal")
        assert egarch_res is not None

    def test_garch_truncation_invariance_and_regime_probability(self):
        np.random.seed(77)
        r = pd.Series(np.random.randn(300) * 0.01)

        # rolling_vol_forecast truncation invariance
        rv200 = rolling_vol_forecast(r.iloc[:200], window=50, refit_every=25)
        rv300 = rolling_vol_forecast(r.iloc[:300], window=50, refit_every=25)

        # First 200 rows must be identical
        np.testing.assert_allclose(
            rv200.values,
            rv300.iloc[:200].values,
            equal_nan=True,
            err_msg="rolling_vol_forecast failed truncation invariance!",
        )

        # regime_probability in [0, 1] and truncation invariant
        rp200 = regime_probability(rv200, lookback=40)
        rp300 = regime_probability(rv300, lookback=40)

        valid_rp = rp300.dropna()
        assert (valid_rp >= 0.0).all(), "Regime probability contains values < 0.0"
        assert (valid_rp <= 1.0).all(), "Regime probability contains values > 1.0"

        np.testing.assert_allclose(
            rp200.values,
            rp300.iloc[:200].values,
            equal_nan=True,
            err_msg="regime_probability failed truncation invariance!",
        )


class TestAdditiveTrendSeasonal:
    """Tests for piecewise-linear trend + Fourier seasonality model."""

    def test_additive_recovers_planted_trend_and_weekly_seasonality(self):
        np.random.seed(42)
        n = 120
        t = np.arange(n, dtype=float)
        trend = 0.4 * t + 5.0
        seasonal = 4.0 * np.sin(2.0 * np.pi * t / 7.0)
        noise = np.random.randn(n) * 0.15
        y = trend + seasonal + noise

        train_n = 100
        y_train = pd.Series(y[:train_n])
        y_test = y[train_n:]

        model = AdditiveTrendSeasonal(
            n_changepoints=4,
            weekly_order=3,
            yearly_order=0,
            ridge_alpha=1e-4,
        )
        model.fit(y_train)

        # In-sample R^2 > 0.95
        assert model.r2_ > 0.95, f"Expected in-sample R^2 > 0.95, got {model.r2_}"

        # Holdout RMSE below naive mean baseline
        fc = model.forecast(steps=n - train_n)
        model_rmse = rmse(y_test, fc)
        naive_rmse = rmse(y_test, np.full_like(y_test, y_train.mean()))

        assert model_rmse < naive_rmse, (
            f"Model holdout RMSE ({model_rmse:.4f}) not lower than naive RMSE ({naive_rmse:.4f})"
        )
        assert model_rmse < 0.5, f"Model RMSE unexpectedly high: {model_rmse}"


class TestStateSpaceUCM:
    """Tests for state-space unobserved components modeling."""

    def test_ucm_finite_forecasts_and_dimensions(self):
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.randn(100)) + 25.0)

        ucm = fit_local_linear_trend(s, seasonal=None)

        # Smoothed components
        assert len(ucm.smoothed_level) == 100
        assert len(ucm.smoothed_trend) == 100
        assert np.all(np.isfinite(ucm.smoothed_level.values))
        assert np.all(np.isfinite(ucm.smoothed_trend.values))

        # Finite forecasts of correct length
        fc = ucm.forecast(steps=12)
        assert len(fc) == 12
        assert np.all(np.isfinite(fc.values)), "UCM forecasts contain non-finite values"


class TestEvaluationAndDieboldMariano:
    """Tests for rolling origin validation and Diebold-Mariano testing."""

    def test_metrics_closed_form_exact_values(self):
        y_true = np.array([1.0, -2.0, 3.0, 4.0])
        y_pred = np.array([2.0, -1.0, 3.0, 1.0])

        # Hand-computed:
        # Errors: [-1, -1, 0, 3] -> Squared: [1, 1, 0, 9] -> Mean: 2.75 -> RMSE: sqrt(2.75)
        expected_rmse = np.sqrt(2.75)
        # Abs errors: [1, 1, 0, 3] -> Mean: 5/4 = 1.25
        expected_mae = 1.25
        # Signs: true [+1, -1, +1, +1], pred [+1, -1, +1, +1] -> 4/4 = 1.0
        expected_da = 1.0

        assert abs(rmse(y_true, y_pred) - expected_rmse) < 1e-12
        assert abs(mae(y_true, y_pred) - expected_mae) < 1e-12
        assert abs(directional_accuracy(y_true, y_pred) - expected_da) < 1e-12

    def test_rolling_origin_no_lookahead_guarantee(self):
        np.random.seed(42)
        s_base = pd.Series(np.cumsum(np.random.randn(120)))

        def trailing_mean_model(train_slice: pd.Series):
            val = float(train_slice.mean())
            return np.full(3, val)

        df1 = rolling_origin_evaluation(s_base, trailing_mean_model, initial=50, horizon=3, step=5)

        # Perturb future data from index 80 onward
        s_perturbed = s_base.copy()
        s_perturbed.iloc[80:] += 9999.0

        df2 = rolling_origin_evaluation(s_perturbed, trailing_mean_model, initial=50, horizon=3, step=5)

        # Forecasts made at origins <= 80 only observed data up to 80, so must be identical
        sub1 = df1[df1["origin"] <= 80]["forecast"].values
        sub2 = df2[df2["origin"] <= 80]["forecast"].values

        np.testing.assert_allclose(
            sub1,
            sub2,
            err_msg="rolling_origin_evaluation leaked future perturbed data to earlier origins!",
        )

    def test_diebold_mariano_properties_and_significance(self):
        # 1. Two identical error series return p ~ 1.0 and stat 0.0
        e_same = np.array([0.5, -0.2, 0.8, -0.4])
        dm_ident = diebold_mariano(e_same, e_same, h=1)
        assert dm_ident.statistic == 0.0
        assert dm_ident.p_value == 1.0

        # Hand-computed 3-point example:
        # e1 = [1, 2, 1], e2 = [2, 3, 2], h=1
        # d = [-3, -5, -3], mean = -11/3, gamma0 = 8/9, V = (8/9)/3 = 8/27, DM = (-11/3)/sqrt(8/27) = -6.73609
        dm_hand = diebold_mariano([1.0, 2.0, 1.0], [2.0, 3.0, 2.0], h=1)
        expected_stat = (-11.0 / 3.0) / np.sqrt((8.0 / 9.0) / 3.0)
        assert abs(dm_hand.statistic - expected_stat) < 1e-6
        assert dm_hand.p_value < 1e-4

        # 2. Simulated AR(1) dynamics significantly beats naive mean forecast
        np.random.seed(42)
        n = 1000
        y = np.zeros(n)
        eps = np.random.randn(n)
        for t in range(1, n):
            y[t] = 0.75 * y[t - 1] + eps[t]

        actual = y[1:]
        pred_ar = 0.75 * y[:-1]
        pred_mean = np.full(n - 1, np.mean(y[:100]))

        e_ar = actual - pred_ar
        e_mean = actual - pred_mean

        dm_res = diebold_mariano(e_ar, e_mean, h=1)
        assert dm_res.p_value < 0.05, f"Expected p < 0.05, got {dm_res.p_value}"
        assert dm_res.statistic < 0.0, f"Expected negative DM statistic (AR better), got {dm_res.statistic}"
