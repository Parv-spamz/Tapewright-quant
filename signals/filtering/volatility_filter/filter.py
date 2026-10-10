from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from arch import arch_model

from backtesting.engine.event_loop.events import SignalEvent


def garch_regime_filter(
    returns: pd.Series | np.ndarray,
    threshold_quantile: float = 0.90,
    refit_every: int = 21,
    min_obs: int = 250,
) -> pd.Series:
    """Expanding-window GARCH(1,1) volatility regime filter using strictly past data.

    Forecasts one-step-ahead conditional volatility for each bar from data available up to t-1,
    and blocks trading (allowed=False) when the forecast exceeds an expanding quantile of past
    conditional volatilities.

    Args:
        returns: Periodic simple or log returns series.
        threshold_quantile: Quantile cutoff of past conditional volatilities (e.g. 0.90).
        refit_every: Number of bars between full GARCH parameter re-estimations.
        min_obs: Minimum initial observations required before activating filter.

    Returns:
        pd.Series: Boolean series (True = allowed, False = blocked by high vol regime).
    """
    s = pd.Series(returns, dtype=float).copy()
    T = len(s)
    if T == 0:
        return pd.Series([], dtype=bool)

    allowed = pd.Series(True, index=s.index, dtype=bool)
    if T <= min_obs:
        return allowed

    vals = s.values

    # Initial fit on [0, min_obs)
    init_train = vals[:min_obs] * 100.0  # Scale by 100 for numerical stability in arch
    am = arch_model(init_train, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
    res = am.fit(disp="off")

    omega = float(res.params.get("omega", 0.01))
    alpha = float(res.params.get("alpha[1]", 0.05))
    beta = float(res.params.get("beta[1]", 0.90))

    hist_cond_vols = list(res.conditional_volatility / 100.0)
    last_vol_sq = (hist_cond_vols[-1] * 100.0) ** 2

    for t in range(min_obs, T):
        # Periodic expanding refit strictly using past data up to t
        if (t - min_obs) % refit_every == 0 and t > min_obs:
            try:
                am = arch_model(vals[:t] * 100.0, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
                res = am.fit(disp="off")
                omega = float(res.params.get("omega", omega))
                alpha = float(res.params.get("alpha[1]", alpha))
                beta = float(res.params.get("beta[1]", beta))
            except Exception:
                # Retain previous parameters if numerical optimization fails
                pass

        # One-step-ahead conditional variance using return from bar t - 1
        prev_r = vals[t - 1] * 100.0
        vol_sq = max(omega + alpha * (prev_r ** 2) + beta * last_vol_sq, 1e-8)
        last_vol_sq = vol_sq
        cond_vol = np.sqrt(vol_sq) / 100.0

        # Expanding threshold strictly from past conditional volatilities
        thresh = float(np.quantile(hist_cond_vols, threshold_quantile))
        allowed.iloc[t] = bool(cond_vol <= thresh)

        hist_cond_vols.append(cond_vol)

    return allowed


def simple_vol_filter(
    returns: pd.Series | np.ndarray,
    span: int = 20,
    threshold_quantile: float = 0.90,
    min_obs: int = 30,
) -> pd.Series:
    """Lightweight EWMA volatility filter for fast testing and screening.

    Args:
        returns: Periodic returns series.
        span: EWMA span parameter.
        threshold_quantile: Expanding quantile threshold.
        min_obs: Minimum observations before applying filter.

    Returns:
        pd.Series: Boolean series (True = allowed, False = blocked).
    """
    s = pd.Series(returns, dtype=float).copy()
    if len(s) == 0:
        return pd.Series([], dtype=bool)

    vol = s.ewm(span=span).std()
    past_vol = vol.shift(1)
    threshold = past_vol.expanding(min_periods=min_obs).quantile(threshold_quantile)

    # Allowed if in warmup (threshold is NaN) or vol <= threshold
    allowed = threshold.isna() | (vol <= threshold)
    return allowed.astype(bool)


def apply_filter(
    signals: pd.Series | list[SignalEvent],
    allowed: pd.Series | bool,
) -> pd.Series | list[SignalEvent]:
    """Apply boolean volatility regime filter to suppress trading signals during blocked regimes.

    Args:
        signals: Signal series in {-1, 0, 1} or list of SignalEvent instances.
        allowed: Boolean series aligned with signals, or scalar boolean flag.

    Returns:
        pd.Series | list[SignalEvent]: Filtered signals.
    """
    if isinstance(signals, pd.Series):
        if isinstance(allowed, pd.Series):
            aligned_allowed = allowed.reindex(signals.index, fill_value=True)
            return signals.where(aligned_allowed, 0).astype(int)
        elif isinstance(allowed, (bool, np.bool_)):
            return signals if allowed else pd.Series(0, index=signals.index, dtype=int)
        return signals

    elif isinstance(signals, list):
        if isinstance(allowed, (bool, np.bool_)):
            return signals if allowed else []
        elif isinstance(allowed, pd.Series):
            filtered: list[SignalEvent] = []
            for sig in signals:
                is_ok = bool(allowed.get(sig.timestamp, True))
                if is_ok:
                    filtered.append(sig)
            return filtered

    return signals
