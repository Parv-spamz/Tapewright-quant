"""Market microstructure spread and order book metrics.

Includes:
- Roll spread estimator (full-sample and rolling)
- Corwin-Schultz high-low spread estimator
- Order Book Imbalance (OBI) & Depth Imbalance
- Microprice (volume-weighted top of book midprice)
- Kyle's Lambda price impact estimator
- Amihud Illiquidity ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def roll_spread(
    close: pd.Series | np.ndarray,
    window: int | None = None,
) -> float | pd.Series:
    """Calculate Roll (1984) effective bid-ask spread estimator.

    Formula: S = 2 * sqrt(-cov(dP_t, dP_{t-1})), 0 if cov >= 0.

    Args:
        close: Price series.
        window: If specified, computes rolling Roll spread over window length.
                If None, computes full-sample scalar spread.

    Returns:
        float if window is None, else pd.Series of rolling spread values.
    """
    s = pd.Series(close) if not isinstance(close, pd.Series) else close
    dp = s.diff()

    if window is None:
        valid_dp = dp.dropna()
        if len(valid_dp) < 2:
            return 0.0
        cov_matrix = np.cov(valid_dp.iloc[1:], valid_dp.iloc[:-1])
        c = float(cov_matrix[0, 1])
        return 2.0 * np.sqrt(-c) if c < 0.0 else 0.0

    # Rolling variant
    dp_curr = dp
    dp_prev = dp.shift(1)
    rolling_cov = dp_curr.rolling(window=window).cov(dp_prev)
    res = np.where(rolling_cov < 0.0, 2.0 * np.sqrt((-rolling_cov).clip(lower=0.0)), 0.0)
    return pd.Series(res, index=s.index, name="roll_spread")


def corwin_schultz(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    window: int | None = None,
) -> pd.Series:
    """Calculate Corwin-Schultz (2012) High-Low spread estimator.

    Formulas:
        beta = sum_{j=0}^1 [ln(H_{t-j}/L_{t-j})]^2
        gamma = [ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))]^2
        alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
        S = 2 * (e^alpha - 1) / (1 + e^alpha), clipped at 0.

    Args:
        high: High price series.
        low: Low price series.
        window: If specified, smooths spread using rolling mean over window.

    Returns:
        pd.Series: Corwin-Schultz spread values bounded in [0, 1].
    """
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low) if not isinstance(low, pd.Series) else low

    log_hl = np.log(h / l)
    beta = log_hl**2 + log_hl.shift(1)**2

    h_max = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l_min = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h_max / l_min)**2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    sqrt_beta = np.sqrt(beta)
    alpha = (np.sqrt(2.0) * sqrt_beta - sqrt_beta) / denom - np.sqrt(gamma / denom)

    # Spread calculation: S = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    exp_alpha = np.exp(alpha)
    s = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
    s_clipped = s.clip(lower=0.0).fillna(0.0)

    if window is not None and window > 1:
        return s_clipped.rolling(window=window).mean().rename("corwin_schultz")

    return s_clipped.rename("corwin_schultz")


def order_book_imbalance(
    bid_vol: pd.Series | np.ndarray | float,
    ask_vol: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    """Calculate Order Book Imbalance (OBI).

    OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol), bounded in [-1, 1].
    Safely returns 0.0 when bid_vol + ask_vol == 0.

    Args:
        bid_vol: Bid volume or depth.
        ask_vol: Ask volume or depth.

    Returns:
        Imbalance value(s) in [-1, 1].
    """
    tot = bid_vol + ask_vol
    if isinstance(tot, (pd.Series, pd.DataFrame)):
        safe_tot = tot.replace(0.0, np.nan)
        return ((bid_vol - ask_vol) / safe_tot).fillna(0.0).clip(-1.0, 1.0)
    if isinstance(tot, np.ndarray):
        safe_tot = np.where(tot == 0.0, np.nan, tot)
        res = (bid_vol - ask_vol) / safe_tot
        res = np.nan_to_num(res, nan=0.0)
        return np.clip(res, -1.0, 1.0)

    return 0.0 if tot == 0.0 else max(-1.0, min(1.0, float((bid_vol - ask_vol) / tot)))


def depth_imbalance(book_df: pd.DataFrame, levels: int = 5) -> pd.Series:
    """Calculate cumulative order book depth imbalance across the top K levels.

    Args:
        book_df: DataFrame with columns 'bid_sz_k' and 'ask_sz_k' for k in 1..levels.
        levels: Number of levels to aggregate (default 5).

    Returns:
        pd.Series: Depth imbalance in [-1, 1].
    """
    bid_sum = pd.Series(0.0, index=book_df.index)
    ask_sum = pd.Series(0.0, index=book_df.index)

    for k in range(1, levels + 1):
        bid_col = f"bid_sz_{k}"
        ask_col = f"ask_sz_{k}"
        if bid_col in book_df.columns:
            bid_sum += book_df[bid_col]
        if ask_col in book_df.columns:
            ask_sum += book_df[ask_col]

    return order_book_imbalance(bid_sum, ask_sum).rename("depth_imbalance")


def microprice(book_df: pd.DataFrame) -> pd.Series:
    """Calculate microprice (volume-weighted top of book midprice).

    microprice = (ask_px_1 * bid_sz_1 + bid_px_1 * ask_sz_1) / (bid_sz_1 + ask_sz_1).

    Args:
        book_df: DataFrame containing 'bid_px_1', 'bid_sz_1', 'ask_px_1', 'ask_sz_1'.

    Returns:
        pd.Series: Microprice values strictly bounded between bid_px_1 and ask_px_1.
    """
    b_px = book_df["bid_px_1"]
    b_sz = book_df["bid_sz_1"]
    a_px = book_df["ask_px_1"]
    a_sz = book_df["ask_sz_1"]

    tot_sz = b_sz + a_sz
    safe_sz = tot_sz.replace(0.0, np.nan)
    mp = (a_px * b_sz + b_px * a_sz) / safe_sz
    simple_mid = (b_px + a_px) / 2.0
    return mp.fillna(simple_mid).rename("microprice")


def estimate_kyle_lambda(
    price_changes: pd.Series | np.ndarray,
    signed_volume: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Estimate Kyle's Lambda price impact coefficient via OLS with an intercept.

    Model: dP = alpha + lambda * Q + epsilon.

    Args:
        price_changes: Sequence of price changes dP.
        signed_volume: Sequence of order flow / signed volumes Q.

    Returns:
        tuple[float, float]: (lambda_coeff, t_stat).
    """
    dp = np.asarray(price_changes, dtype=float)
    q = np.asarray(signed_volume, dtype=float)

    valid = ~(np.isnan(dp) | np.isnan(q))
    dp = dp[valid]
    q = q[valid]

    n = len(dp)
    if n < 3:
        return 0.0, 0.0

    # Demeaned OLS with intercept
    q_mean = float(np.mean(q))
    dp_mean = float(np.mean(dp))

    q_dm = q - q_mean
    dp_dm = dp - dp_mean

    ss_q = float(np.sum(q_dm**2))
    if ss_q == 0.0:
        return 0.0, 0.0

    lam = float(np.sum(q_dm * dp_dm) / ss_q)
    alpha = dp_mean - lam * q_mean

    residuals = dp - (alpha + lam * q)
    ssr = float(np.sum(residuals**2))
    df = n - 2
    s2 = ssr / df if df > 0 else 0.0
    se_lam = np.sqrt(s2 / ss_q)

    t_stat = float(lam / se_lam) if se_lam > 0.0 else 0.0
    return lam, t_stat


def amihud_illiquidity(
    returns: pd.Series | np.ndarray,
    dollar_volume: pd.Series | np.ndarray,
    window: int = 20,
) -> pd.Series:
    """Calculate Amihud (2002) illiquidity ratio.

    Amihud = rolling_mean(|Return| / Dollar_Volume).

    Args:
        returns: Asset return series.
        dollar_volume: Traded dollar volume series.
        window: Rolling window length (default 20).

    Returns:
        pd.Series: Amihud illiquidity metric.
    """
    ret = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
    dvol = pd.Series(dollar_volume) if not isinstance(dollar_volume, pd.Series) else dollar_volume

    safe_dvol = dvol.replace(0.0, np.nan)
    ratio = (ret.abs() / safe_dvol).fillna(0.0)
    return ratio.rolling(window=window).mean().rename("amihud_illiquidity")
