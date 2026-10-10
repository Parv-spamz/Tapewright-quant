from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def momentum_scores(
    panel: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """Calculate cross-sectional 12-1 momentum scores skipping recent short-term reversal period.

    Formula:
        Momentum(t) = Price(t - skip) / Price(t - lookback) - 1.0

    Strictly causal with zero forward-looking bias.

    Args:
        panel: DataFrame of asset close prices (rows = dates, columns = symbols).
        lookback: Full momentum lookback window in periods (default 252).
        skip: Short-term reversal skip period in periods (default 21).

    Returns:
        pd.DataFrame: Cross-sectional momentum returns.
    """
    if lookback <= skip:
        raise ValueError(f"lookback ({lookback}) must be strictly greater than skip ({skip})")

    p_skip = panel.shift(skip)
    p_lookback = panel.shift(lookback)
    scores = (p_skip / p_lookback) - 1.0
    return scores


def value_scores(
    panel: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate cross-sectional value factor scores from provided fundamental metrics.

    Fundamental metrics (e.g. Earnings/Price, Book/Price) MUST be supplied explicitly;
    never fabricates or mocks fundamental accounting data.

    Args:
        panel: DataFrame of asset prices for alignment.
        fundamentals_df: DataFrame of fundamental valuation ratios (e.g., E/P, B/P).

    Returns:
        pd.DataFrame: Cross-sectionally standardized z-scores.

    Raises:
        ValueError: If fundamentals_df is None or empty.
    """
    if fundamentals_df is None or fundamentals_df.empty:
        raise ValueError("fundamentals_df must be supplied; never fabricate fundamental accounting metrics.")

    # Align fundamentals with panel index and columns
    aligned_funds = fundamentals_df.reindex(index=panel.index, columns=panel.columns)

    # Cross-sectional z-score normalization across assets at each timestamp
    mean_val = aligned_funds.mean(axis=1)
    std_val = aligned_funds.std(axis=1, ddof=1)
    std_val = std_val.replace(0.0, np.nan)

    z_scores = aligned_funds.sub(mean_val, axis=0).div(std_val, axis=0)
    return z_scores


def decile_buckets(
    scores: pd.Series | pd.DataFrame,
    n: int = 10,
) -> pd.Series | pd.DataFrame:
    """Assign cross-sectional decile bucket signals with balanced long (+1) and short (-1) tails.

    Top decile assets are assigned +1 (long), bottom decile assets are assigned -1 (short),
    and intermediate deciles are assigned 0. Guarantees balanced count of +1 and -1 signals.

    Args:
        scores: Series or DataFrame of factor scores across assets.
        n: Number of quantiles/buckets (default 10 for deciles).

    Returns:
        pd.Series | pd.DataFrame: Integer signals in {-1, 0, 1}.
    """
    if isinstance(scores, pd.Series):
        s = scores.dropna()
        n_assets = len(s)
        if n_assets < 2:
            return pd.Series(0, index=scores.index, dtype=int)

        k = max(1, int(np.floor(n_assets / n)))
        ranked = s.rank(ascending=True, method="first")

        out = pd.Series(0, index=scores.index, dtype=int)
        bottom_idx = ranked[ranked <= k].index
        top_idx = ranked[ranked > (n_assets - k)].index

        out.loc[bottom_idx] = -1
        out.loc[top_idx] = 1
        return out

    elif isinstance(scores, pd.DataFrame):
        return scores.apply(lambda row: decile_buckets(row, n=n), axis=1)

    raise TypeError(f"scores must be pd.Series or pd.DataFrame, got {type(scores)}")


def long_short_weights(
    scores: pd.Series | pd.DataFrame,
    top_q: float = 0.10,
    bottom_q: float = 0.10,
) -> pd.Series | pd.DataFrame:
    """Compute dollar-neutral long/short portfolio allocation weights with gross exposure 1.0.

    Top quantile assets receive equal positive weights summing to +0.5.
    Bottom quantile assets receive equal negative weights summing to -0.5.
    Net weight sums to 0.0, and gross leverage sum(|w_i|) equals 1.0 within 1e-12.

    Args:
        scores: Cross-sectional factor score Series or DataFrame over time.
        top_q: Quantile threshold for long bucket (default 0.10).
        bottom_q: Quantile threshold for short bucket (default 0.10).

    Returns:
        pd.Series | pd.DataFrame: Sized portfolio weights.
    """
    if isinstance(scores, pd.Series):
        s = scores.dropna()
        n_assets = len(s)
        if n_assets < 2:
            return pd.Series(0.0, index=scores.index, dtype=float)

        k_top = max(1, int(np.round(n_assets * top_q)))
        k_bot = max(1, int(np.round(n_assets * bottom_q)))
        # Guard against overlapping buckets in small cross-sections
        if k_top + k_bot > n_assets:
            k_top = max(1, n_assets // 2)
            k_bot = max(1, n_assets // 2)

        ranked = s.rank(ascending=True, method="first")
        weights = pd.Series(0.0, index=scores.index, dtype=float)

        bot_indices = ranked[ranked <= k_bot].index
        top_indices = ranked[ranked > (n_assets - k_top)].index

        weights.loc[top_indices] = 0.5 / len(top_indices)
        weights.loc[bot_indices] = -0.5 / len(bot_indices)
        return weights

    elif isinstance(scores, pd.DataFrame):
        return scores.apply(lambda row: long_short_weights(row, top_q=top_q, bottom_q=bottom_q), axis=1)

    raise TypeError(f"scores must be pd.Series or pd.DataFrame, got {type(scores)}")
