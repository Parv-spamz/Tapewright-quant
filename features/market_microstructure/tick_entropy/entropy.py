"""Tick entropy and flow toxicity metrics.

Includes:
- Shannon Entropy of directional runs / word transitions
- Volume-Synchronized Probability of Toxicity (VPIN) via Bulk Volume Classification
- VPIN from signed trade tape
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.stats import norm


def shannon_entropy_of_runs(
    signs: Sequence[int | float] | np.ndarray | pd.Series,
    word_len: int = 2,
    encode_runs: bool = True,
) -> float:
    """Calculate the Shannon entropy of directional runs.

    In market microstructure runs analysis, a directional run transition is
    encoded by whether successive steps continue (+1) or reverse (-1) the run
    (s_t * s_{t-1}). When encode_runs=True, deterministic alternating (+1, -1)
    or trending sequences produce identical transitions with near-zero entropy,
    while i.i.d. random walks generate maximum entropy (~log2(2^word_len)).

    Args:
        signs: Sequence of trade / price change signs (typically in {-1, +1}).
        word_len: Length of overlapping symbolic words (default 2).
        encode_runs: If True, evaluates entropy on directional run transitions (s_t * s_{t-1}).
                     If False, evaluates entropy on raw sign words.

    Returns:
        float: Shannon entropy in bits (-sum p * log2(p)).
    """
    arr = np.asarray(signs, dtype=float)
    # Filter out exact zeros (neutral price ticks) if mixed
    valid_signs = arr[arr != 0.0]
    if len(valid_signs) < (word_len + 1 if encode_runs else word_len):
        return 0.0

    symbols = np.sign(valid_signs)

    if encode_runs:
        # Run transition sequence: +1 for continuation, -1 for alternation/reversal
        seq = symbols[1:] * symbols[:-1]
    else:
        seq = symbols

    if len(seq) < word_len:
        return 0.0

    # Extract overlapping words of length word_len
    words = [tuple(seq[i : i + word_len]) for i in range(len(seq) - word_len + 1)]
    if not words:
        return 0.0

    counts = Counter(words)
    total_words = len(words)
    probs = np.array(list(counts.values()), dtype=float) / total_words

    # Shannon entropy: -sum p * log2(p)
    entropy = -float(np.sum(probs * np.log2(probs)))
    return max(0.0, entropy)


def vpin(
    trades_df: pd.DataFrame,
    bucket_volume: float,
    n_buckets: int = 50,
) -> pd.Series:
    """Calculate Volume-Synchronized Probability of Toxicity (VPIN).

    Uses Bulk Volume Classification (BVC) where buy fraction = Phi(dP / sigma_dP).
    sigma_dp is calculated strictly expanding/past-only to prevent lookahead bias.
    Trades are partitioned into equal-volume buckets of size bucket_volume, splitting
    individual trades across bucket boundaries if necessary.

    Args:
        trades_df: DataFrame containing 'price' and 'size' columns.
        bucket_volume: Target volume per bucket (V).
        n_buckets: Number of buckets in the rolling calculation window (N).

    Returns:
        pd.Series: VPIN index per bucket, bounded in [0, 1].
    """
    return _compute_vpin_core(
        trades_df=trades_df,
        bucket_volume=bucket_volume,
        n_buckets=n_buckets,
        use_signed=False,
    )


def vpin_from_signed(
    trades_df: pd.DataFrame,
    bucket_volume: float,
    n_buckets: int = 50,
) -> pd.Series:
    """Calculate VPIN from a trade tape with explicitly signed trades.

    Uses known trade side (+1 for buy, -1 for sell) rather than BVC.

    Args:
        trades_df: DataFrame containing 'price', 'size', and 'side' columns.
        bucket_volume: Target volume per bucket (V).
        n_buckets: Number of buckets in the rolling calculation window (N).

    Returns:
        pd.Series: VPIN index per bucket, bounded in [0, 1].
    """
    return _compute_vpin_core(
        trades_df=trades_df,
        bucket_volume=bucket_volume,
        n_buckets=n_buckets,
        use_signed=True,
    )


def _compute_vpin_core(
    trades_df: pd.DataFrame,
    bucket_volume: float,
    n_buckets: int,
    use_signed: bool,
) -> pd.Series:
    if len(trades_df) == 0 or bucket_volume <= 0.0 or n_buckets <= 0:
        return pd.Series(dtype=float, name="vpin")

    prices = trades_df["price"].values
    sizes = trades_df["size"].values

    if use_signed and "side" in trades_df.columns:
        sides = trades_df["side"].values
        buy_fracs = np.where(sides > 0, 1.0, np.where(sides < 0, 0.0, 0.5))
    else:
        # Causal, past-only price change standard deviation
        dp = np.diff(prices, prepend=prices[0])
        sigma_dp = pd.Series(dp).expanding(min_periods=2).std().fillna(0.0).values
        safe_sigma = np.where(sigma_dp > 0.0, sigma_dp, np.nan)
        z = np.nan_to_num(dp / safe_sigma, nan=0.0)
        buy_fracs = norm.cdf(z)

    # Form equal-volume buckets with trade splitting
    bucket_v_buy: list[float] = []
    bucket_v_sell: list[float] = []

    curr_b_buy = 0.0
    curr_b_sell = 0.0
    curr_b_vol = 0.0

    for size, bf in zip(sizes, buy_fracs):
        rem_size = float(size)
        sf = 1.0 - float(bf)

        while rem_size > 0.0:
            needed = bucket_volume - curr_b_vol
            if rem_size >= needed:
                curr_b_buy += needed * bf
                curr_b_sell += needed * sf
                bucket_v_buy.append(curr_b_buy)
                bucket_v_sell.append(curr_b_sell)

                rem_size -= needed
                curr_b_buy = 0.0
                curr_b_sell = 0.0
                curr_b_vol = 0.0
            else:
                curr_b_buy += rem_size * bf
                curr_b_sell += rem_size * sf
                curr_b_vol += rem_size
                rem_size = 0.0

    if not bucket_v_buy:
        return pd.Series(dtype=float, name="vpin")

    v_buy = np.array(bucket_v_buy, dtype=float)
    v_sell = np.array(bucket_v_sell, dtype=float)

    bucket_imbalances = np.abs(v_buy - v_sell) / bucket_volume
    vpin_values = (
        pd.Series(bucket_imbalances)
        .rolling(window=n_buckets)
        .mean()
        .clip(0.0, 1.0)
    )
    vpin_values.name = "vpin"
    return vpin_values
