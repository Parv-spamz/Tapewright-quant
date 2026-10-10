"""Lopez de Prado financial machine learning labeling toolkit.

Includes:
- Exponential moving average volatility estimation (daily_vol)
- Symmetric CUSUM event filter
- Triple-barrier method with horizontal, stop-loss, and vertical timeout barriers
- Meta-labeling for secondary position-sizing models
- Average label uniqueness and sample weight calculation
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Calculate causal, past-only exponential moving standard deviation of returns.

    Args:
        close: Price series.
        span: EWM span period (default 100).

    Returns:
        pd.Series: Daily volatility estimate aligned with close.index.
    """
    returns = close.pct_change()
    vol = returns.ewm(span=span).std()
    return vol.fillna(0.0).rename("daily_vol")


def cusum_filter(
    close: pd.Series,
    threshold: float | pd.Series,
) -> pd.Index:
    """Symmetric CUSUM event filter for structural price breaks and volatility events.

    Accumulates signed returns:
        s_pos = max(0, s_pos + r_t)
        s_neg = min(0, s_neg + r_t)
    When s_pos > threshold or s_neg < -threshold, an event is triggered and that
    side's accumulator is reset to 0.

    Args:
        close: Asset price series.
        threshold: Event barrier threshold (constant float or dynamic pd.Series e.g. daily_vol).

    Returns:
        pd.Index: Timestamps / indices where CUSUM barrier events were triggered.
    """
    s = pd.Series(close)
    returns = s.pct_change().dropna()
    events: list[Any] = []

    s_pos = 0.0
    s_neg = 0.0
    is_series_threshold = isinstance(threshold, pd.Series)

    for t, ret in returns.items():
        th = float(threshold.loc[t]) if is_series_threshold else float(threshold)
        if th <= 0.0:
            continue

        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if s_pos > th:
            events.append(t)
            s_pos = 0.0
        elif s_neg < -th:
            events.append(t)
            s_neg = 0.0

    return pd.Index(events)


def triple_barrier(
    close: pd.Series,
    events: pd.Index | Sequence[Any],
    pt_sl: tuple[float, float] = (1.0, 1.0),
    target: pd.Series | float = 0.02,
    horizon_bars: int = 10,
    side: pd.Series | int | float | None = None,
    min_ret: float = 0.0,
    vertical_label: str = "zero",
) -> pd.DataFrame:
    """Apply the Lopez de Prado Triple-Barrier labeling method.

    For each event timestamp t0:
    - Upper barrier (profit-take) = +pt * target
    - Lower barrier (stop-loss) = -sl * target
    - Vertical barrier = t0 + horizon_bars
    First touch strictly after t0 determines the outcome.

    Resolution Rule: If both upper and lower barriers are breached in the same bar,
    profit-taking ('pt') is prioritized and recorded as the touch.

    Args:
        close: Price series.
        events: Event timestamps to label.
        pt_sl: Tuple (pt_multiplier, sl_multiplier).
        target: Target return scalar or pd.Series (e.g. daily_vol).
        horizon_bars: Number of bars until vertical expiration.
        side: Position side (+1 long, -1 short, or None). If given, scales return path.
        min_ret: Minimum absolute return required for non-zero vertical label.
        vertical_label: 'zero' (default) gives label 0 on vertical barrier timeout;
                        'sign' gives label = sign(ret) on timeout.

    Returns:
        pd.DataFrame with columns ['t0', 't1_touch', 'ret', 'label', 'barrier'].
            label: -1, 0, or 1.
            barrier: 'pt', 'sl', or 'vertical'.
    """
    if isinstance(close, pd.DataFrame):
        close_idx = close.index
        has_hl = "high" in close.columns and "low" in close.columns
        s_close = close["close"] if "close" in close.columns else close.iloc[:, 0]
        s_high = close["high"] if has_hl else s_close
        s_low = close["low"] if has_hl else s_close
    else:
        s_close = pd.Series(close)
        close_idx = s_close.index
        s_high = s_close
        s_low = s_close
        has_hl = False

    pt, sl = pt_sl

    is_target_series = isinstance(target, pd.Series)
    is_side_series = isinstance(side, pd.Series)

    records: list[dict[str, Any]] = []

    for t0 in events:
        if t0 not in close_idx:
            continue

        loc0 = close_idx.get_loc(t0)
        p0 = float(s_close.iloc[loc0])

        trgt = float(target.loc[t0]) if is_target_series else float(target)
        if trgt <= 0.0:
            continue

        s = float(side.loc[t0]) if is_side_series else (float(side) if side is not None else 1.0)

        end_loc = min(loc0 + horizon_bars, len(s_close) - 1)
        if end_loc <= loc0:
            continue

        up_barrier = pt * trgt
        dn_barrier = -sl * trgt

        touch_time = close_idx[end_loc]
        touch_barrier = "vertical"
        realized_ret = (float(s_close.iloc[end_loc]) - p0) / p0 * s
        touch_label = 0

        # Evaluate strictly on bars AFTER t0
        for step in range(loc0 + 1, end_loc + 1):
            t_step = close_idx[step]
            p_step = float(s_close.iloc[step])
            p_hi = float(s_high.iloc[step])
            p_lo = float(s_low.iloc[step])

            if s >= 0:
                ret_up = (p_hi - p0) / p0 * s
                ret_dn = (p_lo - p0) / p0 * s
                ret_step = (p_step - p0) / p0 * s
            else:
                ret_up = (p0 - p_lo) / p0
                ret_dn = (p0 - p_hi) / p0
                ret_step = (p0 - p_step) / p0

            hit_up = (pt > 0.0 and ret_up >= up_barrier) or (pt == 0.0 and ret_up >= 0.0)
            hit_dn = (sl > 0.0 and ret_dn <= dn_barrier) or (sl == 0.0 and ret_dn <= 0.0)

            if hit_up and hit_dn:
                # Documented Resolution: Profit-taking touch wins when both breached
                touch_time = t_step
                touch_barrier = "pt"
                realized_ret = up_barrier if has_hl else ret_step
                touch_label = 1
                break
            elif hit_up:
                touch_time = t_step
                touch_barrier = "pt"
                realized_ret = up_barrier if has_hl else ret_step
                touch_label = 1
                break
            elif hit_dn:
                touch_time = t_step
                touch_barrier = "sl"
                realized_ret = dn_barrier if has_hl else ret_step
                touch_label = -1
                break

        if touch_barrier == "vertical":
            if abs(realized_ret) < min_ret:
                touch_label = 0
            elif vertical_label == "sign":
                touch_label = int(np.sign(realized_ret))
            else:
                touch_label = 0

        records.append(
            {
                "t0": t0,
                "t1_touch": touch_time,
                "ret": realized_ret,
                "label": touch_label,
                "barrier": touch_barrier,
            }
        )

    return pd.DataFrame(records)


def meta_labels(
    triple_barrier_df: pd.DataFrame,
    side: pd.Series | Sequence[int | float] | int | float,
) -> pd.Series:
    """Generate meta-labels for secondary model position sizing.

    Meta-label = 1 if the primary model's position generates realized return > 0, else 0.

    Args:
        triple_barrier_df: DataFrame output from triple_barrier.
        side: Primary trade side direction (+1 or -1).

    Returns:
        pd.Series: Binary meta-labels in {0, 1}.
    """
    ret = triple_barrier_df["ret"].values
    if isinstance(side, pd.Series):
        s_arr = side.loc[triple_barrier_df["t0"]].values.astype(float)
    elif isinstance(side, (list, np.ndarray)):
        s_arr = np.asarray(side, dtype=float)
    else:
        s_arr = np.full(len(ret), float(side))

    meta_lbl = ((ret * s_arr) > 0.0).astype(int)
    return pd.Series(meta_lbl, index=triple_barrier_df.index, name="meta_label")


def average_uniqueness(
    t0_t1_df: pd.DataFrame,
    close_index: pd.Index | Sequence[Any],
) -> pd.Series:
    """Calculate the average uniqueness of overlapping label intervals.

    For each label i spanning [t0_i, t1_i]:
        u_i = 1 / |interval_i| * sum_{t in interval_i} (1 / c_t)
    where c_t is the concurrent number of active labels spanning bar t.

    Args:
        t0_t1_df: DataFrame containing 't0' and 't1_touch' columns.
        close_index: Underlying complete price time-series index.

    Returns:
        pd.Series: Average uniqueness per label in (0, 1].
    """
    c_index = pd.Index(close_index)
    idx_map = {ts: i for i, ts in enumerate(c_index)}
    n_bars = len(c_index)

    # Concurrency count array
    counts = np.zeros(n_bars, dtype=int)
    intervals: list[tuple[int, int]] = []

    for _, row in t0_t1_df.iterrows():
        t0 = row["t0"]
        t1 = row["t1_touch"]
        if t0 in idx_map and t1 in idx_map:
            i0 = idx_map[t0]
            i1 = idx_map[t1]
            if i0 > i1:
                i0, i1 = i1, i0
            counts[i0 : i1 + 1] += 1
            intervals.append((i0, i1))
        else:
            intervals.append((-1, -1))

    # Average uniqueness per label
    uniqueness_vals: list[float] = []
    for i0, i1 in intervals:
        if i0 < 0 or i1 < 0 or i1 < i0:
            uniqueness_vals.append(1.0)
            continue

        c_slice = counts[i0 : i1 + 1]
        u_mean = float(np.mean(1.0 / c_slice))
        uniqueness_vals.append(u_mean)

    return pd.Series(uniqueness_vals, index=t0_t1_df.index, name="uniqueness")


def sample_weights(
    t0_t1_df: pd.DataFrame,
    close_index: pd.Index | Sequence[Any],
    returns: pd.Series | Sequence[float] | None = None,
) -> pd.Series:
    """Calculate sample weights based on average uniqueness and optional absolute return.

    Weights are normalized so they sum to N (the number of labeled samples).

    Args:
        t0_t1_df: DataFrame with 't0' and 't1_touch'.
        close_index: Underlying price index.
        returns: Optional returns per sample to scale weights by |ret|.

    Returns:
        pd.Series: Normalized sample weights summing to len(t0_t1_df).
    """
    u = average_uniqueness(t0_t1_df, close_index)
    n = len(u)
    if n == 0:
        return pd.Series(dtype=float, name="weight")

    if returns is not None:
        r_arr = np.abs(np.asarray(returns, dtype=float))
        raw_w = u.values * r_arr
    else:
        raw_w = u.values

    w_sum = np.sum(raw_w)
    if w_sum > 0.0:
        weights = raw_w * (float(n) / w_sum)
    else:
        weights = np.ones(n, dtype=float)

    return pd.Series(weights, index=t0_t1_df.index, name="weight")
