import numpy as np
import pandas as pd


def simulate_l2_tape(
    n_steps: int,
    seed: int | None = None,
    s0: float = 100.0,
    tick: float = 0.01,
    sigma_step: float = 0.0004,
    kappa: float = 0.0003,
    levels: int = 5,
    base_size: float = 500.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate Level-2 limit order book tape with a planted microstructure signal.

    Generates:
    - imbalance I_t = 0.9 * I_{t-1} + 0.3 * N(0,1), clipped to [-3, 3]
    - mid_return_t = kappa * tanh(I_{t-1}) + sigma_step * N(0,1)
    - mid_t snapped to half-tick grid
    - spread_ticks_t = 1 + Poisson(0.3 + 0.5 * |mid_return_t| / sigma_step * 0.1) (>= 1)
    - bid/ask level sizes: base_size * exp(+/- 0.5 * I_t + 0.3 * N(0,1)) / k^0.5
    - trades per step: Poisson(3) count, side = +1 with prob 0.5 + 0.3 * tanh(I_t), size lognormal(log(80), 0.8)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (book_df, trades_df)
    """
    if n_steps <= 0:
        book_cols = (
            [f"bid_px_{k}" for k in range(1, levels + 1)]
            + [f"bid_sz_{k}" for k in range(1, levels + 1)]
            + [f"ask_px_{k}" for k in range(1, levels + 1)]
            + [f"ask_sz_{k}" for k in range(1, levels + 1)]
            + ["mid", "spread", "imbalance_true"]
        )
        return (
            pd.DataFrame(columns=book_cols),
            pd.DataFrame(columns=["timestamp", "price", "size", "side"]),
        )

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-02 09:30:00", periods=n_steps, freq="1s")

    imbalance_true = np.zeros(n_steps, dtype=float)
    mid_returns = np.zeros(n_steps, dtype=float)
    mid_prices = np.zeros(n_steps, dtype=float)
    mid_prices[0] = round(s0 / (tick / 2.0)) * (tick / 2.0)

    spread_ticks = np.zeros(n_steps, dtype=int)
    spreads = np.zeros(n_steps, dtype=float)

    # Pre-generate noise for step efficiency
    z_imb = rng.normal(0.0, 1.0, size=n_steps)
    z_ret = rng.normal(0.0, 1.0, size=n_steps)

    for t in range(n_steps):
        if t == 0:
            imbalance_true[0] = float(np.clip(0.3 * z_imb[0], -3.0, 3.0))
            mid_returns[0] = 0.0
        else:
            prev_imb = imbalance_true[t - 1]
            imbalance_true[t] = float(np.clip(0.9 * prev_imb + 0.3 * z_imb[t], -3.0, 3.0))
            # Planted predictability: last step's imbalance predicts this step's return
            mid_returns[t] = float(kappa * np.tanh(prev_imb) + sigma_step * z_ret[t])
            raw_mid = mid_prices[t - 1] * np.exp(mid_returns[t])
            mid_prices[t] = round(raw_mid / (tick / 2.0)) * (tick / 2.0)

        # Spread dynamic
        lam_spread = 0.3 + 0.5 * (abs(mid_returns[t]) / sigma_step) * 0.1
        sp_ticks = 1 + int(rng.poisson(lam_spread))
        spread_ticks[t] = max(1, sp_ticks)
        spreads[t] = spread_ticks[t] * tick

    # Generate Book DataFrame columns
    book_dict: dict[str, Any] = {}

    best_bids = np.zeros(n_steps, dtype=float)
    best_asks = np.zeros(n_steps, dtype=float)

    for t in range(n_steps):
        half_sp = spreads[t] / 2.0
        b1 = round((mid_prices[t] - half_sp) / tick) * tick
        a1 = round((mid_prices[t] + half_sp) / tick) * tick
        if a1 <= b1:
            a1 = b1 + tick
        best_bids[t] = round(b1, 4)
        best_asks[t] = round(a1, 4)

    # Populate 5 levels of bids and asks
    k_factors = np.sqrt(np.arange(1, levels + 1, dtype=float))

    for k in range(1, levels + 1):
        book_dict[f"bid_px_{k}"] = np.round(best_bids - (k - 1) * tick, 4)

    for k in range(1, levels + 1):
        z_sz_bid = rng.normal(0.0, 1.0, size=n_steps)
        raw_bid_sz = base_size * np.exp(0.5 * imbalance_true + 0.3 * z_sz_bid) / k_factors[k - 1]
        book_dict[f"bid_sz_{k}"] = np.maximum(1, np.round(raw_bid_sz)).astype(int)

    for k in range(1, levels + 1):
        book_dict[f"ask_px_{k}"] = np.round(best_asks + (k - 1) * tick, 4)

    for k in range(1, levels + 1):
        z_sz_ask = rng.normal(0.0, 1.0, size=n_steps)
        raw_ask_sz = base_size * np.exp(-0.5 * imbalance_true + 0.3 * z_sz_ask) / k_factors[k - 1]
        book_dict[f"ask_sz_{k}"] = np.maximum(1, np.round(raw_ask_sz)).astype(int)

    book_dict["mid"] = mid_prices
    book_dict["spread"] = np.round(best_asks - best_bids, 4)
    book_dict["imbalance_true"] = imbalance_true

    book_df = pd.DataFrame(book_dict, index=timestamps)

    # Generate Trades DataFrame
    trade_ts: list[pd.Timestamp] = []
    trade_px: list[float] = []
    trade_sz: list[int] = []
    trade_side: list[int] = []

    m_trades = rng.poisson(3, size=n_steps)

    for t in range(n_steps):
        count = m_trades[t]
        if count == 0:
            continue
        cur_ts = timestamps[t]
        p_buy = float(np.clip(0.5 + 0.3 * np.tanh(imbalance_true[t]), 0.05, 0.95))
        u_sides = rng.uniform(0.0, 1.0, size=count)
        sides = np.where(u_sides < p_buy, 1, -1)

        # Lognormal size: median around 80, sd = 0.8
        sizes = np.maximum(1, np.round(rng.lognormal(mean=np.log(80.0), sigma=0.8, size=count))).astype(int)

        ask1 = best_asks[t]
        bid1 = best_bids[t]

        for i in range(count):
            trade_ts.append(cur_ts)
            trade_side.append(int(sides[i]))
            trade_sz.append(int(sizes[i]))
            trade_px.append(ask1 if sides[i] == 1 else bid1)

    trades_df = pd.DataFrame(
        {
            "timestamp": trade_ts,
            "price": trade_px,
            "size": trade_sz,
            "side": trade_side,
        }
    )

    return book_df, trades_df
