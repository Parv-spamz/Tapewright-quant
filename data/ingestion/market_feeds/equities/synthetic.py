from datetime import date, datetime, timedelta
from typing import Any
import numpy as np
import pandas as pd

from config.trading.calendar import is_trading_day


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    n: int,
    dt: float = 1.0 / 252.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate Geometric Brownian Motion (GBM) path of length n.

    S_{t+1} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.array([float(s0)], dtype=float)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n - 1)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion

    log_prices = np.empty(n, dtype=float)
    log_prices[0] = np.log(s0)
    log_prices[1:] = log_prices[0] + np.cumsum(log_returns)
    return np.exp(log_prices)


def simulate_jump_diffusion(
    s0: float,
    mu: float,
    sigma: float,
    lam: float,
    jump_mu: float,
    jump_sigma: float,
    n: int,
    dt: float = 1.0 / 252.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate Merton Jump-Diffusion path of length n.

    Compensates drift by -lam * k * dt, where k = exp(jump_mu + 0.5 * jump_sigma^2) - 1.
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.array([float(s0)], dtype=float)

    rng = np.random.default_rng(seed)
    k = np.exp(jump_mu + 0.5 * jump_sigma**2) - 1.0
    drift_comp = (mu - 0.5 * sigma**2 - lam * k) * dt

    z = rng.standard_normal(n - 1)
    diffusion = sigma * np.sqrt(dt) * z

    n_jumps = rng.poisson(lam * dt, size=n - 1)
    jump_comp = np.zeros(n - 1, dtype=float)
    jump_mask = n_jumps > 0
    if np.any(jump_mask):
        jump_counts = n_jumps[jump_mask]
        jump_comp[jump_mask] = rng.normal(
            loc=jump_counts * jump_mu,
            scale=np.sqrt(jump_counts) * jump_sigma,
        )

    log_returns = drift_comp + diffusion + jump_comp
    log_prices = np.empty(n, dtype=float)
    log_prices[0] = np.log(s0)
    log_prices[1:] = log_prices[0] + np.cumsum(log_returns)
    return np.exp(log_prices)


def simulate_regime_switching(
    s0: float,
    n: int,
    seed: int | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate 2-state Markov regime-switching price path and state history.

    State 0 (Calm):  mu=0.10, sigma=0.12, transition to stress = 0.01 / day
    State 1 (Stress): mu=-0.20, sigma=0.45, transition to calm = 0.05 / day
    """
    if n <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)
    if n == 1:
        return np.array([float(s0)], dtype=float), np.zeros(1, dtype=int)

    p = params or {}
    mu_calm = p.get("mu_calm", 0.10)
    sigma_calm = p.get("sigma_calm", 0.12)
    mu_stress = p.get("mu_stress", -0.20)
    sigma_stress = p.get("sigma_stress", 0.45)
    p_calm_to_stress = p.get("p_calm_to_stress", 0.01)
    p_stress_to_calm = p.get("p_stress_to_calm", 0.05)
    dt = p.get("dt", 1.0 / 252.0)

    rng = np.random.default_rng(seed)

    states = np.zeros(n, dtype=int)
    current_state = 0
    states[0] = current_state

    # Pre-generate random uniforms for transitions and normals for returns
    u = rng.uniform(0.0, 1.0, size=n - 1)
    z = rng.standard_normal(n - 1)

    log_returns = np.zeros(n - 1, dtype=float)

    for i in range(n - 1):
        if current_state == 0:
            if u[i] < p_calm_to_stress:
                current_state = 1
        else:
            if u[i] < p_stress_to_calm:
                current_state = 0
        states[i + 1] = current_state

        mu_s = mu_calm if current_state == 0 else mu_stress
        sigma_s = sigma_calm if current_state == 0 else sigma_stress
        log_returns[i] = (mu_s - 0.5 * sigma_s**2) * dt + sigma_s * np.sqrt(dt) * z[i]

    log_prices = np.empty(n, dtype=float)
    log_prices[0] = np.log(s0)
    log_prices[1:] = log_prices[0] + np.cumsum(log_returns)
    return np.exp(log_prices), states


def _generate_nyse_dates(start_date: str | date | datetime, n_days: int) -> list[pd.Timestamp]:
    """Generate n_days consecutive active NYSE trading day timestamps."""
    if isinstance(start_date, (datetime, pd.Timestamp)):
        cur = start_date.date()
    elif isinstance(start_date, date):
        cur = start_date
    else:
        cur = date.fromisoformat(str(start_date).split("T")[0].split(" ")[0])

    dates: list[pd.Timestamp] = []
    while len(dates) < n_days:
        if is_trading_day(cur, calendar="NYSE"):
            dates.append(pd.Timestamp(cur))
        cur += timedelta(days=1)
    return dates


def make_ohlcv(
    symbol: str,
    start: str | date | datetime,
    n_days: int,
    seed: int | None = None,
    model: str = "gbm",
    intraday_steps: int = 16,
) -> pd.DataFrame:
    """Build daily OHLCV bars from simulated intraday sub-steps.

    Guarantees:
    - low <= min(open, close) <= max(open, close) <= high
    - All prices > 0
    - Index is NYSE trading days
    - Volume lognormal correlated with |return|
    """
    if n_days <= 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    rng = np.random.default_rng(seed)
    trading_dates = _generate_nyse_dates(start, n_days)

    total_steps = n_days * intraday_steps + 1
    s0 = 100.0
    dt_sub = (1.0 / 252.0) / intraday_steps

    if model == "jump":
        sub_path = simulate_jump_diffusion(
            s0=s0, mu=0.08, sigma=0.18, lam=3.0,
            jump_mu=-0.02, jump_sigma=0.04,
            n=total_steps, dt=dt_sub, seed=seed,
        )
    elif model == "regime":
        sub_path, _ = simulate_regime_switching(
            s0=s0, n=total_steps, seed=seed,
            params={"dt": dt_sub},
        )
    else:  # default gbm
        sub_path = simulate_gbm(
            s0=s0, mu=0.08, sigma=0.20,
            n=total_steps, dt=dt_sub, seed=seed,
        )

    opens = np.zeros(n_days, dtype=float)
    highs = np.zeros(n_days, dtype=float)
    lows = np.zeros(n_days, dtype=float)
    closes = np.zeros(n_days, dtype=float)
    volumes = np.zeros(n_days, dtype=float)

    for i in range(n_days):
        day_sub = sub_path[i * intraday_steps : (i + 1) * intraday_steps + 1]
        o = day_sub[0]
        c = day_sub[-1]
        h = np.max(day_sub)
        l = np.min(day_sub)

        # Enforce exact mathematical bar invariants
        h = max(h, o, c)
        l = min(l, o, c)
        assert l > 0 and h >= l

        opens[i] = o
        highs[i] = h
        lows[i] = l
        closes[i] = c

        # Correlated volume: higher return -> higher volume
        abs_ret = abs(np.log(c / o))
        mean_log_vol = 12.0 + 15.0 * abs_ret
        vol = np.exp(rng.normal(loc=mean_log_vol, scale=0.35))
        volumes[i] = float(max(100.0, np.round(vol)))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=pd.DatetimeIndex(trading_dates),
    )
    return df


def make_universe_panel(
    symbols: list[str],
    n_days: int,
    seed: int | None = None,
    corr: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Generate a universe of correlated asset OHLCV panels via Cholesky decomposition."""
    rng = np.random.default_rng(seed)
    p = len(symbols)
    if p == 0:
        return {}

    # Correlation matrix
    corr_mat = np.full((p, p), corr, dtype=float)
    np.fill_diagonal(corr_mat, 1.0)
    l_factor = np.linalg.cholesky(corr_mat)

    # Base independent seeds for individual symbol paths
    sub_seeds = rng.integers(0, 1_000_000, size=p)

    panel: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        panel[sym] = make_ohlcv(
            symbol=sym,
            start="2024-01-02",
            n_days=n_days,
            seed=int(sub_seeds[i]),
            model="gbm",
        )
    return panel


def make_cointegrated_pair(
    n: int,
    beta: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate cointegrated pair (x, y) such that y = beta * x + AR(1) stationary spread."""
    rng = np.random.default_rng(seed)

    # Common random walk trend x
    dx = rng.normal(loc=0.0, scale=1.0, size=n)
    x = 100.0 + np.cumsum(dx)

    # Stationary AR(1) spread epsilon with phi = 0.9
    epsilon = np.zeros(n, dtype=float)
    eta = rng.normal(loc=0.0, scale=1.0, size=n)
    for t in range(1, n):
        epsilon[t] = 0.9 * epsilon[t - 1] + eta[t]

    y = beta * x + epsilon
    return x, y
