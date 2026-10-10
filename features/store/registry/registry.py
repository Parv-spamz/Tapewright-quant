"""Point-in-time feature store registry.

Enforces strict causal evaluation, declared feature lags, and historical truncation
invariance to guarantee zero forward-looking data leakage.
"""

from __future__ import annotations

from typing import Any, Callable
import pandas as pd

from features.econometric.fractional_diff.ffd import frac_diff_ffd
from features.technical_indicators.momentum.momentum import (
    macd,
    roc,
    rsi,
    stochastic,
)
from features.technical_indicators.trend.trend import (
    adx,
    ema,
    sma,
    supertrend,
)
from features.technical_indicators.volatility.volatility import (
    atr,
    bollinger,
    historical_vol,
    keltner,
)
from features.technical_indicators.volume.volume import (
    chaikin_money_flow,
    obv,
    vwap,
)


class FeatureRegistry:
    """Central registry for quantitative feature definitions with point-in-time guarantees."""

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
        depends_on: list[str] | None = None,
        lag: int = 1,
        description: str = "",
    ) -> None:
        """Register a feature calculation function.

        Args:
            name: Unique feature identifier.
            func: Pure function taking raw DataFrame and returning Series or DataFrame.
            depends_on: List of raw input column dependencies (e.g. ['close', 'volume']).
            lag: Causal publication delay bars (must be >= 0, default 1).
            description: Human-readable explanation.

        Raises:
            ValueError: If lag is negative.
        """
        if lag < 0:
            raise ValueError(f"Feature lag cannot be negative (got lag={lag})")

        self._registry[name] = {
            "name": name,
            "func": func,
            "depends_on": depends_on or [],
            "lag": lag,
            "description": description,
        }

    def compute(
        self,
        name: str,
        raw: pd.DataFrame,
    ) -> pd.Series | pd.DataFrame:
        """Compute the feature across the full raw DataFrame with declared lag shifting.

        Args:
            name: Registered feature name.
            raw: Input DataFrame.

        Returns:
            pd.Series | pd.DataFrame: Computed feature shifted by its declared lag.
        """
        if name not in self._registry:
            raise KeyError(f"Feature '{name}' is not registered")

        entry = self._registry[name]
        func = entry["func"]
        lag = entry["lag"]

        raw_feat = func(raw)
        if lag > 0:
            return raw_feat.shift(lag)
        return raw_feat

    def point_in_time(
        self,
        name: str,
        raw: pd.DataFrame,
        asof: Any,
    ) -> pd.Series | pd.DataFrame:
        """Compute feature with strict point-in-time truncation as of timestamp 'asof'.

        Truncates raw data to rows <= asof BEFORE invoking the calculation function,
        and applies declared causal lag.

        Args:
            name: Feature identifier.
            raw: Full historical DataFrame.
            asof: Historical evaluation timestamp.

        Returns:
            pd.Series | pd.DataFrame: Point-in-time feature output.
        """
        if name not in self._registry:
            raise KeyError(f"Feature '{name}' is not registered")

        # Strict historical truncation BEFORE compute
        truncated_raw = raw.loc[raw.index <= asof].copy()
        return self.compute(name, truncated_raw)

    def catalog(self) -> pd.DataFrame:
        """Return a catalog DataFrame of all registered features and metadata."""
        records = []
        for entry in self._registry.values():
            records.append(
                {
                    "name": entry["name"],
                    "lag": entry["lag"],
                    "depends_on": ", ".join(entry["depends_on"]),
                    "description": entry["description"],
                }
            )
        return pd.DataFrame(records)


def register_default_features(registry: FeatureRegistry) -> None:
    """Register standard technical indicators and FFD features into the registry.

    All features are configured with lag=1 by default to ensure no lookahead.
    """
    # Trend
    registry.register(
        "sma_20",
        lambda df: sma(df["close"], n=20),
        depends_on=["close"],
        lag=1,
        description="20-day Simple Moving Average",
    )
    registry.register(
        "ema_20",
        lambda df: ema(df["close"], n=20),
        depends_on=["close"],
        lag=1,
        description="20-day Exponential Moving Average",
    )
    registry.register(
        "supertrend",
        lambda df: supertrend(df, n=10, mult=3.0),
        depends_on=["high", "low", "close"],
        lag=1,
        description="10-bar Supertrend indicator with final-band carry-forward",
    )
    registry.register(
        "adx_14",
        lambda df: adx(df, n=14)["adx"],
        depends_on=["high", "low", "close"],
        lag=1,
        description="14-day Average Directional Index",
    )

    # Momentum
    registry.register(
        "rsi_14",
        lambda df: rsi(df["close"], n=14),
        depends_on=["close"],
        lag=1,
        description="14-day Relative Strength Index (Wilder smoothing)",
    )
    registry.register(
        "macd_hist",
        lambda df: macd(df["close"])["hist"],
        depends_on=["close"],
        lag=1,
        description="MACD (12, 26, 9) histogram",
    )
    registry.register(
        "stoch_k",
        lambda df: stochastic(df, k=14, d=3)["stoch_k"],
        depends_on=["high", "low", "close"],
        lag=1,
        description="14-period Stochastic %K",
    )
    registry.register(
        "roc_10",
        lambda df: roc(df["close"], n=10),
        depends_on=["close"],
        lag=1,
        description="10-day Rate of Change",
    )

    # Volatility
    registry.register(
        "atr_14",
        lambda df: atr(df, n=14),
        depends_on=["high", "low", "close"],
        lag=1,
        description="14-day Average True Range",
    )
    registry.register(
        "bb_pctb",
        lambda df: bollinger(df["close"], n=20, k=2.0)["pctb"],
        depends_on=["close"],
        lag=1,
        description="20-day Bollinger %B",
    )
    registry.register(
        "keltner_mid",
        lambda df: keltner(df, n=20, mult=2.0)["mid"],
        depends_on=["high", "low", "close"],
        lag=1,
        description="20-day Keltner Channel midline",
    )
    registry.register(
        "hist_vol_21",
        lambda df: historical_vol(df["close"], n=21),
        depends_on=["close"],
        lag=1,
        description="21-day Annualized Historical Volatility",
    )

    # Volume
    registry.register(
        "obv",
        lambda df: obv(df["close"], df["volume"]),
        depends_on=["close", "volume"],
        lag=1,
        description="On Balance Volume",
    )
    registry.register(
        "vwap",
        lambda df: vwap(df, session="D"),
        depends_on=["high", "low", "close", "volume"],
        lag=1,
        description="Volume Weighted Average Price",
    )
    registry.register(
        "cmf_20",
        lambda df: chaikin_money_flow(df, n=20),
        depends_on=["high", "low", "close", "volume"],
        lag=1,
        description="20-day Chaikin Money Flow",
    )

    # Econometric Memory
    registry.register(
        "ffd_04",
        lambda df: frac_diff_ffd(df["close"], d=0.4),
        depends_on=["close"],
        lag=1,
        description="Fractionally differentiated close price (d=0.4)",
    )
