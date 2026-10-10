"""Technical indicator frame builder.

Combines trend, momentum, volatility, and volume indicators into a wide,
cleanly prefixed feature DataFrame.
"""

from __future__ import annotations

from typing import Any
import pandas as pd

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


def build_indicator_frame(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a comprehensive wide DataFrame of technical indicators with prefixed columns.

    Args:
        df: Input market data DataFrame with 'open', 'high', 'low', 'close', 'volume'.
        config: Optional configuration dictionary overriding indicator hyperparameters.

    Returns:
        pd.DataFrame: Wide feature matrix containing computed indicators.
    """
    cfg = config or {}

    sma_n = cfg.get("sma_window", 20)
    ema_n = cfg.get("ema_window", 20)
    st_n = cfg.get("supertrend_window", 10)
    st_mult = cfg.get("supertrend_mult", 3.0)
    adx_n = cfg.get("adx_window", 14)

    rsi_n = cfg.get("rsi_window", 14)
    macd_fast = cfg.get("macd_fast", 12)
    macd_slow = cfg.get("macd_slow", 26)
    macd_sig = cfg.get("macd_signal", 9)
    stoch_k = cfg.get("stoch_k", 14)
    stoch_d = cfg.get("stoch_d", 3)
    roc_n = cfg.get("roc_window", 10)

    atr_n = cfg.get("atr_window", 14)
    bb_n = cfg.get("bollinger_window", 20)
    bb_k = cfg.get("bollinger_k", 2.0)
    kelt_n = cfg.get("keltner_window", 20)
    kelt_mult = cfg.get("keltner_mult", 2.0)
    hv_n = cfg.get("hv_window", 21)

    cmf_n = cfg.get("cmf_window", 20)
    vwap_sess = cfg.get("vwap_session", "D")

    close = df["close"]
    vol = df["volume"]

    features: dict[str, pd.Series] = {}

    # Trend
    features[f"trend_sma_{sma_n}"] = sma(close, n=sma_n)
    features[f"trend_ema_{ema_n}"] = ema(close, n=ema_n)

    st_df = supertrend(df, n=st_n, mult=st_mult)
    features["trend_supertrend"] = st_df["supertrend"]
    features["trend_supertrend_dir"] = st_df["direction"]

    adx_df = adx(df, n=adx_n)
    features["trend_adx"] = adx_df["adx"]
    features["trend_plus_di"] = adx_df["plus_di"]
    features["trend_minus_di"] = adx_df["minus_di"]

    # Momentum
    features[f"mom_rsi_{rsi_n}"] = rsi(close, n=rsi_n)

    macd_df = macd(close, fast=macd_fast, slow=macd_slow, signal=macd_sig)
    features["mom_macd"] = macd_df["macd"]
    features["mom_macd_signal"] = macd_df["signal"]
    features["mom_macd_hist"] = macd_df["hist"]

    stoch_df = stochastic(df, k=stoch_k, d=stoch_d)
    features["mom_stoch_k"] = stoch_df["stoch_k"]
    features["mom_stoch_d"] = stoch_df["stoch_d"]

    features[f"mom_roc_{roc_n}"] = roc(close, n=roc_n)

    # Volatility
    features[f"vol_atr_{atr_n}"] = atr(df, n=atr_n)

    bb_df = bollinger(close, n=bb_n, k=bb_k)
    features["vol_bb_mid"] = bb_df["mid"]
    features["vol_bb_upper"] = bb_df["upper"]
    features["vol_bb_lower"] = bb_df["lower"]
    features["vol_bb_pctb"] = bb_df["pctb"]
    features["vol_bb_bandwidth"] = bb_df["bandwidth"]

    kelt_df = keltner(df, n=kelt_n, mult=kelt_mult)
    features["vol_keltner_mid"] = kelt_df["mid"]
    features["vol_keltner_upper"] = kelt_df["upper"]
    features["vol_keltner_lower"] = kelt_df["lower"]

    features[f"vol_hist_vol_{hv_n}"] = historical_vol(close, n=hv_n)

    # Volume
    features["vol_obv"] = obv(close, vol)
    features["vol_vwap"] = vwap(df, session=vwap_sess)
    features[f"vol_cmf_{cmf_n}"] = chaikin_money_flow(df, n=cmf_n)

    return pd.DataFrame(features, index=df.index)
