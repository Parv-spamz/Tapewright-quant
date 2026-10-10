"""
Point-in-time daily signal generation pipeline.

Executes causal inference strictly on historical data <= asof,
applies regime volatility filters and predictive causality gates,
determines risk-budgeted position sizing, and logs signals to the relational database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from backtesting.metrics.risk_measures.sizing import position_size
from config.trading.risk_limits import RiskLimits
from data.caching.parquet_store.store import ParquetStore
from database.repositories.signal_repo.repo import SignalRepo
from signals.filtering.volatility_filter import CausalGate, simple_vol_filter
from signals.generation.breakout.breakout import donchian_breakout
from signals.generation.crossover.crossover import dual_ema_signals
from signals.generation.mean_reversion.mean_reversion import bollinger_rsi_reversion


def run_daily_signals(
    symbols: Union[str, Sequence[str]],
    asof: Any,
    store: Union[str, Path, ParquetStore],
    strategy_names: Sequence[str] = ("crossover",),
    signal_repo: Optional[SignalRepo] = None,
    model: Optional[Any] = None,
    gate: Optional[CausalGate] = None,
    limits: Optional[RiskLimits] = None,
) -> list[dict[str, Any]]:
    """Generates daily quantitative trading signals under strict causal truncation.

    Guarantees:
    - Loads price and indicator history strictly where timestamp <= asof.
    - Applies regime volatility filter and causal Granger gating.
    - Sizes target position via multi-stage Half-Kelly and CVaR risk budget.
    - Optionally logs signal audit records into the database via SignalRepo.

    Parameters
    ----------
    symbols : str or sequence of str
        Target ticker symbols.
    asof : str, datetime, or pd.Timestamp
        Historical point-in-time evaluation timestamp.
    store : str, Path, or ParquetStore
        Parquet cache storage.
    strategy_names : sequence of str, default ('crossover',)
        Candidate signal generation algorithms.
    signal_repo : SignalRepo, optional
        Database repository for persisting signals.
    model : Any, optional
        Trained ML/DL model for inference.
    gate : CausalGate, optional
        Predictive causality gate filter.
    limits : RiskLimits, optional
        Position sizing risk limits.

    Returns
    -------
    list of dict
        Generated signal records.
    """
    if isinstance(symbols, str):
        symbol_list = [symbols]
    else:
        symbol_list = list(symbols)

    if isinstance(store, (str, Path)):
        p_store = ParquetStore(store)
    else:
        p_store = store

    risk_limits = limits or RiskLimits()
    asof_ts = pd.to_datetime(asof)
    generated_signals: list[dict[str, Any]] = []

    for sym in symbol_list:
        clean_sym = sym.strip().upper()

        # Strict point-in-time truncation: load prices strictly <= asof
        df = p_store.read(clean_sym, interval="1d", end=asof_ts)
        if df.empty or len(df) < 5:
            continue

        # Trailing return distribution
        returns_series = df["close"].pct_change().dropna()
        if len(returns_series) < 5:
            continue

        rets_arr = returns_series.values
        mu = float(np.mean(rets_arr))
        sigma = float(np.std(rets_arr, ddof=1)) + 1e-8
        current_vol = float(np.std(rets_arr[-20:], ddof=1)) * np.sqrt(252.0) if len(rets_arr) >= 20 else sigma * np.sqrt(252.0)

        # Volatility regime filter
        vol_mask = simple_vol_filter(returns_series, span=20, threshold_quantile=0.80)
        is_vol_allowed = bool(vol_mask.iloc[-1]) if len(vol_mask) > 0 else True

        # Evaluate candidate strategies
        for strat in strategy_names:
            strat_lower = strat.strip().lower()

            if strat_lower == "crossover":
                raw_sig = dual_ema_signals(df["close"], fast=10, slow=30).iloc[-1]
            elif strat_lower == "breakout":
                raw_sig = donchian_breakout(df, n=20).iloc[-1]
            elif strat_lower in ("mean_reversion", "reversion"):
                raw_sig = bollinger_rsi_reversion(df).iloc[-1]
            elif model is not None and hasattr(model, "predict"):
                features_df = p_store.read(clean_sym, interval="features_1d", end=asof_ts)
                if not features_df.empty:
                    last_features = features_df.iloc[-1:].values
                    pred = model.predict(last_features)
                    raw_sig = 1.0 if pred > 0.5 else -1.0
                else:
                    raw_sig = 0.0
            else:
                raw_sig = 0.0

            # Causal gating check
            gate_allowed = True
            if gate is not None and hasattr(gate, "allows"):
                gate_allowed = bool(gate.allows(clean_sym, asof_ts))

            # Resolve signal type
            if not is_vol_allowed or not gate_allowed:
                signal_type = "HOLD"
                confidence = 0.0
                target_fraction = 0.0
            elif raw_sig > 0:
                signal_type = "BUY"
                confidence = float(min(1.0, abs(raw_sig)))
                base_weight, _ = position_size(
                    mu=mu,
                    sigma=sigma,
                    returns_window=rets_arr,
                    limits=risk_limits,
                    current_vol=current_vol,
                )
                target_fraction = float(base_weight * confidence)
            elif raw_sig < 0:
                signal_type = "SELL"
                confidence = float(min(1.0, abs(raw_sig)))
                base_weight, _ = position_size(
                    mu=mu,
                    sigma=sigma,
                    returns_window=rets_arr,
                    limits=risk_limits,
                    current_vol=current_vol,
                )
                target_fraction = float(-base_weight * confidence)
            else:
                signal_type = "HOLD"
                confidence = 0.5
                target_fraction = 0.0

            signal_record = {
                "symbol": clean_sym,
                "strategy": strat_lower,
                "asof": asof_ts,
                "signal_type": signal_type,
                "confidence_score": confidence,
                "target_weight": target_fraction,
                "current_vol": current_vol,
            }

            # Optional persistence to database
            if signal_repo is not None:
                try:
                    sig_id = signal_repo.insert_signal(
                        asset_id=clean_sym,
                        model_id=strat_lower,
                        signal_type=signal_type,
                        confidence_score=confidence,
                        target_weight=target_fraction,
                        timestamp=asof_ts,
                        metadata={"vol_filter": is_vol_allowed, "gate": gate_allowed},
                    )
                    signal_record["signal_id"] = sig_id
                except Exception as e:
                    logger.warning("Error inserting signal: %s", e)

            generated_signals.append(signal_record)

    return generated_signals
