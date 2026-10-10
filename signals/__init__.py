from __future__ import annotations

from signals.arbitrage import (
    PairsSignal,
    PairsStrategy,
    scan_pairs,
)
from signals.filtering import (
    CausalGate,
    CausalGatedStrategy,
    apply_filter,
    garch_regime_filter,
    simple_vol_filter,
)
from signals.generation import (
    BreakoutStrategy,
    CrossoverStrategy,
    MeanReversionStrategy,
    bollinger_rsi_reversion,
    donchian_breakout,
    dual_ema_signals,
    golden_cross_events,
    volatility_breakout,
)
from signals.persist import persist_signals
from signals.ranking import (
    decile_buckets,
    long_short_weights,
    momentum_scores,
    value_scores,
)

__all__ = [
    "dual_ema_signals",
    "golden_cross_events",
    "CrossoverStrategy",
    "donchian_breakout",
    "volatility_breakout",
    "BreakoutStrategy",
    "bollinger_rsi_reversion",
    "MeanReversionStrategy",
    "garch_regime_filter",
    "simple_vol_filter",
    "apply_filter",
    "CausalGate",
    "CausalGatedStrategy",
    "momentum_scores",
    "value_scores",
    "decile_buckets",
    "long_short_weights",
    "PairsSignal",
    "scan_pairs",
    "PairsStrategy",
    "persist_signals",
]
