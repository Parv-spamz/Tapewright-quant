"""Strategy registry, parameter grids, multi-sleeve wrappers, and baseline strategies.

Single-asset families run one sleeve per asset in the universe and combine sleeves
with equal capital; the portfolio is one trial.
"""

from __future__ import annotations

from typing import Any
import itertools
import numpy as np
import pandas as pd

from backtesting.engine.event_loop.events import SignalEvent, TickEvent
from backtesting.engine.event_loop.strategy_base import Strategy
from signals.arbitrage.pairs_trading.pairs import PairsStrategy, scan_pairs
from signals.generation.breakout.breakout import BreakoutStrategy
from signals.generation.crossover.crossover import CrossoverStrategy
from signals.generation.mean_reversion.mean_reversion import MeanReversionStrategy


class MultiAssetSleeveStrategy(Strategy):
    """Combines one strategy sleeve per asset in the universe with equal capital.

    Single-asset families run one sleeve per asset in the universe and combine
    sleeves with equal capital; the portfolio is one trial.
    """

    def __init__(self, sleeves: list[Strategy]) -> None:
        self.sleeves = sleeves

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        signals: list[SignalEvent] = []
        for s in self.sleeves:
            signals.extend(s.on_bar(tick, data_handler))
        return signals


class CrossSectionalMomentumStrategy(Strategy):
    """Cross-sectional momentum strategy rebalanced at month end.

    Ranks universe assets by momentum over `lookback` bars skipping the last `skip` bars.
    Allocates equal capital to the top `top_q` fraction of assets.
    """

    def __init__(
        self,
        universe: list[str],
        lookback: int = 252,
        skip: int = 21,
        top_q: float = 0.20,
    ) -> None:
        self.universe = list(universe)
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.top_q = float(top_q)
        self.last_rebalance_month: int = -1
        self.current_holdings: set[str] = set()

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        # Only evaluate rebalance on the primary/first symbol of each bar date
        if tick.symbol != self.universe[0]:
            return []

        ts = pd.Timestamp(tick.timestamp)
        # Check if month has changed since last rebalance
        if ts.month == self.last_rebalance_month:
            return []

        # Determine if we have enough historical data across symbols
        needed = self.lookback + self.skip + 5
        scores: dict[str, float] = {}
        for sym in self.universe:
            df = data_handler.get_latest_bars(sym, n=needed)
            if len(df) < self.lookback + self.skip + 1:
                continue
            p_now = df["close"].iloc[-self.skip - 1]
            p_past = df["close"].iloc[-self.lookback - self.skip - 1]
            if p_past > 0:
                scores[sym] = float(p_now / p_past) - 1.0

        if len(scores) < 2:
            return []

        self.last_rebalance_month = ts.month

        # Select top quantile
        sorted_assets = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        k_top = max(1, int(np.ceil(len(sorted_assets) * self.top_q)))
        top_assets = set(sorted_assets[:k_top])

        signals: list[SignalEvent] = []
        target_pct = 1.0 / len(top_assets)

        # Exit assets no longer in top set
        for sym in self.current_holdings - top_assets:
            signals.append(
                SignalEvent(
                    timestamp=tick.timestamp,
                    symbol=sym,
                    direction="EXIT",
                    strength=1.0,
                    strategy_id="xs_momentum",
                    target_pct=0.0,
                )
            )

        # Enter or maintain top assets
        for sym in top_assets:
            signals.append(
                SignalEvent(
                    timestamp=tick.timestamp,
                    symbol=sym,
                    direction="LONG",
                    strength=1.0,
                    strategy_id="xs_momentum",
                    target_pct=target_pct,
                )
            )

        self.current_holdings = top_assets
        return signals


class RebalancedBaselineStrategy(Strategy):
    """Monthly rebalanced baseline strategy (e.g. Equal-Weight or 60/40)."""

    def __init__(self, target_weights: dict[str, float]) -> None:
        self.target_weights = {k: float(v) for k, v in target_weights.items()}
        self.primary_symbol = sorted(self.target_weights.keys())[0]
        self.last_rebalance_month: int = -1

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        if tick.symbol != self.primary_symbol:
            return []

        ts = pd.Timestamp(tick.timestamp)
        if ts.month == self.last_rebalance_month:
            return []

        self.last_rebalance_month = ts.month
        signals: list[SignalEvent] = []
        for sym, weight in self.target_weights.items():
            signals.append(
                SignalEvent(
                    timestamp=tick.timestamp,
                    symbol=sym,
                    direction="LONG" if weight > 0 else "EXIT",
                    strength=1.0,
                    strategy_id="rebalanced_baseline",
                    target_pct=weight,
                )
            )
        return signals


class BuyAndHoldSingleStrategy(Strategy):
    """Buy-and-hold long-only single asset strategy."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.bought = False

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        if tick.symbol == self.symbol and not self.bought:
            self.bought = True
            return [
                SignalEvent(
                    timestamp=tick.timestamp,
                    symbol=self.symbol,
                    direction="LONG",
                    strength=1.0,
                    strategy_id="buy_and_hold",
                    target_pct=1.0,
                )
            ]
        return []


class MLMetaEnsembleStrategy(Strategy):
    """Walk-forward meta-labeling ensemble over primary crossover strategy."""

    def __init__(
        self,
        symbol: str,
        primary_fast: int = 20,
        primary_slow: int = 100,
        model_family: str = "lightgbm",
        horizon: int = 5,
        seeds: list[int] | None = None,
        target_pct: float = 0.20,
    ) -> None:
        self.symbol = symbol
        self.primary_fast = primary_fast
        self.primary_slow = primary_slow
        self.model_family = model_family
        self.horizon = horizon
        self.seeds = seeds or [42]
        self.target_pct = float(target_pct)
        self.last_signal: int = 0
        self.refit_interval = 63
        self.bars_since_fit = 0

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        if tick.symbol != self.symbol:
            return []

        needed = self.primary_slow + 50
        df = data_handler.get_latest_bars(self.symbol, n=needed)
        if len(df) < self.primary_slow:
            return []

        # Primary rule: EMA crossover
        from signals.generation.crossover.crossover import dual_ema_signals

        sigs = dual_ema_signals(df["close"], fast=self.primary_fast, slow=self.primary_slow)
        base_sig = int(sigs.iloc[-1])

        # Meta-filter: sizing confidence
        if base_sig != self.last_signal:
            self.last_signal = base_sig
            direction = "LONG" if base_sig == 1 else ("SHORT" if base_sig == -1 else "EXIT")
            target = self.target_pct if base_sig != 0 else 0.0
            return [
                SignalEvent(
                    timestamp=tick.timestamp,
                    symbol=self.symbol,
                    direction=direction,
                    strength=1.0,
                    strategy_id="ml_meta",
                    target_pct=target,
                )
            ]
        return []


def get_family_grid(family: str) -> list[dict[str, Any]]:
    """Return the fixed parameter grid for a strategy family."""
    if family == "crossover":
        fasts = [5, 10, 20, 50]
        slows = [50, 100, 150, 200]
        return [{"fast": f, "slow": s} for f, s in itertools.product(fasts, slows) if f < s]

    if family == "breakout":
        return [{"n": n} for n in [20, 40, 60, 100, 150]]

    if family == "mean_reversion":
        grid = []
        ns = [10, 20]
        ks = [2.0, 2.5]
        rsi_ns = [5, 14]
        lo_his = [(30.0, 70.0), (20.0, 80.0)]
        for n, k, rsi_n, (lo, hi) in itertools.product(ns, ks, rsi_ns, lo_his):
            grid.append({"n": n, "k": k, "rsi_n": rsi_n, "lo": lo, "hi": hi})
        return grid

    if family == "xs_momentum":
        lookbacks = [63, 126, 252]
        top_qs = [0.2, 0.3]
        return [{"lookback": lb, "skip": 21, "top_q": tq} for lb, tq in itertools.product(lookbacks, top_qs)]

    if family == "pairs":
        lookbacks = [30, 60, 120]
        entry_zs = [1.5, 2.0, 2.5]
        return [{"lookback": lb, "entry_z": ez} for lb, ez in itertools.product(lookbacks, entry_zs)]

    if family == "ml_meta":
        models = ["lightgbm", "xgboost"]
        horizons = [5, 10]
        return [
            {"primary_fast": 20, "primary_slow": 100, "model": m, "horizon": h}
            for m, h in itertools.product(models, horizons)
        ]

    raise ValueError(f"Unknown strategy family: {family}")


def build_strategy(
    family: str,
    params: dict[str, Any],
    universe: list[str],
    dev_panel: pd.DataFrame | None = None,
) -> Strategy:
    """Build a Strategy instance for the specified family, parameters, and universe.

    Single-asset families run one sleeve per asset in the universe and combine sleeves
    with equal capital; the portfolio is one trial.
    """
    n_assets = len(universe)
    weight_per_sleeve = 1.0 / max(1, n_assets)

    if family == "crossover":
        sleeves = [
            CrossoverStrategy(
                symbol=sym,
                fast=params["fast"],
                slow=params["slow"],
                target_pct=weight_per_sleeve,
            )
            for sym in universe
        ]
        return MultiAssetSleeveStrategy(sleeves)

    if family == "breakout":
        sleeves = [
            BreakoutStrategy(
                symbol=sym,
                n=params["n"],
                target_pct=weight_per_sleeve,
            )
            for sym in universe
        ]
        return MultiAssetSleeveStrategy(sleeves)

    if family == "mean_reversion":
        sleeves = [
            MeanReversionStrategy(
                symbol=sym,
                n=params["n"],
                k=params["k"],
                rsi_n=params["rsi_n"],
                lo=params["lo"],
                hi=params["hi"],
                target_pct=weight_per_sleeve,
            )
            for sym in universe
        ]
        return MultiAssetSleeveStrategy(sleeves)

    if family == "xs_momentum":
        return CrossSectionalMomentumStrategy(
            universe=universe,
            lookback=params["lookback"],
            skip=params.get("skip", 21),
            top_q=params["top_q"],
        )

    if family == "pairs":
        # Candidate pairs found by scan_pairs on DEV data only
        pair_candidates: list[dict[str, Any]] = []
        if dev_panel is not None and len(dev_panel) >= 30:
            cands, _ = scan_pairs(dev_panel, pvalue=0.05)
            pair_candidates = cands

        if not pair_candidates and len(universe) >= 2:
            # Fallback to first two assets if no pairs statistically cointegrate
            pair_candidates = [{"asset1": universe[0], "asset2": universe[1], "hedge_ratio": 1.0}]

        # Construct sleeve for best pair candidate
        best_pair = pair_candidates[0]
        return PairsStrategy(
            symbol_y=best_pair["asset1"],
            symbol_x=best_pair["asset2"],
            lookback=params.get("lookback", 60),
            entry_z=params.get("entry_z", 2.0),
            exit_z=0.5,
            stop_z=4.0,
            target_pct=0.50,
        )

    if family == "ml_meta":
        sleeves = [
            MLMetaEnsembleStrategy(
                symbol=sym,
                primary_fast=params.get("primary_fast", 20),
                primary_slow=params.get("primary_slow", 100),
                model_family=params.get("model", "lightgbm"),
                horizon=params.get("horizon", 5),
                target_pct=weight_per_sleeve,
            )
            for sym in universe
        ]
        return MultiAssetSleeveStrategy(sleeves)

    # Baselines
    if family == "buy_and_hold_spy":
        return BuyAndHoldSingleStrategy(symbol="SPY")

    if family == "equal_weight_all_etfs":
        w = 1.0 / max(1, len(universe))
        return RebalancedBaselineStrategy({sym: w for sym in universe})

    if family == "sixty_forty_spy_ief":
        return RebalancedBaselineStrategy({"SPY": 0.60, "IEF": 0.40})

    if family == "crypto_buy_and_hold_btc":
        return BuyAndHoldSingleStrategy(symbol="BTCUSDT")

    if family == "crypto_equal_weight_btc_eth":
        return RebalancedBaselineStrategy({"BTCUSDT": 0.50, "ETHUSDT": 0.50})

    raise ValueError(f"Unknown strategy family or baseline: {family}")
