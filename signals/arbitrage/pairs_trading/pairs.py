from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from backtesting.engine.event_loop.events import SignalEvent, TickEvent
from backtesting.engine.event_loop.strategy_base import Strategy
from features.econometric.cointegration.tests import engle_granger
from features.econometric.kalman_filter.kalman import KalmanHedgeRatio


class PairsSignal:
    """Mean-reverting pairs trading spread signal generator using a 3-state machine {+1, -1, 0}."""

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        lookback: int = 60,
    ) -> None:
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.stop_z = float(stop_z)
        self.lookback = int(lookback)

    def compute_spread(
        self,
        y: pd.Series | np.ndarray,
        x: pd.Series | np.ndarray,
        method: str = "kalman",
    ) -> pd.Series:
        """Calculate cointegrating spread y - beta * x - alpha.

        Args:
            y: Dependent asset price series.
            x: Independent asset price series.
            method: 'kalman' for dynamic hedge ratio or 'ols' for static hedge ratio.

        Returns:
            pd.Series: Estimated spread series.
        """
        y_s = pd.Series(y, dtype=float)
        x_s = pd.Series(x, dtype=float)

        if method == "kalman":
            kf = KalmanHedgeRatio()
            res = kf.fit(x_s, y_s)
            return pd.Series(res["spread"].values, index=y_s.index, name="spread")
        else:
            eg = engle_granger(y_s, x_s)
            return pd.Series(eg["residual"].values, index=y_s.index, name="spread")

    def z_score(self, spread: pd.Series | np.ndarray) -> pd.Series:
        """Compute rolling z-score of the spread."""
        s = pd.Series(spread, dtype=float)
        mean_s = s.rolling(self.lookback).mean()
        std_s = s.rolling(self.lookback).std(ddof=1)
        # Avoid division by zero
        std_s = std_s.replace(0.0, np.nan)
        z = (s - mean_s) / std_s
        return z.fillna(0.0)

    def generate_from_zscores(self, z_scores: pd.Series | np.ndarray) -> pd.Series:
        """Run the pairs trading 3-state machine over a sequence of z-scores.

        States:
            0: Flat (no position)
           +1: Long spread (long y, short x)
           -1: Short spread (short y, long x)

        Transitions:
            From 0:
              z <= -entry_z -> +1
              z >=  entry_z -> -1
            From +1:
              z <= -stop_z  ->  0 (stop loss)
              z >= -exit_z  ->  0 (mean reversion exit)
            From -1:
              z >=  stop_z  ->  0 (stop loss)
              z <=  exit_z  ->  0 (mean reversion exit)

        Args:
            z_scores: Series or array of standardized spread z-scores.

        Returns:
            pd.Series: Position state sequence in {+1, -1, 0}.
        """
        z_arr = np.asarray(z_scores, dtype=float)
        T = len(z_arr)
        positions = np.zeros(T, dtype=int)
        curr_state = 0

        for t in range(T):
            z = z_arr[t]
            if np.isnan(z):
                positions[t] = curr_state
                continue

            if curr_state == 0:
                if z <= -self.entry_z:
                    curr_state = 1
                elif z >= self.entry_z:
                    curr_state = -1
            elif curr_state == 1:
                if z <= -self.stop_z:
                    curr_state = 0  # Stop loss
                elif z >= -self.exit_z:
                    curr_state = 0  # Reversion target hit
            elif curr_state == -1:
                if z >= self.stop_z:
                    curr_state = 0  # Stop loss
                elif z <= self.exit_z:
                    curr_state = 0  # Reversion target hit

            positions[t] = curr_state

        idx = z_scores.index if isinstance(z_scores, pd.Series) else None
        return pd.Series(positions, index=idx, dtype=int)

    def generate_signals(self, spread: pd.Series | np.ndarray) -> pd.Series:
        """Compute z-score and execute state machine over input spread series."""
        z = self.z_score(spread)
        return self.generate_from_zscores(z)


def scan_pairs(
    panel: pd.DataFrame,
    pvalue: float = 0.05,
) -> tuple[list[dict[str, Any]], int]:
    """Scan all asset pairs in a panel for cointegration using Engle-Granger two-step tests.

    MULTIPLE TESTING CAVEAT:
        Testing M * (M - 1) / 2 pairs concurrently inflates the family-wise error rate.
        Pairs identified as statistically significant at nominal p-value thresholds
        must be validated out-of-sample and subjected to False Discovery Rate (FDR)
        or Bonferroni adjustments prior to live allocation.

    Args:
        panel: DataFrame of asset price series (columns are asset symbols).
        pvalue: Significance cutoff threshold for cointegration (default 0.05).

    Returns:
        tuple[list[dict[str, Any]], int]: (candidates_list, n_tests)
    """
    cols = list(panel.columns)
    M = len(cols)
    n_tests = M * (M - 1) // 2

    candidates: list[dict[str, Any]] = []

    for i in range(M):
        for j in range(i + 1, M):
            sym1 = cols[i]
            sym2 = cols[j]
            s1 = panel[sym1].dropna()
            s2 = panel[sym2].dropna()
            common_idx = s1.index.intersection(s2.index)
            if len(common_idx) < 30:
                continue

            res = engle_granger(s1.loc[common_idx], s2.loc[common_idx])
            pval = res["pvalue"]
            if pval <= pvalue:
                candidates.append({
                    "asset1": sym1,
                    "asset2": sym2,
                    "stat": res["stat"],
                    "pvalue": pval,
                    "hedge_ratio": res["hedge_ratio"],
                })

    return candidates, n_tests


class PairsStrategy(Strategy):
    """Event-driven statistical arbitrage pairs trading strategy."""

    def __init__(
        self,
        symbol_y: str,
        symbol_x: str,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        lookback: int = 60,
        target_pct: float = 0.20,
    ) -> None:
        self.symbol_y = symbol_y
        self.symbol_x = symbol_x
        self.target_pct = float(target_pct)
        self.signal_gen = PairsSignal(
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            lookback=lookback,
        )
        self.last_state: int = 0

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        # Only evaluate upon receiving the tick for dependent asset symbol_y
        if tick.symbol != self.symbol_y:
            return []

        needed = self.signal_gen.lookback + 30
        df_y = data_handler.get_latest_bars(self.symbol_y, n=needed)
        df_x = data_handler.get_latest_bars(self.symbol_x, n=needed)

        if len(df_y) < self.signal_gen.lookback or len(df_x) < self.signal_gen.lookback:
            return []

        common_idx = df_y.index.intersection(df_x.index)
        if len(common_idx) < self.signal_gen.lookback:
            return []

        y_close = df_y.loc[common_idx, "close"]
        x_close = df_x.loc[common_idx, "close"]

        spread = self.signal_gen.compute_spread(y_close, x_close, method="kalman")
        states = self.signal_gen.generate_signals(spread)
        curr_state = int(states.iloc[-1])

        if curr_state == self.last_state:
            return []

        self.last_state = curr_state
        signals: list[SignalEvent] = []

        if curr_state == 1:
            # Long spread: Long y, Short x
            signals.append(SignalEvent(timestamp=tick.timestamp, symbol=self.symbol_y, direction="LONG", target_pct=self.target_pct, strategy_id="pairs"))
            signals.append(SignalEvent(timestamp=tick.timestamp, symbol=self.symbol_x, direction="SHORT", target_pct=self.target_pct, strategy_id="pairs"))
        elif curr_state == -1:
            # Short spread: Short y, Long x
            signals.append(SignalEvent(timestamp=tick.timestamp, symbol=self.symbol_y, direction="SHORT", target_pct=self.target_pct, strategy_id="pairs"))
            signals.append(SignalEvent(timestamp=tick.timestamp, symbol=self.symbol_x, direction="LONG", target_pct=self.target_pct, strategy_id="pairs"))
        else:
            # Exit spread
            signals.append(SignalEvent(timestamp=tick.timestamp, symbol=self.symbol_y, direction="EXIT", target_pct=0.0, strategy_id="pairs"))
            signals.append(SignalEvent(timestamp=tick.timestamp, symbol=self.symbol_x, direction="EXIT", target_pct=0.0, strategy_id="pairs"))

        return signals
