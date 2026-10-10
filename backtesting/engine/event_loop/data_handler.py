from __future__ import annotations

from typing import Any
import pandas as pd

from backtesting.engine.event_loop.events import TickEvent


class HistoricBarDataHandler:
    """Historical bar data provider with look-ahead prevention and multi-asset timeline alignment."""

    def __init__(self, panel: dict[str, pd.DataFrame] | pd.DataFrame) -> None:
        self.bars: dict[str, pd.DataFrame] = {}

        if isinstance(panel, dict):
            for sym, df in panel.items():
                self.bars[sym] = self._normalize_df(df)
        elif isinstance(panel, pd.DataFrame):
            if isinstance(panel.index, pd.MultiIndex):
                # Levels could be (timestamp, symbol) or (symbol, timestamp)
                names = [str(n).lower() for n in panel.index.names]
                if "symbol" in names:
                    sym_level = names.index("symbol")
                    symbols = panel.index.get_level_values(sym_level).unique()
                    for sym in symbols:
                        sub_df = panel.xs(sym, level=sym_level)
                        self.bars[str(sym)] = self._normalize_df(sub_df)
                else:
                    raise ValueError("MultiIndex DataFrame must contain 'symbol' level.")
            elif "symbol" in panel.columns:
                for sym, group in panel.groupby("symbol"):
                    group_clean = group.drop(columns=["symbol"])
                    self.bars[str(sym)] = self._normalize_df(group_clean)
            else:
                # Single symbol dataframe
                self.bars["DEFAULT"] = self._normalize_df(panel)
        else:
            raise TypeError(f"panel must be dict[str, pd.DataFrame] or pd.DataFrame, got {type(panel)}.")

        # Build unified sorted timeline
        unique_timestamps: set[pd.Timestamp] = set()
        for df in self.bars.values():
            unique_timestamps.update(df.index)

        self._timeline: list[pd.Timestamp] = sorted(unique_timestamps)
        self._cursor: int = -1
        self.current_timestamp: pd.Timestamp | None = None

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()
        if not isinstance(clean.index, pd.DatetimeIndex):
            clean.index = pd.to_datetime(clean.index)
        clean.sort_index(inplace=True)
        # Rename columns to lowercase standard
        col_map = {str(c): str(c).lower() for c in clean.columns}
        clean.rename(columns=col_map, inplace=True)
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(clean.columns)
        if missing:
            raise ValueError(f"Bar data missing required columns: {missing}")
        return clean

    @property
    def continue_backtest(self) -> bool:
        """True if there are more timestamps to process along the timeline."""
        return self._cursor + 1 < len(self._timeline)

    def update_bars(self) -> list[TickEvent]:
        """Advance the timeline by one bar and emit TickEvents for symbols active at this timestamp."""
        if not self.continue_backtest:
            return []

        self._cursor += 1
        t = self._timeline[self._cursor]
        self.current_timestamp = t

        ticks: list[TickEvent] = []
        for symbol in sorted(self.bars.keys()):
            df = self.bars[symbol]
            if t in df.index:
                row = df.loc[t]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                tick = TickEvent(
                    timestamp=t,
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                ticks.append(tick)

        return ticks

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last n historical bars strictly up to and including the current timestamp."""
        if self.current_timestamp is None:
            return pd.DataFrame()

        df = self.bars.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        # Strict no-lookahead slice
        historical_slice = df.loc[df.index <= self.current_timestamp]
        if historical_slice.empty or n <= 0:
            return pd.DataFrame()

        return historical_slice.tail(n)

    def get_latest_bar_value(self, symbol: str, field: str = "close") -> float | None:
        """Return scalar latest value for a specific field strictly at or before current timestamp."""
        latest = self.get_latest_bars(symbol, n=1)
        if latest.empty or field.lower() not in latest.columns:
            return None
        return float(latest.iloc[-1][field.lower()])
