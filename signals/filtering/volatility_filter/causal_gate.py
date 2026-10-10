from __future__ import annotations

from typing import Any
import pandas as pd

from backtesting.engine.event_loop.events import SignalEvent, TickEvent
from backtesting.engine.event_loop.strategy_base import Strategy
from features.econometric.cointegration.causal_graph import CausalGraph


class CausalGate:
    """Predictive causality gating filter enforcing econometric precedence (Granger/TE)."""

    def __init__(self, graph: CausalGraph) -> None:
        self.graph = graph

    def allows(self, feature_name: str, target_name: str) -> bool:
        """Check whether feature_name is an admitted predictive parent of target_name."""
        return self.graph.is_admitted(feature_name, target_name)

    def filter_features(self, feature_df: pd.DataFrame, target_name: str) -> pd.DataFrame:
        """Filter a feature DataFrame to keep only statistically admitted predictive drivers.

        Args:
            feature_df: Candidate feature DataFrame (columns are feature names).
            target_name: Target asset or price return column.

        Returns:
            pd.DataFrame: Sliced DataFrame containing only admitted columns.

        Raises:
            ValueError: If no candidate features are admitted for target_name.
        """
        admitted = [c for c in feature_df.columns if self.allows(c, target_name)]
        if not admitted:
            raise ValueError(
                f"No features admitted by CausalGate for target '{target_name}'. "
                f"Candidate features {list(feature_df.columns)} failed Granger/TE causality criteria."
            )
        return feature_df[admitted]

    def gate_report(self) -> pd.DataFrame:
        """Produce tabular report of all evaluated edges with Granger and Transfer Entropy metrics."""
        return self.graph.edges.copy()


class CausalGatedStrategy(Strategy):
    """Strategy wrapper that suppresses trade signals whose driver feature is rejected by CausalGate."""

    def __init__(
        self,
        strategy: Strategy,
        gate: CausalGate,
        driver_features: dict[str, str] | str | None = None,
        target_name: str = "target",
    ) -> None:
        self.strategy = strategy
        self.gate = gate
        self.driver_features = driver_features
        self.target_name = target_name

    def on_bar(self, tick: TickEvent, data_handler: Any) -> list[SignalEvent]:
        # Identify driver feature associated with this symbol
        if isinstance(self.driver_features, dict):
            feat = self.driver_features.get(tick.symbol, tick.symbol)
        elif isinstance(self.driver_features, str):
            feat = self.driver_features
        else:
            feat = tick.symbol

        # Gating check: suppress signal if driver is not admitted
        if not self.gate.allows(feat, self.target_name):
            return []

        return self.strategy.on_bar(tick, data_handler)
