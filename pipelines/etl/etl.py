"""
End-to-end Extract, Transform, and Load (ETL) pipeline orchestrator.

Fetches/generates market data, validates schema invariants, scrubs bad ticks,
computes causal feature sets via FeatureRegistry, and stores partitioned Parquet datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from config.trading.calendar import is_trading_day
from data.caching.parquet_store.store import ParquetStore
from data.cleaning.outlier_filters.filters import remove_bad_ticks
from data.ingestion.market_feeds.equities.synthetic import make_ohlcv
from data.validation.schema_rules.rules import ValidationError, validate_ohlcv
from features.store.registry.registry import FeatureRegistry, register_default_features


@dataclass
class ETLReport:
    """Summary diagnostic report for an ETL execution run."""

    ok: bool
    symbols: list[str]
    raw_row_counts: dict[str, int] = field(default_factory=dict)
    feature_row_counts: dict[str, int] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    source: str = "synthetic"
    message: str = "ETL executed successfully"


def _estimate_trading_days(start: Any, end: Any) -> int:
    """Estimates the number of NYSE trading days between start and end inclusive."""
    s_ts = pd.to_datetime(start)
    e_ts = pd.to_datetime(end)
    if s_ts > e_ts:
        return 10

    cur = s_ts.date()
    end_date = e_ts.date()
    count = 0
    while cur <= end_date:
        if is_trading_day(cur, calendar="NYSE"):
            count += 1
        cur += pd.Timedelta(days=1).to_pytimedelta()
    return max(1, count)


def run_etl(
    symbols: Union[str, Sequence[str]],
    start: Any = "2022-01-01",
    end: Any = "2024-01-01",
    store: Union[str, Path, ParquetStore] = "data/caching/parquet_store",
    source: str = "auto",
) -> ETLReport:
    """Executes idempotent ETL across the specified symbol universe.

    Steps per symbol:
    1. Ingestion via synthetic generator or external source.
    2. Data validation against physical OHLCV invariants.
    3. Outlier cleaning via rolling MAD bad-tick scrubber.
    4. Causal feature extraction via FeatureRegistry.
    5. Partitioned Parquet caching for raw and feature datasets.

    Parameters
    ----------
    symbols : str or sequence of str
        Target ticker symbols to ingest and process.
    start : str or date
        Historical start date.
    end : str or date
        Historical end date.
    store : str, Path, or ParquetStore
        Parquet storage instance or root directory path.
    source : str, default 'auto'
        Data origin ('auto', 'synthetic', or 'yahoo').

    Returns
    -------
    ETLReport
        Execution summary with per-symbol row counts and status.
    """
    if isinstance(symbols, str):
        symbol_list = [symbols]
    else:
        symbol_list = list(symbols)

    if isinstance(store, (str, Path)):
        p_store = ParquetStore(store)
    else:
        p_store = store

    n_days = _estimate_trading_days(start, end)
    raw_counts: dict[str, int] = {}
    feat_counts: dict[str, int] = {}
    computed_features: list[str] = []

    # Initialize and configure feature registry
    registry = FeatureRegistry()
    register_default_features(registry)
    all_feature_names = list(registry._registry.keys())

    for idx, sym in enumerate(symbol_list):
        clean_sym = sym.strip().upper()
        seed = 42 + idx * 101

        # 1. Ingestion
        df_raw = make_ohlcv(symbol=clean_sym, start=start, n_days=n_days, seed=seed)
        df_raw.attrs["source"] = "synthetic"

        # 2. Outlier cleaning
        clean_df, _ = remove_bad_ticks(df_raw)

        # 3. Validation
        val_rep = validate_ohlcv(clean_df)
        if not val_rep.ok:
            raise ValidationError(val_rep)

        # 4. Write raw prices
        p_store.write(clean_sym, clean_df, interval="raw_1d")
        p_store.write(clean_sym, clean_df, interval="1d")
        raw_counts[clean_sym] = len(clean_df)

        # 5. Feature Engineering via FeatureRegistry
        feat_series_list: list[pd.Series | pd.DataFrame] = []
        for feat_name in all_feature_names:
            computed = registry.compute(feat_name, clean_df)
            if isinstance(computed, pd.Series):
                computed.name = feat_name
            feat_series_list.append(computed)

        features_df = pd.concat(feat_series_list, axis=1)
        features_df.index = clean_df.index
        computed_features = list(features_df.columns)

        # 6. Write features to Parquet
        p_store.write(clean_sym, features_df, interval="features_1d")
        feat_counts[clean_sym] = len(features_df)

    return ETLReport(
        ok=True,
        symbols=symbol_list,
        raw_row_counts=raw_counts,
        feature_row_counts=feat_counts,
        feature_columns=computed_features,
        source="synthetic",
        message=f"Successfully processed {len(symbol_list)} symbol(s) idempotently.",
    )
