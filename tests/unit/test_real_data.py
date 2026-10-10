"""Unit tests for strict real data ingestion, parsing, splits, and storage isolation.

All test payloads are hand-written offline mock data; zero external network requests are executed.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from research.splits import EQUITY_SPLITS, CRYPTO_SPLITS, split_of
from research.real_data import (
    FeedError,
    fetch_binance_symbol,
    parse_binance_klines,
    parse_fred_csv,
    parse_yahoo_json,
    store_dataset_splits,
)
from data.caching.parquet_store.store import ParquetStore
from integrity.ledger import Ledger


# Hand-written mock payload for Yahoo Finance chart endpoint
HAND_WRITTEN_YAHOO_JSON = {
    "chart": {
        "result": [
            {
                "meta": {
                    "currency": "USD",
                    "symbol": "SPY",
                    "gmtoffset": -18000,
                },
                "timestamp": [1704205800, 1704292200, 1704378600],  # 2024-01-02, 2024-01-03, 2024-01-04
                "indicators": {
                    "quote": [
                        {
                            "open": [472.16, 470.50, 469.00],
                            "high": [473.00, 471.20, 470.00],
                            "low": [470.00, 469.00, 468.00],
                            "close": [471.50, 470.00, 468.50],
                            "volume": [1000000, 1100000, 950000],
                        }
                    ],
                    "adjclose": [
                        {
                            "adjclose": [468.00, 466.51, 465.02]
                        }
                    ],
                },
            }
        ],
        "error": None,
    }
}

# Hand-written mock payload for Yahoo Finance with null entries
HAND_WRITTEN_YAHOO_WITH_NULLS = {
    "chart": {
        "result": [
            {
                "meta": {"currency": "USD", "symbol": "SPY", "gmtoffset": 0},
                "timestamp": [100000, 100001, 100002, 100003],
                "indicators": {
                    "quote": [
                        {
                            "open": [10.0, None, 12.0, 13.0],
                            "high": [11.0, 11.5, 12.5, 13.5],
                            "low": [9.5, 10.0, 11.5, 12.5],
                            "close": [10.5, 11.0, 12.0, 13.0],
                            "volume": [100, 200, 300, 400],
                        }
                    ],
                    "adjclose": [
                        {"adjclose": [10.5, 11.0, 12.0, 13.0]}
                    ],
                },
            }
        ],
        "error": None,
    }
}

# Hand-written mock payload for Binance daily klines
# Each element: [open_time, open, high, low, close, volume, close_time, quote_vol, n_trades, tb_base, tb_quote, ignore]
HAND_WRITTEN_BINANCE_KLINES = [
    [
        1698796800000,  # 2023-11-01
        "34500.00",
        "35200.50",
        "34100.00",
        "35150.00",
        "12500.5",
        1698883199999,
        "435000000.0",
        45000,
        "6200.0",
        "216000000.0",
        "0",
    ],
    [
        1698883200000,  # 2023-11-02
        "35150.00",
        "35900.00",
        "34900.00",
        "35800.00",
        "14200.0",
        1698969599999,
        "502000000.0",
        51000,
        "7100.0",
        "251000000.0",
        "0",
    ],
]

# Hand-written mock payload for FRED CSV
HAND_WRITTEN_FRED_CSV = """DATE,DGS10
2023-10-30,4.88
2023-10-31,.
2023-11-01,4.77
2023-11-02,4.67
"""


def test_split_of_boundaries():
    """Verify split_of boundary dates classify correctly across dev, val, and lockbox."""
    # Equity split boundaries
    assert split_of("2005-01-03", EQUITY_SPLITS) == "dev"
    assert split_of("2019-12-31", EQUITY_SPLITS) == "dev"
    assert split_of("2020-01-01", EQUITY_SPLITS) == "val"
    assert split_of("2023-12-31", EQUITY_SPLITS) == "val"
    assert split_of("2024-01-01", EQUITY_SPLITS) == "lockbox"
    assert split_of("2025-06-15", EQUITY_SPLITS) == "lockbox"

    # Crypto split boundaries
    assert split_of("2017-08-17", CRYPTO_SPLITS) == "dev"
    assert split_of("2021-12-31", CRYPTO_SPLITS) == "dev"
    assert split_of("2022-01-01", CRYPTO_SPLITS) == "val"
    assert split_of("2023-12-31", CRYPTO_SPLITS) == "val"
    assert split_of("2024-01-01", CRYPTO_SPLITS) == "lockbox"


def test_parse_yahoo_json():
    """Verify Yahoo JSON parsing computes adj_factor and total-return adjusted OHLC."""
    raw_bytes = json.dumps(HAND_WRITTEN_YAHOO_JSON).encode("utf-8")
    df = parse_yahoo_json(raw_bytes)

    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # Check that prices were adjusted downward by adj_factor
    # On first row: close was 471.50, adjclose was 468.00 -> close must equal 468.00
    assert abs(df["close"].iloc[0] - 468.00) < 1e-6
    # Volume is preserved
    assert df["volume"].iloc[0] == 1000000.0


def test_yahoo_null_threshold_raises():
    """Verify that > 0.5% null entries raises FeedError."""
    raw_bytes = json.dumps(HAND_WRITTEN_YAHOO_WITH_NULLS).encode("utf-8")
    # 1 null out of 4 rows = 25% > 0.5%
    with pytest.raises(FeedError) as exc_info:
        parse_yahoo_json(raw_bytes)
    assert "exceeding 0.5% limit" in str(exc_info.value)


def test_parse_binance_klines():
    """Verify Binance klines parsing with string-to-float conversions and timestamps."""
    raw_bytes = json.dumps(HAND_WRITTEN_BINANCE_KLINES).encode("utf-8")
    df = parse_binance_klines(raw_bytes)

    assert len(df) == 2
    assert "quote_volume" in df.columns
    assert "n_trades" in df.columns
    assert df["open"].iloc[0] == 34500.00
    assert df["close"].iloc[1] == 35800.00
    assert df.index[0] == pd.Timestamp("2023-11-01")


def test_binance_pagination_logic(tmp_path):
    """Verify Binance pagination iterates across pages until limit is reached."""
    # Fake fetch function simulating 2 pages of 1000 bars then 1 final page
    call_count = 0

    def fake_fetch(url: str) -> bytes:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 1000 bars
            bars = []
            for i in range(1000):
                ms = 1600000000000 + i * 86400000
                bars.append([ms, "100", "110", "90", "105", "10", ms + 86399999, "1050", 5, "5", "525", "0"])
            return json.dumps(bars).encode("utf-8")
        elif call_count == 2:
            # 5 bars (less than 1000 triggers end)
            bars = []
            for i in range(5):
                ms = 1600000000000 + (1000 + i) * 86400000
                bars.append([ms, "105", "115", "95", "110", "12", ms + 86399999, "1320", 6, "6", "660", "0"])
            return json.dumps(bars).encode("utf-8")
        return b"[]"

    df, pages, raw_bytes, base = fetch_binance_symbol(
        "BTCUSDT",
        start_ms=1600000000000,
        archive_dir=tmp_path / "archive",
        fetch_func=fake_fetch,
    )

    assert call_count == 2
    assert len(df) == 1005
    assert len(pages) == 2


def test_parse_fred_csv():
    """Verify FRED CSV parsing drops '.' missing values and returns float series."""
    df = parse_fred_csv(HAND_WRITTEN_FRED_CSV.encode("utf-8"))

    # 4 rows originally, 1 with '.' -> 3 valid rows
    assert len(df) == 3
    assert df["value"].iloc[0] == 4.88
    assert df["value"].iloc[1] == 4.77
    assert df.index[1] == pd.Timestamp("2023-11-01")


def test_store_dataset_splits_isolation(tmp_path):
    """Verify dev/val rows land strictly in store_devval and lockbox rows land in vault."""
    store_devval_path = tmp_path / "store_devval"
    vault_path = tmp_path / "vault"

    # Create synthetic frame spanning dev, val, and lockbox
    dates = pd.date_range("2019-12-30", periods=5, freq="D")
    # 2019-12-30: dev
    # 2019-12-31: dev
    # 2020-01-01: val
    # 2020-01-02: val
    # 2024-01-02: lockbox (explicit date)
    idx = [
        pd.Timestamp("2019-12-30"),
        pd.Timestamp("2019-12-31"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2024-01-02"),
    ]
    df = pd.DataFrame({
        "open": [10.0, 11.0, 12.0, 13.0, 14.0],
        "high": [11.0, 12.0, 13.0, 14.0, 15.0],
        "low": [9.0, 10.0, 11.0, 12.0, 13.0],
        "close": [10.5, 11.5, 12.5, 13.5, 14.5],
        "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
    }, index=pd.DatetimeIndex(idx))

    devval_n, vault_n = store_dataset_splits(
        "SPY", df, is_crypto=False, store_devval_path=store_devval_path, vault_path=vault_path
    )

    assert devval_n == 4
    assert vault_n == 1

    # Verify store_devval contains only dev and val dates
    store_devval = ParquetStore(store_devval_path)
    devval_read = store_devval.read("SPY", "1d")
    assert len(devval_read) == 4
    assert pd.Timestamp("2024-01-02") not in devval_read.index

    # Verify vault contains only lockbox date
    vault = ParquetStore(vault_path)
    vault_read = vault.read("SPY", "1d")
    assert len(vault_read) == 1
    assert pd.Timestamp("2024-01-02") in vault_read.index
    assert pd.Timestamp("2019-12-30") not in vault_read.index
