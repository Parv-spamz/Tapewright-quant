import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data.caching.parquet_store.store import ParquetStore
from data.ingestion.macroeconomic.fred_rates.fred import (
    inversion_episodes,
    inversion_flag,
    parse_fred_csv,
    yield_curve_spread,
)
from data.ingestion.market_feeds.crypto.ccxt_feed import parse_binance_klines
from data.ingestion.market_feeds.equities.l2_tape import simulate_l2_tape
from data.ingestion.market_feeds.equities.synthetic import make_ohlcv
from data.ingestion.market_feeds.equities.yahoo import FeedError, parse_yahoo_chart
from data.ingestion.market_feeds.fx_forex.fx_feed import parse_fx_rates
from data.ingestion.news_feeds.financial_rss.rss import parse_rss_feed
from data.ingestion.news_feeds.sec_edgar.edgar import parse_edgar_submissions
from data.ingestion.provider import get_prices
from data.transformation.resamplers.bars import (
    dollar_bars,
    make_synthetic_ticks,
    resample_time_bars,
    ticks_to_bars,
    volume_bars,
)


class TestParquetStore:
    def test_parquet_round_trip_and_partitioning(self, tmp_path: Path) -> None:
        store = ParquetStore(tmp_path / "parquet_cache")

        # 1. Multi-year OHLCV DataFrame
        df1 = make_ohlcv("AAPL", start="2023-11-01", n_days=30, seed=42)
        df2 = make_ohlcv("AAPL", start="2024-01-02", n_days=30, seed=43)
        combined_df = pd.concat([df1, df2], axis=0).sort_index()

        store.write("AAPL", combined_df, interval="1d")

        # Verify partition directories exist
        part_2023 = tmp_path / "parquet_cache" / "1d" / "symbol=AAPL" / "year=2023" / "part.parquet"
        part_2024 = tmp_path / "parquet_cache" / "1d" / "symbol=AAPL" / "year=2024" / "part.parquet"
        assert part_2023.exists()
        assert part_2024.exists()

        # Read back and compare
        read_df = store.read("AAPL", interval="1d")
        pd.testing.assert_frame_equal(read_df, combined_df)

        # 2. Overlapping merge de-duplicates (keep last)
        overlap_slice = combined_df.iloc[-5:].copy()
        overlap_slice["close"] = overlap_slice["close"] + 10.0
        store.write("AAPL", overlap_slice, interval="1d")

        merged_read = store.read("AAPL", interval="1d")
        assert len(merged_read) == len(combined_df)  # No duplicates created
        np.testing.assert_allclose(
            merged_read.iloc[-5:]["close"].values,
            overlap_slice["close"].values,
        )

        # 3. L2 Book round trip
        book_df, _ = simulate_l2_tape(n_steps=50, seed=42)
        store.write("AAPL", book_df, interval="l2_book")
        read_book = store.read("AAPL", interval="l2_book")
        pd.testing.assert_frame_equal(read_book, book_df, check_freq=False)

        # 4. Symbol existence and listing
        assert store.exists("AAPL", interval="1d") is True
        assert store.exists("MSFT", interval="1d") is False
        assert store.list_symbols(interval="1d") == ["AAPL"]


class TestBarResamplers:
    def test_ticks_to_time_bars_from_6_hand_ticks(self) -> None:
        # Hand-constructed 6 ticks within a single 1-minute window
        # Prices: [100.0, 105.0, 95.0, 102.0, 98.0, 103.0]
        # Sizes: [10, 20, 15, 25, 30, 50] -> Total Volume = 150
        # Expected OHLCV: Open=100.0, High=105.0, Low=95.0, Close=103.0, Volume=150.0
        timestamps = pd.date_range("2024-01-02 09:30:00", periods=6, freq="8s")
        ticks_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": [100.0, 105.0, 95.0, 102.0, 98.0, 103.0],
                "size": [10.0, 20.0, 15.0, 25.0, 30.0, 50.0],
            }
        )

        bars = ticks_to_bars(ticks_df, rule="1min")
        assert len(bars) == 1
        row = bars.iloc[0]
        assert row["open"] == 100.0
        assert row["high"] == 105.0
        assert row["low"] == 95.0
        assert row["close"] == 103.0
        assert row["volume"] == 150.0

    def test_volume_bars_conservation_and_threshold(self) -> None:
        ticks_df = make_synthetic_ticks(n=300, seed=42)
        threshold = 500.0

        bars = volume_bars(ticks_df, volume_threshold=threshold)
        assert len(bars) > 0

        # Each closed bar must satisfy volume >= threshold
        assert (bars["volume"] >= threshold).all()

        # Total closed bar volume + remainder == total tick volume
        remainder = bars.attrs["remainder"]
        total_bar_vol = bars["volume"].sum()
        total_tick_vol = ticks_df["size"].sum()

        assert pytest.approx(total_bar_vol + remainder, rel=1e-9) == total_tick_vol

    def test_dollar_bars_conservation_and_threshold(self) -> None:
        ticks_df = make_synthetic_ticks(n=300, seed=42)
        threshold = 50_000.0

        bars = dollar_bars(ticks_df, dollar_threshold=threshold)
        assert len(bars) > 0

        # Each closed bar must satisfy dollar_volume >= threshold
        assert (bars["dollar_volume"] >= threshold).all()

        # Total closed bar dollar volume + remainder == total tick dollar volume
        remainder = bars.attrs["remainder"]
        total_bar_dollar = bars["dollar_volume"].sum()
        total_tick_dollar = (ticks_df["price"] * ticks_df["size"]).sum()

        assert pytest.approx(total_bar_dollar + remainder, rel=1e-9) == total_tick_dollar


class TestFeedParsersWithoutNetwork:
    def test_parse_yahoo_chart_hand_payload(self) -> None:
        # Hand-written mock Yahoo chart v8 JSON payload
        hand_payload = json.dumps(
            {
                "chart": {
                    "result": [
                        {
                            "meta": {"currency": "USD", "symbol": "AAPL"},
                            "timestamp": [1704186000, 1704272400],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [185.0, 186.0],
                                        "high": [187.0, 188.0],
                                        "low": [184.0, 185.5],
                                        "close": [186.5, 187.5],
                                        "volume": [50000000, 45000000],
                                    }
                                ]
                            },
                        }
                    ],
                    "error": None,
                }
            }
        )

        df = parse_yahoo_chart(hand_payload)
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.iloc[0]["open"] == 185.0
        assert df.iloc[1]["close"] == 187.5

    def test_parse_binance_klines_hand_payload(self) -> None:
        # Hand-written mock Binance REST klines response payload
        hand_payload = json.dumps(
            [
                [1704153600000, "42000.0", "43000.0", "41500.0", "42500.0", "1250.5", 1704239999999],
                [1704240000000, "42500.0", "44000.0", "42200.0", "43800.0", "1800.2", 1704326399999],
            ]
        )

        df = parse_binance_klines(hand_payload)
        assert len(df) == 2
        assert df.iloc[0]["open"] == 42000.0
        assert df.iloc[1]["volume"] == 1800.2

    def test_parse_fx_rates_hand_payload(self) -> None:
        # Hand-written mock ECB/Frankfurter multi-date FX JSON payload
        hand_payload = json.dumps(
            {
                "amount": 1.0,
                "base": "USD",
                "rates": {
                    "2024-01-02": {"EUR": 0.912, "JPY": 141.5},
                    "2024-01-03": {"EUR": 0.915, "JPY": 142.1},
                },
            }
        )

        df = parse_fx_rates(hand_payload)
        assert len(df) == 2
        assert df.loc["2024-01-02", "EUR"] == 0.912
        assert df.loc["2024-01-03", "JPY"] == 142.1

    def test_parse_edgar_submissions_hand_payload(self) -> None:
        # Hand-written mock SEC EDGAR company submissions JSON payload
        hand_payload = json.dumps(
            {
                "cik": "0000320193",
                "entityType": "operating",
                "sic": "3571",
                "name": "Apple Inc.",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000320193-24-000006", "0000320193-24-000001"],
                        "filingDate": ["2024-02-02", "2024-01-15"],
                        "reportDate": ["2023-12-30", "2024-01-15"],
                        "form": ["10-Q", "8-K"],
                    }
                },
            }
        )

        df = parse_edgar_submissions(hand_payload)
        assert len(df) == 2
        assert "form" in df.columns
        assert df.iloc[0]["form"] == "10-Q"

    def test_parse_rss_feed_hand_payload(self) -> None:
        # Hand-written mock RSS 2.0 XML payload
        hand_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Tapewright Quantitative News</title>
                <link>https://tapewright.org/feed</link>
                <description>Macroeconomic and market signals</description>
                <item>
                    <title>Federal Reserve Signals Rate Plateau</title>
                    <link>https://tapewright.org/news/101</link>
                    <description>FOMC minutes indicate inflation moderation.</description>
                    <pubDate>Tue, 02 Jan 2024 14:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """

        df = parse_rss_feed(hand_xml)
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Federal Reserve Signals Rate Plateau"
        assert df.iloc[0]["published_at"].tzinfo is not None

    def test_parse_fred_csv_hand_payload(self) -> None:
        # Hand-written mock FRED CSV payload with missing data '.' markers
        hand_csv = """DATE,DGS10
2024-01-02,3.94
2024-01-03,.
2024-01-04,3.99
2024-01-05,4.05
"""
        series = parse_fred_csv(hand_csv)
        assert len(series) == 4
        assert series.name == "DGS10"
        assert series.iloc[0] == 3.94
        assert np.isnan(series.iloc[1])  # '.' successfully mapped to NaN
        assert series.iloc[2] == 3.99


class TestYieldCurveAnalysis:
    def test_yield_curve_spread_inversion_and_episodes(self) -> None:
        # Hand-crafted DGS10 and DGS2 series with '.' gaps and negative spread periods
        # Dates: 2024-01-01 through 2024-01-07
        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        dgs10_vals = [4.0, 4.0, np.nan, 3.8, 3.7, 4.2, 4.3]  # Gaps at idx 2
        dgs2_vals = [3.8, 4.2, 4.2, 4.1, 4.0, 3.9, 3.9]

        dgs10 = pd.Series(dgs10_vals, index=dates, name="DGS10")
        dgs2 = pd.Series(dgs2_vals, index=dates, name="DGS2")

        spread = yield_curve_spread(dgs10, dgs2)
        assert len(spread) == 7

        # Day 1: 4.0 - 3.8 = +0.20 (normal)
        # Day 2: 4.0 - 4.2 = -0.20 (inverted)
        # Day 3: (ffilled 4.0) - 4.2 = -0.20 (inverted)
        # Day 4: 3.8 - 4.1 = -0.30 (inverted)
        # Day 5: 3.7 - 4.0 = -0.30 (inverted)
        # Day 6: 4.2 - 3.9 = +0.30 (normal)
        # Day 7: 4.3 - 3.9 = +0.40 (normal)
        np.testing.assert_allclose(
            spread.values,
            [0.2, -0.2, -0.2, -0.3, -0.3, 0.3, 0.4],
            atol=1e-10,
        )

        inv_flags = inversion_flag(spread)
        assert not inv_flags.iloc[0]
        assert inv_flags.iloc[1] and inv_flags.iloc[2] and inv_flags.iloc[3] and inv_flags.iloc[4]
        assert not inv_flags.iloc[5]

        # Episodes table: exactly 1 contiguous episode from Day 2 to Day 5
        episodes = inversion_episodes(spread)
        assert len(episodes) == 1
        ep = episodes.iloc[0]
        assert ep["start"] == dates[1]
        assert ep["end"] == dates[4]
        assert ep["days"] == 4
        assert np.isclose(ep["min_spread"], -0.30)


class TestProviderFallback:
    def test_provider_offline_fallback_to_synthetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock fetch_yahoo_chart to simulate offline network failure
        def mock_failing_fetch(*args, **kwargs):
            raise FeedError("Network unreachable: simulated offline mode")

        monkeypatch.setattr("data.ingestion.provider.fetch_yahoo_chart", mock_failing_fetch)

        df = get_prices("AAPL", start="2023-01-01", end="2023-03-01", source="auto")
        assert not df.empty
        assert df.attrs["source"] == "synthetic"
