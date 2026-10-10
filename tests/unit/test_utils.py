import logging
from datetime import date, datetime, timezone
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import pytest

from utils.dates.trading_dates import (
    add_business_days,
    business_days_between,
    to_utc,
    trading_days_between,
)
from utils.logging.logger import get_logger
from utils.math.linalg import corr_from_cov, is_psd, ledoit_wolf_cov, nearest_psd
from utils.profiling.profilers import measure_memory, timed


class TestLogger:
    def test_get_logger_idempotency(self) -> None:
        logger_name = "test_idempotent_logger"
        logger1 = get_logger(logger_name)
        logger2 = get_logger(logger_name)

        assert logger1 is logger2

        # Check exactly 1 console StreamHandler and 1 RotatingFileHandler
        stream_handlers = [
            h for h in logger1.handlers
            if type(h) is logging.StreamHandler or (isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler))
        ]
        file_handlers = [h for h in logger1.handlers if isinstance(h, RotatingFileHandler)]

        assert len(stream_handlers) == 1
        assert len(file_handlers) == 1


class TestLinalgMath:
    def test_nearest_psd_on_non_psd_matrix(self) -> None:
        # Non-PSD symmetric matrix with eigenvalues 3 and -1
        non_psd = np.array([[1.0, 2.0], [2.0, 1.0]])

        assert is_psd(non_psd) is False

        psd_result = nearest_psd(non_psd)

        # Output must pass is_psd
        assert is_psd(psd_result) is True

        # Independent closed-form verification:
        # Spectral projection of [[1, 2], [2, 1]] onto PSD cone analytically equals [[1.5, 1.5], [1.5, 1.5]]
        expected_closed_form = np.array([[1.5, 1.5], [1.5, 1.5]])
        np.testing.assert_allclose(psd_result, expected_closed_form, atol=1e-8)

    def test_corr_from_cov_closed_form(self) -> None:
        # Hand-computed case:
        # Var1 = 4 (std1 = 2), Var2 = 9 (std2 = 3), Cov12 = 3
        # Corr12 = 3 / (2 * 3) = 0.5
        cov = np.array([[4.0, 3.0], [3.0, 9.0]])
        corr = corr_from_cov(cov)

        expected_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        np.testing.assert_allclose(corr, expected_corr, atol=1e-10)

        # DataFrame input preservation
        cov_df = pd.DataFrame(cov, index=["A", "B"], columns=["A", "B"])
        corr_df = corr_from_cov(cov_df)
        assert isinstance(corr_df, pd.DataFrame)
        assert list(corr_df.columns) == ["A", "B"]
        np.testing.assert_allclose(corr_df.values, expected_corr, atol=1e-10)

    def test_ledoit_wolf_cov_properties(self) -> None:
        # Fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)
        n_samples, n_assets = 100, 3
        returns_data = rng.normal(loc=0.001, scale=0.02, size=(n_samples, n_assets))
        returns_df = pd.DataFrame(returns_data, columns=["AAPL", "MSFT", "GOOG"])

        cov_df = ledoit_wolf_cov(returns_df)

        assert isinstance(cov_df, pd.DataFrame)
        assert list(cov_df.columns) == ["AAPL", "MSFT", "GOOG"]
        assert list(cov_df.index) == ["AAPL", "MSFT", "GOOG"]
        assert is_psd(cov_df) is True


class TestTradingDates:
    def test_to_utc_conversion(self) -> None:
        # Naive datetime
        dt_naive = datetime(2024, 1, 1, 12, 0, 0)
        dt_utc = to_utc(dt_naive)
        assert dt_utc.tzinfo == timezone.utc
        assert dt_utc.hour == 12

        # ISO string
        iso_str = "2024-06-15T09:30:00Z"
        dt_from_str = to_utc(iso_str)
        assert dt_from_str.tzinfo == timezone.utc
        assert dt_from_str.year == 2024
        assert dt_from_str.month == 6

        # Unix timestamp in seconds
        epoch_ts = 1704067200.0  # 2024-01-01 00:00:00 UTC
        dt_from_epoch = to_utc(epoch_ts)
        assert dt_from_epoch.year == 2024
        assert dt_from_epoch.month == 1
        assert dt_from_epoch.day == 1

    def test_business_days_between_and_add(self) -> None:
        # Monday to Friday: exactly 4 days half-open, 5 days closed
        mon = date(2024, 1, 8)
        fri = date(2024, 1, 12)

        assert business_days_between(mon, fri, include_end=False) == 4
        assert business_days_between(mon, fri, include_end=True) == 5

        # Adding business days
        assert add_business_days(mon, 4) == fri
        # Adding 1 business day to Friday skips Saturday & Sunday to next Monday (Jan 15)
        assert add_business_days(fri, 1) == date(2024, 1, 15)
        # Stepping backwards
        assert add_business_days(date(2024, 1, 15), -1) == fri

    def test_trading_days_between_with_nyse_holidays(self) -> None:
        # 2024-03-27 (Wed) to 2024-04-02 (Tue)
        # Wed (trading), Thu (trading), Fri (Good Friday holiday - NOT trading),
        # Sat/Sun (weekend), Mon 2024-04-01 (trading).
        # Half-open [Wed, Tue): Wed, Thu, Mon = 3 trading days
        start = date(2024, 3, 27)
        end = date(2024, 4, 2)
        count = trading_days_between(start, end, calendar="NYSE", include_end=False)
        assert count == 3

    def test_no_lookahead_business_days_expansion(self) -> None:
        # NO-LOOKAHEAD test: generating sequence of 200 dates must match
        # first 200 dates when generating 300 dates
        start = date(2020, 1, 6)
        dates_200 = [add_business_days(start, i) for i in range(200)]
        dates_300 = [add_business_days(start, i) for i in range(300)]

        assert dates_200 == dates_300[:200]


class TestProfiling:
    def test_timed_decorator(self) -> None:
        @timed
        def sample_computation(n: int) -> int:
            return sum(range(n))

        result = sample_computation(10_000)
        assert result == sum(range(10_000))

    def test_measure_memory(self) -> None:
        def allocate_array() -> int:
            arr = np.ones(50_000, dtype=np.float64)
            return len(arr)

        res, peak = measure_memory(allocate_array)
        assert res == 50_000
        assert peak > 0
