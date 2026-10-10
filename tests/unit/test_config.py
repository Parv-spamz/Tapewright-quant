import os
from datetime import date
from pathlib import Path
import pytest

from config.environments.env_loader import RunMode, get_log_level, get_run_mode, load_env
from config.trading.universe import DEFAULT_UNIVERSE, Asset, get_universe
from config.trading.risk_limits import RiskLimits, check_order_allowed, vol_scaling_factor
from config.trading.calendar import is_trading_day, get_nyse_holidays
from config.database.db_config import DBConfig, retry_with_backoff


class TestEnvLoader:
    def test_load_env_parsing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_content = """
        # This is a comment
        
        TAPEWRIGHT_TEST_KEY_A=alpha
        TAPEWRIGHT_TEST_KEY_B="beta_quoted"
        TAPEWRIGHT_TEST_KEY_C='gamma_single'
        TAPEWRIGHT_TEST_KEY_D=delta with spaces
        """
        env_file = tmp_path / ".env.test"
        env_file.write_text(env_content, encoding="utf-8")

        monkeypatch.delenv("TAPEWRIGHT_TEST_KEY_A", raising=False)
        monkeypatch.delenv("TAPEWRIGHT_TEST_KEY_B", raising=False)
        monkeypatch.setenv("TAPEWRIGHT_PRESET_KEY", "original_value")

        env_file.write_text(
            env_content + "\nTAPEWRIGHT_PRESET_KEY=overwritten_attempt\n",
            encoding="utf-8",
        )

        loaded = load_env(env_file)

        assert loaded["TAPEWRIGHT_TEST_KEY_A"] == "alpha"
        assert loaded["TAPEWRIGHT_TEST_KEY_B"] == "beta_quoted"
        assert loaded["TAPEWRIGHT_TEST_KEY_C"] == "gamma_single"
        assert loaded["TAPEWRIGHT_TEST_KEY_D"] == "delta with spaces"
        assert os.environ["TAPEWRIGHT_TEST_KEY_A"] == "alpha"
        # Must not overwrite existing environment variables
        assert os.environ["TAPEWRIGHT_PRESET_KEY"] == "original_value"

    def test_run_mode_and_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAPEWRIGHT_MODE", raising=False)
        monkeypatch.delenv("TAPEWRIGHT_LOG_LEVEL", raising=False)
        assert get_run_mode() == RunMode.DEV
        assert get_log_level() == "DEBUG"

        monkeypatch.setenv("TAPEWRIGHT_MODE", "PROD")
        assert get_run_mode() == RunMode.PROD
        assert get_log_level() == "INFO"

        monkeypatch.setenv("TAPEWRIGHT_MODE", "TEST")
        assert get_run_mode() == RunMode.TEST
        assert get_log_level() == "WARNING"

        monkeypatch.setenv("TAPEWRIGHT_LOG_LEVEL", "ERROR")
        assert get_log_level() == "ERROR"

        monkeypatch.setenv("TAPEWRIGHT_MODE", "UNKNOWN_MODE")
        with pytest.raises(ValueError, match="Unknown run mode"):
            get_run_mode()


class TestUniverse:
    def test_default_universe_structure(self) -> None:
        assert len(DEFAULT_UNIVERSE) == 10
        symbols = [a.symbol for a in DEFAULT_UNIVERSE]
        expected_symbols = [
            "AAPL", "MSFT", "JPM", "XOM", "JNJ", "SPY",
            "BTC-USD", "ETH-USD",
            "EURUSD", "USDJPY",
        ]
        assert symbols == expected_symbols

    def test_get_universe_filtering(self) -> None:
        all_assets = get_universe()
        assert len(all_assets) == 10

        equities = get_universe("EQUITY")
        assert len(equities) == 6
        assert all(a.asset_class == "EQUITY" for a in equities)

        crypto = get_universe("crypto")
        assert len(crypto) == 2
        assert all(a.asset_class == "CRYPTO" for a in crypto)

        fx = get_universe("FX")
        assert len(fx) == 2
        assert all(a.asset_class == "FX" for a in fx)


class TestRiskLimits:
    def test_risk_limits_validation(self) -> None:
        # Valid default instantiates cleanly
        limits = RiskLimits()
        assert limits.max_position_pct == 0.20

        # Reject max_position_pct > 1.0
        with pytest.raises(ValueError):
            RiskLimits(max_position_pct=1.5)

        # Reject vol_scale_min >= vol_scale_max
        with pytest.raises(ValueError):
            RiskLimits(vol_scale_min=2.0, vol_scale_max=2.0)
        with pytest.raises(ValueError):
            RiskLimits(vol_scale_min=2.5, vol_scale_max=2.0)

        # Reject non-positive parameters
        with pytest.raises(ValueError):
            RiskLimits(target_annual_vol=-0.1)

    def test_vol_scaling_factor(self) -> None:
        limits = RiskLimits(target_annual_vol=0.15, vol_scale_min=0.25, vol_scale_max=2.0)

        # target 0.15 / vol 0.30 = 0.5
        factor_half = vol_scaling_factor(limits, current_vol=0.30)
        assert pytest.approx(factor_half, rel=1e-6) == 0.5

        # vol 0.01 clips at vol_scale_max (15.0 -> 2.0)
        factor_clipped_max = vol_scaling_factor(limits, current_vol=0.01)
        assert factor_clipped_max == 2.0

        # vol 1.00 clips at vol_scale_min (0.15 -> 0.25)
        factor_clipped_min = vol_scaling_factor(limits, current_vol=1.00)
        assert factor_clipped_min == 0.25

        # vol <= 0 raises ValueError
        with pytest.raises(ValueError):
            vol_scaling_factor(limits, current_vol=0.0)
        with pytest.raises(ValueError):
            vol_scaling_factor(limits, current_vol=-0.05)

    def test_check_order_allowed(self) -> None:
        limits = RiskLimits(max_position_pct=0.20, max_leverage=1.0)
        equity = 100_000.0

        # Allowed order: 15,000 / 100,000 = 15%
        allowed, reason = check_order_allowed(limits, current_position_value=0.0, order_value=15_000.0, equity=equity)
        assert allowed is True
        assert "within risk limits" in reason

        # Disallowed order: 25,000 / 100,000 = 25% > 20%
        allowed, reason = check_order_allowed(limits, current_position_value=0.0, order_value=25_000.0, equity=equity)
        assert allowed is False
        assert "exceeds max limit" in reason

        # Negative equity rejected
        allowed, reason = check_order_allowed(limits, current_position_value=0.0, order_value=1_000.0, equity=0.0)
        assert allowed is False
        assert "positive" in reason


class TestCalendar:
    def test_nyse_specific_dates(self) -> None:
        # NOT NYSE days: 2024-07-04, 2024-12-25, 2024-03-29 (Good Friday)
        assert is_trading_day("2024-07-04", calendar="NYSE") is False
        assert is_trading_day("2024-12-25", calendar="NYSE") is False
        assert is_trading_day("2024-03-29", calendar="NYSE") is False

        # IS a trading day: 2024-03-28 (Thursday before Good Friday)
        assert is_trading_day("2024-03-28", calendar="NYSE") is True

        # 2026-07-03 (Fri) is NOT (Jul 4 is Saturday)
        assert is_trading_day("2026-07-03", calendar="NYSE") is False

        # Weekend test
        assert is_trading_day("2024-03-30", calendar="NYSE") is False  # Saturday
        assert is_trading_day("2024-03-31", calendar="NYSE") is False  # Sunday

    def test_crypto_calendar(self) -> None:
        # Crypto calendar True on a Sunday
        assert is_trading_day("2024-03-31", calendar="CRYPTO_24_7") is True
        assert is_trading_day("2024-07-04", calendar="CRYPTO_24_7") is True
        assert is_trading_day("2024-12-25", calendar="CRYPTO_24_7") is True

    def test_new_years_saturday_exception(self) -> None:
        # In 2022, Jan 1 was a Saturday.
        # Rule: Saturday New Year's is NOT observed on prior Friday (Dec 31, 2021).
        assert is_trading_day("2021-12-31", calendar="NYSE") is True


class TestDBConfig:
    def test_db_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAPEWRIGHT_DB_HOST", "db.internal")
        monkeypatch.setenv("TAPEWRIGHT_DB_PORT", "3307")
        monkeypatch.setenv("TAPEWRIGHT_DB_USER", "quant_user")
        monkeypatch.setenv("TAPEWRIGHT_DB_PASSWORD", "secret_pass")
        monkeypatch.setenv("TAPEWRIGHT_DB_NAME", "quant_research")

        cfg = DBConfig.from_env()
        assert cfg.host == "db.internal"
        assert cfg.port == 3307
        assert cfg.user == "quant_user"
        assert cfg.password == "secret_pass"
        assert cfg.database == "quant_research"

    def test_retry_with_backoff_success_after_failures(self) -> None:
        call_count = 0

        def flaky_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary connection failure")
            return "SUCCESS"

        result = retry_with_backoff(flaky_function, max_retries=3, backoff_s=0.0)
        assert result == "SUCCESS"
        assert call_count == 3

    def test_retry_with_backoff_exhaustion(self) -> None:
        call_count = 0

        def failing_function() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Fatal DB failure")

        with pytest.raises(RuntimeError, match="Fatal DB failure"):
            retry_with_backoff(failing_function, max_retries=3, backoff_s=0.0)
        assert call_count == 3
