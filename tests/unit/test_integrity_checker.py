import importlib.util
from pathlib import Path
import pytest

_script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "check_test_integrity.py"
_spec = importlib.util.spec_from_file_location("check_test_integrity_mod", _script_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
check_test_integrity = _mod.check_test_integrity


def test_clean_tree_passes(tmp_path: Path) -> None:
    """Verifies that a compliant tree produces 0 violations."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "SKIP_WHITELIST.txt").write_text("", encoding="utf-8")
    (tests_dir / "test_valid.py").write_text(
        "def test_math():\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    src_file = tmp_path / "valid_module.py"
    src_file.write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def run():\n"
        "    try:\n"
        "        x = 1 / 0\n"
        "    except ZeroDivisionError as e:\n"
        "        logger.warning('Div zero: %s', e)\n",
        encoding="utf-8",
    )

    violations = check_test_integrity(tmp_path)
    assert violations == []


def test_r1_violation_flagged(tmp_path: Path) -> None:
    """R1: Test function with no assertion or raises/warns must be flagged."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "SKIP_WHITELIST.txt").write_text("", encoding="utf-8")
    (tests_dir / "test_no_assert.py").write_text(
        "def test_does_nothing():\n    x = 42\n",
        encoding="utf-8",
    )

    violations = check_test_integrity(tmp_path)
    assert any("R1: test function 'test_does_nothing' has no assertions" in v for v in violations)


def test_r2_violation_flagged(tmp_path: Path) -> None:
    """R2: Constant assertion or self-comparison must be flagged."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "SKIP_WHITELIST.txt").write_text("", encoding="utf-8")
    (tests_dir / "test_tautology.py").write_text(
        "def test_tautology():\n    assert True\n",
        encoding="utf-8",
    )

    violations = check_test_integrity(tmp_path)
    assert any("R2: assert constant literal True" in v for v in violations)


def test_r3_violation_flagged(tmp_path: Path) -> None:
    """R3: Un-whitelisted skip decorator or call must be flagged."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "SKIP_WHITELIST.txt").write_text("", encoding="utf-8")
    (tests_dir / "test_skip.py").write_text(
        "import pytest\n"
        "@pytest.mark.skip(reason='arbitrary')\n"
        "def test_skipped():\n"
        "    assert 1 == 1\n",
        encoding="utf-8",
    )

    violations = check_test_integrity(tmp_path)
    assert any("R3: un-whitelisted skip/xfail decorator" in v for v in violations)


def test_r4_violation_flagged(tmp_path: Path) -> None:
    """R4: Non-test source importing mock or detecting test runners must be flagged."""
    src_file = tmp_path / "bad_mock.py"
    src_file.write_text(
        "from unittest.mock import MagicMock\n",
        encoding="utf-8",
    )

    violations = check_test_integrity(tmp_path)
    assert any("R4: forbidden test mock" in v for v in violations)


def test_r5_violation_flagged(tmp_path: Path) -> None:
    """R5: Except handler that swallows exceptions without warning+ log or re-raise must be flagged."""
    src_file = tmp_path / "swallow.py"
    src_file.write_text(
        "def compute():\n"
        "    try:\n"
        "        x = 1\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )

    violations = check_test_integrity(tmp_path)
    assert any("R5: except handler only passes" in v for v in violations)


def test_r6_violation_flagged(tmp_path: Path) -> None:
    """R6: st.metric literal value in tearsheet app must be flagged."""
    tearsheet_dir = tmp_path / "backtesting" / "reports" / "tearsheet"
    tearsheet_dir.mkdir(parents=True)
    (tearsheet_dir / "app.py").write_text(
        "import streamlit as st\n"
        "st.metric('Sharpe', 1.8)\n",
        encoding="utf-8",
    )

    violations = check_test_integrity(tmp_path)
    assert any("R6: st.metric called with literal value" in v for v in violations)


def test_r7_violation_flagged(tmp_path: Path) -> None:
    """R7: Unauthorized parquet access under research/ must be flagged."""
    research_dir = tmp_path / "research"
    research_dir.mkdir(parents=True)
    (research_dir / "unauthorized.py").write_text(
        "import pandas as pd\n"
        "df = pd.read_parquet('data.parquet')\n",
        encoding="utf-8",
    )

    violations = check_test_integrity(tmp_path)
    assert any("R7: unauthorized parquet access under research/" in v for v in violations)
