"""
End-to-end integration tests for Tapewright-Quant CLI demo, database persistence, and Streamlit app.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from database.connection.manager import ConnectionManager
from pipelines.run_demo import generate_html_dashboard, run_pipeline


class TestDashboardHtmlEscaping:
    """Tests static dashboard.html generation, synthetic banners, and security escaping."""

    def test_synthetic_banner_and_html_escaping(self) -> None:
        """Dashboard HTML must contain 'SYNTHETIC DATA' banner and escape unsafe symbol names."""
        import pandas as pd
        from backtesting.reports.tearsheet.figures import equity_figure

        temp_dir = Path(tempfile.mkdtemp())
        dashboard_path = temp_dir / "dashboard.html"

        symbols = ["AAPL", "<b>x</b>"]
        fig = equity_figure(pd.Series([100.0, 102.0], index=pd.date_range("2023-01-01", periods=2)))
        try:
            generate_html_dashboard(
                symbols=symbols,
                strategy_name="crossover",
                days=100,
                is_synthetic=True,
                metrics={"Sharpe": 1.5},
                stress_df=pd.DataFrame(),
                drift_df=pd.DataFrame(),
                figures=[("Equity Curve", fig)],
                output_path=dashboard_path,
            )

            assert dashboard_path.exists()
            content = dashboard_path.read_text(encoding="utf-8")
            assert "SYNTHETIC DATA" in content
            assert "&lt;b&gt;x&lt;/b&gt;" in content
            # Ensure unescaped injection is not present
            assert "<b>x</b>" not in content
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestStreamlitAppSmoke:
    """Tests Streamlit tearsheet interactive dashboard."""

    def test_streamlit_apptest_or_import_smoke(self) -> None:
        """App smoke test with AppTest.from_file(...): runs without exception and sidebar controls exist.
        If AppTest is unavailable, verifies module import and callable main().
        """
        app_file = (Path(__file__).resolve().parent.parent.parent / "backtesting/reports/tearsheet/app.py").resolve()
        try:
            from streamlit.testing.v1 import AppTest

            at = AppTest.from_file(str(app_file), default_timeout=30).run()
            assert len(at.exception) == 0, f"Streamlit app raised exceptions: {at.exception}"
            assert len(at.sidebar) > 0, "Streamlit app sidebar controls missing"
        except ImportError:
            # Fallback if AppTest unavailable
            import backtesting.reports.tearsheet.app as app_mod

            assert hasattr(app_mod, "main")
            assert callable(app_mod.main)


class TestEndToEndPipelineDemo:
    """Integration test running CLI demo pipeline with realistic costs and checking DB tables."""

    def test_run_demo_e2e(self) -> None:
        """Integration: run_demo on 3 synthetic symbols, 400 days, crossover ->
        tearsheet and dashboard.html exist;
        DB: trade_signals, order_executions, portfolio_snapshots all have rows.
        """
        temp_dir = Path(tempfile.mkdtemp())
        temp_out = temp_dir / "reports_out"
        temp_store = temp_dir / "parquet_store"
        temp_db = temp_dir / "test_tapewright.db"

        try:
            # Run CLI via python -m pipelines.run_demo
            env = os.environ.copy()
            env["TAPEWRIGHT_SQLITE_PATH"] = str(temp_db)
            env["TAPEWRIGHT_DB_BACKEND"] = "sqlite"

            cmd = [
                sys.executable,
                "-m",
                "pipelines.run_demo",
                "--symbols",
                "AAPL",
                "MSFT",
                "SPY",
                "--days",
                "400",
                "--strategy",
                "crossover",
                "--out",
                str(temp_out),
                "--store",
                str(temp_store),
            ]

            res = subprocess.run(cmd, env=env, capture_output=True, text=True)
            assert res.returncode == 0, f"run_demo failed with error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

            # Verify files exist
            dashboard_file = temp_out / "dashboard.html"
            tearsheet_file = temp_out / "tearsheet.txt"

            assert dashboard_file.exists(), "dashboard.html must exist"
            assert tearsheet_file.exists(), "tearsheet.txt must exist"

            html_text = dashboard_file.read_text(encoding="utf-8")
            assert "SYNTHETIC DATA" in html_text

            # Verify SQLite tables have rows
            assert temp_db.exists(), f"Database file {temp_db} must exist"
            conn = sqlite3.connect(str(temp_db))
            cur = conn.cursor()

            cur.execute("SELECT count(*) FROM trade_signals")
            signals_count = cur.fetchone()[0]

            cur.execute("SELECT count(*) FROM order_executions")
            orders_count = cur.fetchone()[0]

            cur.execute("SELECT count(*) FROM portfolio_snapshots")
            snapshots_count = cur.fetchone()[0]

            conn.close()

            assert signals_count > 0, "trade_signals must have rows"
            assert orders_count > 0, "order_executions must have rows"
            assert snapshots_count > 0, "portfolio_snapshots must have rows"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
