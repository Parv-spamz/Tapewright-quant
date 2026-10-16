"""
Tapewright-Quant Production Orchestrator and Live Execution Engine.

Usage:
    python -m pipelines.run_pipeline --symbols AAPL MSFT SPY --days 250 --strategy crossover --source real --out reports_out/
"""

from __future__ import annotations

import sys
from pipelines.run_demo import generate_html_dashboard, init_database, parse_args, run_pipeline

__all__ = [
    "parse_args",
    "init_database",
    "generate_html_dashboard",
    "run_pipeline",
    "main",
]


def main() -> None:
    args_list = sys.argv[1:]
    if "--source" not in args_list:
        args_list = ["--source", "real"] + args_list
    args = parse_args(args_list)
    run_pipeline(
        symbols=args.symbols,
        days=args.days,
        strategy=args.strategy,
        source=args.source,
        out_dir=args.out,
        store_dir=args.store,
    )


if __name__ == "__main__":
    main()
