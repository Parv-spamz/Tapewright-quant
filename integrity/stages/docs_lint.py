"""Verification stage: Documentation linter for metric leakages outside generated blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

from integrity.render import lint_docs

NAME = "docs_lint"


def run(args: argparse.Namespace) -> tuple[str, str]:
    root = Path(".").resolve()
    target_docs: list[Path] = []

    readme = root / "README.md"
    if readme.exists():
        target_docs.append(readme)

    docs_dir = root / "docs"
    if docs_dir.exists():
        for p in docs_dir.glob("*.md"):
            if p.is_file():
                target_docs.append(p)

    if not target_docs:
        return "PASS", "No documentation files to lint"

    violations = lint_docs(target_docs)
    if violations:
        return "FAIL", f"{len(violations)} doc lint violation(s): {violations[0]}"

    return "PASS", f"{len(target_docs)} documentation file(s) passed anti-forgery metric linting"
