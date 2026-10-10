"""Verification stage: Offline store rebuild and archive hash verification."""

from __future__ import annotations

import argparse
from pathlib import Path

from research.rebuild import verify_rebuild_from_archive

NAME = "reproduce"


def run(args: argparse.Namespace) -> tuple[str, str]:
    """Run offline rebuild verification without network inside verify_all."""
    root = Path(".").resolve()
    manifest_dir = root / "research" / "manifests"
    archive_dir = root / "research" / "raw_archive"

    if not manifest_dir.exists():
        return "NOT_RUN", "Manifests directory does not exist"
    if not archive_dir.exists():
        return "NOT_RUN", "Raw archive directory does not exist"

    return verify_rebuild_from_archive(manifest_dir=manifest_dir, archive_dir=archive_dir)
