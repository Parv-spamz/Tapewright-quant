"""Tapewright Trust and Integrity Package."""

from __future__ import annotations

from integrity.ledger import Ledger, code_hash
from integrity.provenance import (
    DatasetManifest,
    ProvenanceError,
    archive_raw,
    load_raw,
    looks_synthetic_gbm,
    parsed_hash,
    realism_fingerprint,
    require_real,
    verify_manifest,
)
from integrity.render import (
    RenderError,
    lint_docs,
    render_all,
    render_template_string,
)

__all__ = [
    "DatasetManifest",
    "Ledger",
    "ProvenanceError",
    "RenderError",
    "archive_raw",
    "code_hash",
    "lint_docs",
    "load_raw",
    "looks_synthetic_gbm",
    "parsed_hash",
    "realism_fingerprint",
    "render_all",
    "render_template_string",
    "require_real",
    "verify_manifest",
]
