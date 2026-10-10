"""Ledger-backed document renderer and anti-forgery documentation linter."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Sequence

from integrity.ledger import Ledger

METRIC_WORDS = [
    "sharpe",
    "sortino",
    "auc",
    "ic",
    "pbo",
    "dsr",
    "psr",
    "cagr",
    "drawdown",
    "p-value",
    "kurtosis",
    "t-stat",
    "return",
    "volatility",
    "correlation",
]

METRIC_PATTERN = re.compile(r"\b(" + "|".join(METRIC_WORDS) + r")\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?\b\d+(?:\.\d+)?%?\b")

TAG_PLACEHOLDER_PATTERN = re.compile(
    r"\{\{tag:([a-zA-Z0-9_\-\.]+):([a-zA-Z0-9_\-\.]+)(?:\|([^}]+))?\}\}"
)
STATUS_PLACEHOLDER_PATTERN = re.compile(r"\{\{status:([a-zA-Z0-9_\-\.]+)\}\}")


class RenderError(Exception):
    """Raised when document template rendering encounters missing tags, keys, or formatting errors."""


def _get_nested_value(obj: Any, key_path: str) -> Any:
    """Traverse nested dictionary or object by dot-delimited key path."""
    parts = key_path.split(".")
    curr = obj
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        elif hasattr(curr, part):
            curr = getattr(curr, part)
        else:
            raise KeyError(part)
    return curr


def render_template_string(template_str: str, ledger: Ledger) -> str:
    """Render a single template string by resolving placeholders against latest ledger records."""

    def replace_status(match: re.Match) -> str:
        tag = match.group(1)
        rec = ledger.latest(tag)
        if rec is None:
            raise RenderError(f"Tag '{tag}' not found in ledger for status placeholder")
        return str(rec.get("status", "NOT_RUN"))

    def replace_tag(match: re.Match) -> str:
        tag = match.group(1)
        key_path = match.group(2)
        fmt = match.group(3)

        rec = ledger.latest(tag)
        if rec is None:
            raise RenderError(f"Tag '{tag}' not found in ledger for placeholder")

        payload = rec.get("payload", {})
        # If key_path starts with 'payload.', strip it for payload traversal
        effective_path = key_path[8:] if key_path.startswith("payload.") else key_path

        try:
            val = _get_nested_value(payload, effective_path)
        except KeyError:
            # Fallback check against full record
            try:
                val = _get_nested_value(rec, key_path)
            except KeyError as e:
                raise RenderError(
                    f"Key path '{key_path}' not found in record for tag '{tag}'"
                ) from e

        if fmt:
            try:
                return format(val, fmt)
            except Exception as e:
                raise RenderError(f"Failed formatting value '{val}' with spec '{fmt}': {e}") from e
        return str(val)

    # 1. Resolve {{status:TAG}}
    content = STATUS_PLACEHOLDER_PATTERN.sub(replace_status, template_str)

    # 2. Resolve {{tag:TAG:key.path|fmt}}
    content = TAG_PLACEHOLDER_PATTERN.sub(replace_tag, content)

    # Verify no un-rendered placeholders remain
    remaining = re.findall(r"\{\{[^}]+\}\}", content)
    if remaining:
        raise RenderError(f"Unresolved placeholders remain in template: {remaining}")

    return content


def render_all(
    ledger: Ledger,
    templates_dir: Path | str = "docs/templates",
    out_dir: Path | str = "docs",
) -> list[Path]:
    """Render all markdown templates (.md.tpl) into output directory wrapped in GENERATED blocks."""
    tmpl_path = Path(templates_dir).resolve()
    target_dir = Path(out_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not tmpl_path.exists():
        return []

    head_h = ledger.head_hash()
    rendered_files: list[Path] = []

    for tpl_file in tmpl_path.glob("*.md.tpl"):
        if not tpl_file.is_file():
            continue

        base_name = tpl_file.name[:-4]  # Remove .tpl
        out_file = target_dir / base_name

        raw_template = tpl_file.read_text(encoding="utf-8")
        rendered_body = render_template_string(raw_template, ledger)

        final_content = (
            f"<!-- GENERATED from ledger head {head_h}. Do not edit by hand. -->\n"
            f"<!-- GENERATED:BEGIN -->\n"
            f"{rendered_body.strip()}\n"
            f"<!-- GENERATED:END -->\n"
        )

        out_file.write_text(final_content, encoding="utf-8")
        rendered_files.append(out_file)

    return rendered_files


def lint_docs(paths: Sequence[Path | str] | None = None) -> list[str]:
    """Scan README.md and documentation files for hardcoded numbers near metric words outside GENERATED blocks.

    Rules:
    - Fenced code blocks are exempt.
    - Text within valid <!-- GENERATED:BEGIN --> ... <!-- GENERATED:END --> blocks is exempt.
    - Any number within 40 characters of a metric word outside these blocks is flagged.
    """
    if paths is None:
        target_paths: list[Path] = []
        readme = Path("README.md")
        if readme.exists():
            target_paths.append(readme)
        docs_dir = Path("docs")
        if docs_dir.exists():
            target_paths.extend(docs_dir.glob("*.md"))
    else:
        target_paths = [Path(p) for p in paths]

    violations: list[str] = []

    for file_path in target_paths:
        if not file_path.is_file():
            continue

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        in_code_block = False
        in_generated_block = False

        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track fenced code blocks
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            # Track GENERATED blocks
            if "<!-- GENERATED:BEGIN -->" in line:
                in_generated_block = True
                continue
            if "<!-- GENERATED:END -->" in line:
                in_generated_block = False
                continue

            if in_generated_block:
                continue

            # Outside both code blocks and generated blocks: scan for metric word + number proximity
            m_matches = list(METRIC_PATTERN.finditer(line))
            n_matches = list(NUMBER_PATTERN.finditer(line))

            if not m_matches or not n_matches:
                continue

            for m in m_matches:
                m_word = m.group(0)
                m_start, m_end = m.start(), m.end()

                for n in n_matches:
                    n_num = n.group(0)
                    n_start, n_end = n.start(), n.end()

                    # Distance between character spans
                    distance = min(abs(m_start - n_end), abs(n_start - m_end))
                    if distance <= 40:
                        violations.append(
                            f"{file_path}:{line_no}: Hardcoded metric '{m_word}' near number '{n_num}' "
                            f"(distance {distance} <= 40 chars) outside GENERATED block"
                        )
                        break

        if in_generated_block:
            violations.append(f"{file_path}: Unclosed <!-- GENERATED:BEGIN --> block at end of file")

    return violations
