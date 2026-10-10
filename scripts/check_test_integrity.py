#!/usr/bin/env python3
"""Enforce test suite and source code integrity rules R1 through R7.

R1: Every test function must contain at least one assertion (ast.Assert,
    a call starting with/containing 'assert', or pytest.raises/pytest.warns).
R2: No 'assert True', 'assert 1', assert on a constant, or comparison of an
    expression with itself (e.g. assert x == x).
R3: No pytest.skip, skipif, xfail, or importorskip unless listed in tests/SKIP_WHITELIST.txt.
R4: In non-test source: no PYTEST_CURRENT_TEST, unittest.mock, MagicMock, monkeypatch,
    or both 'pytest' and 'sys.modules' in one file.
R5: In non-test source: no except handler that only does pass, continue, or returns a constant,
    unless it re-raises or calls a logger method (warning, error, exception, critical).
R6: In backtesting/reports/tearsheet (app.py, figures.py): no st.metric() with numeric/string literal value.
R7: Under research/: no file other than real_data.py, data_access.py, lockbox.py (and __init__.py)
    that contains 'ParquetStore(', 'read_parquet', or '.parquet'.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


def load_skip_whitelist(whitelist_path: Path) -> set[str]:
    """Load authorized test skip entries from tests/SKIP_WHITELIST.txt."""
    if not whitelist_path.exists():
        return set()
    whitelist: set[str] = set()
    with open(whitelist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if parts:
                whitelist.add(parts[0])
    return whitelist


def is_call_to_logger(call_node: ast.Call) -> bool:
    """Check if an AST call invokes a logger at WARNING or higher."""
    func = call_node.func
    attr_name: str | None = None
    if isinstance(func, ast.Attribute):
        attr_name = func.attr
    elif isinstance(func, ast.Name):
        attr_name = func.id
    return attr_name in {"warning", "error", "exception", "critical"}


def check_r1_test_has_assertions(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if test function has at least one assert or raises/warns context."""
    for node in ast.walk(func_node):
        if node is func_node:
            continue
        # Avoid checking nested inner function definitions as the outer function's assertions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            call_name = ""
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            if "assert" in call_name.lower():
                return True
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                ctx_name = ""
                if isinstance(ctx, ast.Call):
                    if isinstance(ctx.func, ast.Name):
                        ctx_name = ctx.func.id
                    elif isinstance(ctx.func, ast.Attribute):
                        ctx_name = ctx.func.attr
                if ctx_name in {"raises", "warns"}:
                    return True
    return False


def check_r2_assert_tautology(assert_node: ast.Assert) -> str | None:
    """Return error message if assert is a constant or self-comparison, else None."""
    test = assert_node.test
    if isinstance(test, ast.Constant):
        return f"assert constant literal {test.value!r}"
    if isinstance(test, ast.Compare):
        left_str = ast.unparse(test.left)
        # Check comparing expression to itself, e.g. x == x or len(x) >= len(x)
        for comp in test.comparators:
            if ast.unparse(comp) == left_str:
                return f"comparison of expression with itself: {ast.unparse(test)}"
        # Check comparing two constants, e.g. 1 == 1
        if isinstance(test.left, ast.Constant) and all(isinstance(c, ast.Constant) for c in test.comparators):
            return f"constant comparison: {ast.unparse(test)}"
    return None


def check_r3_skip(
    node: ast.AST,
    rel_path: str,
    class_name: str | None,
    whitelist: set[str],
) -> str | None:
    """Check if a test decorator or call introduces an un-whitelisted skip/xfail."""
    skip_tokens = {"skip", "skipif", "xfail", "importorskip"}
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        for dec in node.decorator_list:
            dec_str = ast.unparse(dec)
            if any(token in dec_str for token in skip_tokens):
                target_ident = f"{rel_path}::{node.name}"
                if class_name:
                    target_ident_with_class = f"{rel_path}::{class_name}::{node.name}"
                else:
                    target_ident_with_class = target_ident
                is_whitelisted = any(
                    target_ident == w or target_ident_with_class == w or w.startswith(target_ident)
                    for w in whitelist
                )
                if not is_whitelisted:
                    return f"{rel_path}:{node.lineno} R3: un-whitelisted skip/xfail decorator: {dec_str}"

    if isinstance(node, ast.Call):
        call_str = ast.unparse(node.func)
        if any(call_str.endswith("." + token) or call_str == token for token in skip_tokens):
            target_file_matches = any(w.startswith(rel_path) for w in whitelist)
            if not target_file_matches:
                return f"{rel_path}:{node.lineno} R3: un-whitelisted skip/xfail call: {call_str}"
    return None


def check_r5_except_handler(handler: ast.ExceptHandler, rel_path: str) -> str | None:
    """Check if except handler swallows errors without logging at WARNING+ or re-raising."""
    # Check if re-raises or calls logger
    for sub in ast.walk(handler):
        if isinstance(sub, ast.Raise):
            return None
        if isinstance(sub, ast.Call) and is_call_to_logger(sub):
            return None

    # Filter out docstrings from body
    active_stmts = [
        s for s in handler.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
    ]
    if not active_stmts:
        return f"{rel_path}:{handler.lineno} R5: empty except handler"

    only_trivial = True
    for s in active_stmts:
        if isinstance(s, (ast.Pass, ast.Continue)):
            continue
        if isinstance(s, ast.Return):
            if s.value is None or isinstance(s.value, ast.Constant):
                continue
            only_trivial = False
            break
        only_trivial = False
        break

    if only_trivial:
        return f"{rel_path}:{handler.lineno} R5: except handler only passes, continues, or returns constant"
    return None


def check_test_integrity(root_dir: str | Path | None = None) -> list[str]:
    """Execute all test integrity rules R1 through R7 and return list of violations."""
    if root_dir is None:
        root_path = Path(".").resolve()
    else:
        root_path = Path(root_dir).resolve()

    violations: list[str] = []
    whitelist = load_skip_whitelist(root_path / "tests" / "SKIP_WHITELIST.txt")

    ignored_dirs = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".gemini", "build", "dist", "logs"}

    all_py_files: list[Path] = []
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
        for file in files:
            if file.endswith(".py"):
                all_py_files.append(Path(root) / file)

    test_files = [p for p in all_py_files if "tests" in p.relative_to(root_path).parts]
    source_files = [
        p for p in all_py_files
        if "tests" not in p.relative_to(root_path).parts
        and "scripts" not in p.relative_to(root_path).parts
    ]

    # --- Rules R1, R2, R3 on test files ---
    for fpath in test_files:
        rel_path = fpath.relative_to(root_path).as_posix()
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=rel_path)
        except Exception as e:
            violations.append(f"{rel_path}:1 Parse error: {e}")
            continue

        # Check R3 on classes
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                r3_err = check_r3_skip(node, rel_path, None, whitelist)
                if r3_err:
                    violations.append(r3_err)

        for node in ast.walk(tree):
            # Check R3 on function definitions and calls
            if isinstance(node, ast.FunctionDef):
                # find if inside class
                r3_err = check_r3_skip(node, rel_path, None, whitelist)
                if r3_err:
                    violations.append(r3_err)

                # Check R1 on test functions
                if node.name.startswith("test_"):
                    if not check_r1_test_has_assertions(node):
                        violations.append(
                            f"{rel_path}:{node.lineno} R1: test function '{node.name}' has no assertions or raises/warns"
                        )

            elif isinstance(node, ast.Call):
                r3_err = check_r3_skip(node, rel_path, None, whitelist)
                if r3_err:
                    violations.append(r3_err)

            # Check R2 on assert statements
            elif isinstance(node, ast.Assert):
                r2_err = check_r2_assert_tautology(node)
                if r2_err:
                    violations.append(f"{rel_path}:{node.lineno} R2: {r2_err}")

    # --- Rules R4, R5 on non-test source files ---
    r4_forbidden_tokens = ["PYTEST_CURRENT_TEST", "unittest.mock", "MagicMock", "monkeypatch"]
    for fpath in source_files:
        rel_path = fpath.relative_to(root_path).as_posix()
        text = fpath.read_text(encoding="utf-8")

        # R4 checks
        for token in r4_forbidden_tokens:
            if token in text:
                # Find line number
                for idx, line in enumerate(text.splitlines(), start=1):
                    if token in line:
                        violations.append(f"{rel_path}:{idx} R4: forbidden test mock / detection token '{token}'")
                        break

        if "pytest" in text and "sys.modules" in text:
            violations.append(f"{rel_path}:1 R4: both 'pytest' and 'sys.modules' present in source file")

        # R5 checks
        try:
            tree = ast.parse(text, filename=rel_path)
        except Exception as e:
            violations.append(f"{rel_path}:1 Parse error: {e}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                r5_err = check_r5_except_handler(node, rel_path)
                if r5_err:
                    violations.append(r5_err)

    # --- Rule R6 on tearsheet app.py and figures.py ---
    tearsheet_files = [
        root_path / "backtesting" / "reports" / "tearsheet" / "app.py",
        root_path / "backtesting" / "reports" / "tearsheet" / "figures.py",
    ]
    for fpath in tearsheet_files:
        if fpath.exists():
            rel_path = fpath.relative_to(root_path).as_posix()
            try:
                tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=rel_path)
            except Exception as e:
                violations.append(f"{rel_path}:1 Parse error: {e}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    call_func = node.func
                    is_st_metric = False
                    if isinstance(call_func, ast.Attribute) and call_func.attr == "metric":
                        is_st_metric = True
                    elif isinstance(call_func, ast.Name) and call_func.id == "metric":
                        is_st_metric = True
                    if is_st_metric:
                        # Determine value argument
                        val_node = None
                        if len(node.args) >= 2:
                            val_node = node.args[1]
                        for kw in node.keywords:
                            if kw.arg == "value":
                                val_node = kw.value
                        if val_node and isinstance(val_node, ast.Constant):
                            violations.append(f"{rel_path}:{node.lineno} R6: st.metric called with literal value {val_node.value!r}")

    # --- Rule R7 under research/ ---
    research_dir = root_path / "research"
    allowed_r7 = {"real_data.py", "data_access.py", "lockbox.py", "__init__.py"}
    if research_dir.exists():
        for fpath in research_dir.rglob("*.py"):
            if fpath.name not in allowed_r7:
                rel_path = fpath.relative_to(root_path).as_posix()
                content = fpath.read_text(encoding="utf-8")
                for token in ["ParquetStore(", "read_parquet", ".parquet"]:
                    if token in content:
                        violations.append(f"{rel_path}:1 R7: unauthorized parquet access under research/ ('{token}')")

    return sorted(violations)


def main() -> int:
    target_root = sys.argv[1] if len(sys.argv) > 1 else "."
    violations = check_test_integrity(target_root)
    if violations:
        for v in violations:
            print(v)
        print(f"\ncheck_test_integrity FAILED: {len(violations)} violation(s) found.")
        return 1
    print("check_test_integrity PASSED: 0 violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
