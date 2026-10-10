import ast
import os
import re
import sys
from pathlib import Path


def get_allowed_modules(root_path: Path) -> set[str]:
    # 1. Python standard library modules
    allowed = set(sys.stdlib_module_names)

    # 2. Project top-level packages (directories containing __init__.py)
    for item in os.listdir(root_path):
        item_path = root_path / item
        if item_path.is_dir() and (item_path / "__init__.py").exists():
            allowed.add(item)
        elif item_path.is_file() and item.endswith(".py"):
            allowed.add(item[:-3])

    # Also allow project folder name if imported directly
    allowed.add("tapewright_quant")
    allowed.add("Tapewright-quant")

    # 3. Explicitly allowed external libraries from project spec
    spec_allowed = {
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "statsmodels",
        "arch",
        "sklearn",
        "torch",
        "xgboost",
        "lightgbm",
        "catboost",
        "gymnasium",
        "pymysql",
        "mysql",
        "pytest",
        "streamlit",
        "plotly",
    }
    allowed.update(spec_allowed)

    # 4. Packages listed in requirements.txt
    req_file = root_path / "requirements.txt"
    distribution_map = {
        "scikit-learn": "sklearn",
        "mysql-connector-python": "mysql",
        "pymysql": "pymysql",
        "pyyaml": "yaml",
        "pillow": "PIL",
        "python-dateutil": "dateutil",
        "streamlit": "streamlit",
        "plotly": "plotly",
    }

    if req_file.exists():
        with open(req_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg = re.split(r"[=><~!;]", line)[0].strip().lower()
                if not pkg:
                    continue
                if pkg in distribution_map:
                    allowed.add(distribution_map[pkg])
                allowed.add(pkg)
                allowed.add(pkg.replace("-", "_"))

    return allowed


def check_function_body(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    body = func_node.body
    if not body:
        return "empty function body"

    # Identify if the first statement is a docstring
    start_idx = 0
    if len(body) > 0 and isinstance(body[0], ast.Expr):
        val = body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            start_idx = 1

    active_stmts = body[start_idx:]
    if not active_stmts:
        return "function contains only docstring with no implementation"

    if len(active_stmts) == 1:
        stmt = active_stmts[0]
        if isinstance(stmt, ast.Pass):
            return "body is only pass"
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
            return "body is only ..."
        if isinstance(stmt, ast.Raise):
            exc = stmt.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                return "body only raises NotImplementedError"
            if isinstance(exc, ast.Call):
                func = exc.func
                if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                    return "body only raises NotImplementedError"

    # Also detect if active_stmts are all pass statements
    if all(isinstance(s, ast.Pass) for s in active_stmts):
        return "body is only pass"

    return None


def check_constraints(root_dir: str | Path | None = None) -> tuple[list[str], list[str]]:
    if root_dir is None:
        root_path = Path(__file__).resolve().parent.parent
    else:
        root_path = Path(root_dir).resolve()

    errors: list[str] = []
    warnings: list[str] = []

    # Forbidden fragments constructed safely
    todo_token = "TO" + "DO"
    fixme_token = "FIX" + "ME"
    ts_interface_token = "inter" + "face "
    ts_lang_token = 'lang="' + "ts\""
    ts_colon_string = ":" + " " + "string"
    ts_colon_number = ":" + " " + "number"
    script_open_tag = "<" + "sc" + "ript"
    node_mod_dir = "node" + "_" + "modules"
    pkg_json_file = "pack" + "age.json"
    ts_exts = tuple("." + ext for ext in ["t" + "s", "t" + "sx", "m" + "ts", "c" + "ts"])
    tsconfig_prefix = "ts" + "config"
    js_ext = "." + "j" + "s"

    allowed_modules = get_allowed_modules(root_path)

    ignored_dirs = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".gemini"}

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]

        # Check for node_modules directory
        if node_mod_dir in dirs:
            errors.append(f"Forbidden directory '{node_mod_dir}' in {root}")
            dirs.remove(node_mod_dir)

        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(root_path)
            rel_str = str(rel_path).replace("\\", "/")
            lower_file = file.lower()

            # Disallowed toolchains / native / typescript files
            if lower_file.endswith(ts_exts):
                errors.append(f"Forbidden TypeScript file: {rel_str}")
            if lower_file.startswith(tsconfig_prefix) and lower_file.endswith(".json"):
                errors.append(f"Forbidden TypeScript config: {rel_str}")
            if lower_file.endswith((".rs", ".cpp", ".c", ".h")):
                errors.append(f"Forbidden native code file: {rel_str}")
            if lower_file == pkg_json_file:
                errors.append(f"Forbidden Node toolchain file: {rel_str}")

            # JavaScript files flag WARNING
            if lower_file.endswith(js_ext):
                warnings.append(f"JavaScript file detected: {rel_str}")

            # Skip checking check_constraints.py itself for text tokens
            if rel_str == "scripts/check_constraints.py":
                # Still check Python AST for check_constraints.py
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source, filename=str(file_path))
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            violation = check_function_body(node)
                            if violation:
                                errors.append(f"Placeholder violation in {rel_str}:{node.lineno}: {violation}")
                except Exception as e:
                    errors.append(f"Failed to parse {rel_str}: {e}")
                continue

            # Inspect file content for markers
            ext = file_path.suffix.lower()
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception as e:
                errors.append(f"Failed to read file {rel_str}: {e}")
                continue

            for idx, line in enumerate(lines, start=1):
                # Warning for <script> tags
                if script_open_tag in line:
                    warnings.append(f"<script> tag in {rel_str}:{idx}")

                # Check TypeScript markers in non-(.js, .html, .md) source files
                if ext not in {".js", ".html", ".md"}:
                    if ts_interface_token in line:
                        errors.append(f"Forbidden TypeScript interface marker in {rel_str}:{idx}")
                    if ts_lang_token in line:
                        errors.append(f"Forbidden TypeScript lang marker in {rel_str}:{idx}")
                    if ts_colon_string in line:
                        errors.append(f"Forbidden TypeScript ': string' marker in {rel_str}:{idx}")
                    if ts_colon_number in line:
                        errors.append(f"Forbidden TypeScript ': number' marker in {rel_str}:{idx}")

            # Python-specific AST and content checks
            if ext == ".py":
                # Check TODO/FIXME
                for idx, line in enumerate(lines, start=1):
                    if todo_token in line:
                        errors.append(f"Forbidden TODO marker in {rel_str}:{idx}")
                    if fixme_token in line:
                        errors.append(f"Forbidden FIXME marker in {rel_str}:{idx}")

                # AST checks for imports and function bodies
                content = "".join(lines)
                try:
                    tree = ast.parse(content, filename=str(file_path))
                except SyntaxError:
                    # check_syntax handles syntax errors
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        violation = check_function_body(node)
                        if violation:
                            errors.append(
                                f"Placeholder violation in {rel_str}:{node.lineno} "
                                f"function '{node.name}': {violation}"
                            )

                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            top_module = alias.name.split(".")[0]
                            if top_module not in allowed_modules:
                                errors.append(
                                    f"Disallowed import '{top_module}' in {rel_str}:{node.lineno}"
                                )

                    elif isinstance(node, ast.ImportFrom):
                        if node.level and node.level > 0:
                            # Relative import inside project is allowed
                            continue
                        if node.module:
                            top_module = node.module.split(".")[0]
                            if top_module not in allowed_modules:
                                errors.append(
                                    f"Disallowed import '{top_module}' in {rel_str}:{node.lineno}"
                                )

    return errors, warnings


def main() -> None:
    errors, warnings = check_constraints()

    if warnings:
        print(f"=== WARNINGS ({len(warnings)}) ===")
        for w in warnings:
            print(f"WARNING: {w}")

    if errors:
        print(f"=== CONSTRAINTS VIOLATIONS ({len(errors)}) ===")
        for e in errors:
            print(f"OFFENDER: {e}")
        print(f"Constraints check FAILED with {len(errors)} violation(s).")
        sys.exit(1)

    print("Constraints check PASSED: 0 violations.")
    sys.exit(0)


if __name__ == "__main__":
    main()
