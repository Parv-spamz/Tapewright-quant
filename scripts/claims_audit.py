import ast
import importlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    claims_path = project_root / "research/claims.json"
    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)

    test_files = sorted(Path("tests").rglob("*.py"))

    # Parse all test functions that contain an assertion
    # Each entry: (file_path, function_name, ast_nodes)
    test_functions = []
    for tf in test_files:
        try:
            src = tf.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(tf))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(node))
                if has_assert:
                    test_functions.append((tf, node.name, list(ast.walk(node))))

    results = []
    for claim in claims:
        cid = claim["id"]
        text = claim["text"]
        symbol = claim["symbol"]

        if symbol == "NOT_FOUND":
            results.append((cid, text, symbol, "MISSING"))
            continue

        if ":" not in symbol:
            results.append((cid, text, symbol, "MISSING"))
            continue

        mod_name, obj_name = symbol.split(":", 1)
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, obj_name):
                results.append((cid, text, symbol, "MISSING"))
                continue
        except Exception:
            results.append((cid, text, symbol, "MISSING"))
            continue

        # Check whether any test function references obj_name via Name, Attribute, or ImportFrom
        found = False
        for tf, func_name, nodes in test_functions:
            for n in nodes:
                if isinstance(n, ast.Name) and n.id == obj_name:
                    found = True
                    break
                elif isinstance(n, ast.Attribute) and n.attr == obj_name:
                    found = True
                    break
                elif isinstance(n, ast.ImportFrom):
                    for alias in n.names:
                        if alias.name == obj_name or alias.asname == obj_name:
                            found = True
                            break
                    if found:
                        break
            if found:
                break

        status = "PASS" if found else "UNTESTED"
        results.append((cid, text, symbol, status))

    reports_dir = Path("research/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "claims_audit.txt"

    lines = []
    header = (
        f"{'ID':<6} | {'STATUS':<9} | {'SYMBOL':<62} | {'CLAIM'}\n"
        + "=" * 120
    )
    lines.append(header)
    for cid, text, symbol, status in results:
        lines.append(f"{cid:<6} | {status:<9} | {symbol:<62} | {text}")

    output_text = "\n".join(lines) + "\n"
    out_path.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
