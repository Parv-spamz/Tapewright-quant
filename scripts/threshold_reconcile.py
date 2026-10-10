import ast
import json
import os
from pathlib import Path


def extract_numeric_literals(node: ast.AST) -> list[float | int]:
    literals = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant):
            if isinstance(sub.value, (int, float)) and not isinstance(sub.value, bool):
                literals.append(sub.value)
    return literals


def evaluate_verdict(row_op: str, row_val, literals: list[float | int]) -> str:
    if row_op in (">", ">="):
        if len(literals) == 1:
            lit = literals[0]
            if lit < row_val:
                return "LOOSER"
            return "STRICTER_OR_EQUAL"
        return "READ_BY_HUMAN"
    elif row_op == "<":
        if len(literals) == 1:
            lit = literals[0]
            if lit > row_val:
                return "LOOSER"
            return "STRICTER_OR_EQUAL"
        return "READ_BY_HUMAN"
    elif row_op == "range":
        if len(literals) == 2 and isinstance(row_val, (list, tuple)) and len(row_val) == 2:
            lower, upper = sorted(literals)
            spec_lower, spec_upper = row_val
            if lower < spec_lower or upper > spec_upper:
                return "LOOSER"
            return "STRICTER_OR_EQUAL"
        return "READ_BY_HUMAN"
    return "READ_BY_HUMAN"


def main() -> None:
    spec_path = Path("research/threshold_spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        threshold_specs = json.load(f)

    tests_dir = Path("tests")
    test_files = sorted(tests_dir.rglob("*.py"))

    # Map of (file_path, class_name, func_name, node, source)
    functions = []
    for tfile in test_files:
        src = tfile.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src, filename=str(tfile))
        except Exception:
            continue

        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append((tfile, None, item.name, item, src))
            elif isinstance(item, ast.ClassDef):
                for method in item.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions.append((tfile, item.name, method.name, method, src))

    reports_dir = Path("research/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "threshold_reconciliation.txt"

    lines = []
    header = (
        f"{'ROW ID':<20} | {'SPEC':<22} | {'VERDICT':<18} | {'MATCHED NODE ID'}\n"
        + "=" * 120
    )
    lines.append(header)

    for row in threshold_specs:
        row_id = row["id"]
        keywords = [k.lower() for k in row["keywords"]]
        op = row["op"]
        val = row["value"]
        spec_str = f"{op} {val}"

        matches = []
        for tfile, cls_name, func_name, node, src in functions:
            fname_lower = func_name.lower()
            node_id = (
                f"{tfile.as_posix()}::{cls_name}::{func_name}"
                if cls_name
                else f"{tfile.as_posix()}::{func_name}"
            )
            if all(k in fname_lower for k in keywords):
                matches.append((node_id, node, src))

        # If not matched by function name alone, fallback to matching across node_id (e.g. test_overfit_audit)
        if not matches:
            for tfile, cls_name, func_name, node, src in functions:
                fname_lower = func_name.lower()
                node_id = (
                    f"{tfile.as_posix()}::{cls_name}::{func_name}"
                    if cls_name
                    else f"{tfile.as_posix()}::{func_name}"
                )
                if all(k in node_id.lower() for k in keywords):
                    matches.append((node_id, node, src))

        if not matches:
            lines.append(f"{row_id:<20} | {spec_str:<22} | {'NOT_FOUND':<18} | NONE")
            continue

        for node_id, node, src in matches:
            assert_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
            assert_nodes.sort(key=lambda x: getattr(x, "lineno", 0))

            all_literals = []
            assert_lines = []
            for a_node in assert_nodes:
                seg = ast.get_source_segment(src, a_node) or ""
                seg_clean = " ".join(seg.split())
                lits = extract_numeric_literals(a_node)
                all_literals.extend(lits)
                assert_lines.append((a_node.lineno, seg_clean))

            verdict = evaluate_verdict(op, val, all_literals)
            lines.append(f"{row_id:<20} | {spec_str:<22} | {verdict:<18} | {node_id}")
            for lno, a_src in assert_lines:
                lines.append(f"    assert [line {lno}]: {a_src}")

    output_text = "\n".join(lines) + "\n"
    out_path.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
