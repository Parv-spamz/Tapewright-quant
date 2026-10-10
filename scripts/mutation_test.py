"""AST-based mutation testing framework for quantitative backtesting and research pipelines.

Mutates code ASTs in a temporary copy of the project, evaluates mapped unit tests,
and computes kill rates without modifying the real workspace tree.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path

import random
import shutil
import subprocess
import sys
import tempfile
import time
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrity.ledger import Ledger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mutation_test")

MODULE_MAP: dict[str, dict[str, Any]] = {
    "backtesting/engine/event_loop": {
        "paths": [Path("backtesting/engine/event_loop")],
        "tests": ["tests/unit/test_event_loop.py"],
    },
    "backtesting/execution": {
        "paths": [Path("backtesting/execution")],
        "tests": ["tests/unit/test_execution.py"],
    },
    "backtesting/metrics/risk_measures": {
        "paths": [Path("backtesting/metrics/risk_measures")],
        "tests": ["tests/unit/test_metrics_overfit.py", "tests/unit/test_dsr_units.py"],
    },
    "pipelines/training": {
        "paths": [Path("pipelines/training/labeling.py"), Path("pipelines/training/cv.py")],
        "tests": ["tests/unit/test_labeling_cv.py"],
    },
    "features/econometric/fractional_diff": {
        "paths": [Path("features/econometric/fractional_diff")],
        "tests": ["tests/unit/test_econometrics.py"],
    },
    "features/store/registry": {
        "paths": [Path("features/store/registry")],
        "tests": ["tests/unit/test_labeling_cv.py"],
    },
    "signals/generation": {
        "paths": [Path("signals/generation")],
        "tests": ["tests/unit/test_signals_portfolio.py"],
    },
}


@dataclass(frozen=True)
class MutantCandidate:
    module_name: str
    file_path: str  # relative path
    lineno: int
    col_offset: int
    description: str
    mutant_idx: int


class MutationCollector(ast.NodeVisitor):
    """Identifies all viable AST mutation points in a Python file."""

    def __init__(self) -> None:
        self.mutants: list[tuple[int, int, str]] = []
        self._counter: int = 0

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if isinstance(op, ast.Lt):
                self.mutants.append((self._counter, node.lineno, node.col_offset, "< -> <="))
                self._counter += 1
            elif isinstance(op, ast.LtE):
                self.mutants.append((self._counter, node.lineno, node.col_offset, "<= -> <"))
                self._counter += 1
            elif isinstance(op, ast.Gt):
                self.mutants.append((self._counter, node.lineno, node.col_offset, "> -> >="))
                self._counter += 1
            elif isinstance(op, ast.GtE):
                self.mutants.append((self._counter, node.lineno, node.col_offset, ">= -> >"))
                self._counter += 1
            elif isinstance(op, ast.Eq):
                self.mutants.append((self._counter, node.lineno, node.col_offset, "== -> !="))
                self._counter += 1
            elif isinstance(op, ast.NotEq):
                self.mutants.append((self._counter, node.lineno, node.col_offset, "!= -> =="))
                self._counter += 1
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add):
            self.mutants.append((self._counter, node.lineno, node.col_offset, "+ -> -"))
            self._counter += 1
        elif isinstance(node.op, ast.Sub):
            self.mutants.append((self._counter, node.lineno, node.col_offset, "- -> +"))
            self._counter += 1
        elif isinstance(node.op, ast.Mult):
            self.mutants.append((self._counter, node.lineno, node.col_offset, "* -> /"))
            self._counter += 1
        elif isinstance(node.op, ast.Div):
            self.mutants.append((self._counter, node.lineno, node.col_offset, "/ -> *"))
            self._counter += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, ast.And):
            self.mutants.append((self._counter, node.lineno, node.col_offset, "and -> or"))
            self._counter += 1
        elif isinstance(node.op, ast.Or):
            self.mutants.append((self._counter, node.lineno, node.col_offset, "or -> and"))
            self._counter += 1
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, bool):
            if node.value == 1:
                self.mutants.append((self._counter, node.lineno, node.col_offset, "1 -> 0"))
                self._counter += 1
            elif node.value == 0:
                self.mutants.append((self._counter, node.lineno, node.col_offset, "0 -> 1"))
                self._counter += 1
            elif node.value == 1.0:
                self.mutants.append((self._counter, node.lineno, node.col_offset, "1.0 -> 0.0"))
                self._counter += 1
            elif node.value == 0.0:
                self.mutants.append((self._counter, node.lineno, node.col_offset, "0.0 -> 1.0"))
                self._counter += 1
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self.mutants.append((self._counter, node.lineno, node.col_offset, "remove not"))
            self._counter += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
            if (
                len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 1
            ):
                self.mutants.append((self._counter, node.lineno, node.col_offset, ".shift(1) -> .shift(0)"))
                self._counter += 1
                self.mutants.append((self._counter, node.lineno, node.col_offset, ".shift(1) -> .shift(-1)"))
                self._counter += 1
        self.generic_visit(node)


class MutationApplier(ast.NodeTransformer):
    """Applies a single targeted mutation specified by target_id."""

    def __init__(self, target_id: int) -> None:
        self.target_id = target_id
        self._counter: int = 0

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if isinstance(node.op, ast.Not):
            curr_id = self._counter
            self._counter += 1
            if curr_id == self.target_id:
                return self.visit(node.operand)
        return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        new_ops = []
        for op in node.ops:
            curr_id = self._counter
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                self._counter += 1
                if curr_id == self.target_id:
                    if isinstance(op, ast.Lt):
                        new_ops.append(ast.LtE())
                    elif isinstance(op, ast.LtE):
                        new_ops.append(ast.Lt())
                    elif isinstance(op, ast.Gt):
                        new_ops.append(ast.GtE())
                    elif isinstance(op, ast.GtE):
                        new_ops.append(ast.Gt())
                    elif isinstance(op, ast.Eq):
                        new_ops.append(ast.NotEq())
                    elif isinstance(op, ast.NotEq):
                        new_ops.append(ast.Eq())
                else:
                    new_ops.append(op)
            else:
                new_ops.append(op)
        node.ops = new_ops
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            curr_id = self._counter
            self._counter += 1
            if curr_id == self.target_id:
                if isinstance(node.op, ast.Add):
                    node.op = ast.Sub()
                elif isinstance(node.op, ast.Sub):
                    node.op = ast.Add()
                elif isinstance(node.op, ast.Mult):
                    node.op = ast.Div()
                elif isinstance(node.op, ast.Div):
                    node.op = ast.Mult()
        return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if isinstance(node.op, (ast.And, ast.Or)):
            curr_id = self._counter
            self._counter += 1
            if curr_id == self.target_id:
                if isinstance(node.op, ast.And):
                    node.op = ast.Or()
                elif isinstance(node.op, ast.Or):
                    node.op = ast.And()
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, bool):
            if node.value in (0, 1, 0.0, 1.0):
                curr_id = self._counter
                self._counter += 1
                if curr_id == self.target_id:
                    if node.value == 1:
                        node.value = 0
                    elif node.value == 0:
                        node.value = 1
                    elif node.value == 1.0:
                        node.value = 0.0
                    elif node.value == 0.0:
                        node.value = 1.0
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
            if (
                len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 1
            ):
                curr_id_0 = self._counter
                curr_id_neg1 = self._counter + 1
                self._counter += 2
                if self.target_id == curr_id_0:
                    node.args[0] = ast.Constant(value=0)
                elif self.target_id == curr_id_neg1:
                    node.args[0] = ast.Constant(value=-1)
        return self.generic_visit(node)


def compute_tree_hashes(root: Path) -> dict[str, str]:
    """Compute SHA-256 hashes of all source Python files in root."""
    hashes = {}
    for p in sorted(root.rglob("*.py")):
        # Skip temporary directories or caches
        if any(part in p.parts for part in (".git", "logs", "__pycache__", ".pytest_cache")):
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[str(p.relative_to(root).as_posix())] = h
    return hashes


def ignore_copy_patterns(path: str, names: list[str]) -> set[str]:
    """Filter out non-source directories and heavy stores during temp clone."""
    ignored = set()
    for n in names:
        if n in (".git", "logs", "data", "__pycache__", ".pytest_cache", ".idea", ".vscode"):
            ignored.add(n)
        if "raw_archive" in n or "vault" in n or "store_devval" in n:
            ignored.add(n)
    return ignored


def collect_module_mutants(
    module_name: str,
    root: Path,
    cfg: dict[str, Any],
) -> list[MutantCandidate]:
    """Collect all viable AST mutants for a module."""
    candidates: list[MutantCandidate] = []
    files: list[Path] = []
    for p in cfg["paths"]:
        full_p = root / p
        if full_p.is_file():
            files.append(full_p)
        elif full_p.is_dir():
            files.extend(list(full_p.rglob("*.py")))

    files = [f for f in sorted(files) if f.name != "__init__.py"]

    for f in files:
        rel_path = f.relative_to(root).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not parse %s: %s", rel_path, e)
            continue

        collector = MutationCollector()
        collector.visit(tree)
        for m_id, lineno, col_offset, desc in collector.mutants:
            candidates.append(
                MutantCandidate(
                    module_name=module_name,
                    file_path=rel_path,
                    lineno=lineno,
                    col_offset=col_offset,
                    description=desc,
                    mutant_idx=m_id,
                )
            )

    # Sort deterministically
    candidates.sort(key=lambda c: (c.file_path, c.lineno, c.col_offset, c.description))
    return candidates


def run_mutation_tests(
    modules: list[str] | None = None,
    seed: int = 7,
    max_mutants_per_module: int = 25,
    timeout: int = 120,
    record_ledger: bool = True,
    ledger_path: str = "research/ledger/ledger.jsonl",
) -> dict[str, Any]:
    """Execute mutation tests in an isolated temporary clone."""
    root = Path(".").resolve()
    # 1. Snapshot real tree hashes to verify invariance
    initial_hashes = compute_tree_hashes(root)

    target_modules = modules or list(MODULE_MAP.keys())
    results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        logger.info("Cloning project to temporary sandbox: %s", tmp_root)
        shutil.copytree(root, tmp_root, ignore=ignore_copy_patterns, dirs_exist_ok=True)

        for mod in target_modules:
            if mod not in MODULE_MAP:
                raise ValueError(f"Unknown module '{mod}'. Allowed: {list(MODULE_MAP.keys())}")

            cfg = MODULE_MAP[mod]
            all_candidates = collect_module_mutants(mod, root, cfg)
            logger.info("Module '%s': found %d potential AST mutants", mod, len(all_candidates))

            rng = random.Random(seed)
            if len(all_candidates) > max_mutants_per_module:
                selected_candidates = rng.sample(all_candidates, max_mutants_per_module)
            else:
                selected_candidates = list(all_candidates)

            # Sort selected candidates for consistent test order
            selected_candidates.sort(key=lambda c: (c.file_path, c.lineno, c.col_offset, c.description))

            killed_count = 0
            survived_mutants: list[dict[str, Any]] = []

            for i, cand in enumerate(selected_candidates, 1):
                temp_file = tmp_root / cand.file_path
                real_file = root / cand.file_path

                # Parse and apply targeted mutant
                source_code = real_file.read_text(encoding="utf-8")
                tree = ast.parse(source_code)
                applier = MutationApplier(cand.mutant_idx)
                mutated_tree = applier.visit(tree)
                ast.fix_missing_locations(mutated_tree)
                mutated_code = ast.unparse(mutated_tree)

                # Write mutant to temporary clone
                temp_file.write_text(mutated_code, encoding="utf-8")

                # Run mapped test files with -x -q
                test_cmd = [sys.executable, "-m", "pytest", "-x", "-q", *cfg["tests"]]
                # Also include tests/unit/test_mutation_killers.py if it exists
                killers_test = tmp_root / "tests/unit/test_mutation_killers.py"
                if killers_test.exists():
                    test_cmd.append("tests/unit/test_mutation_killers.py")

                t0 = time.time()
                try:
                    res = subprocess.run(
                        test_cmd,
                        cwd=tmp_root,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    elapsed = time.time() - t0
                    if res.returncode != 0:
                        killed = True
                    else:
                        killed = False
                except subprocess.TimeoutExpired:
                    elapsed = timeout
                    killed = True

                # Restore clean file from real workspace
                shutil.copy2(real_file, temp_file)

                if killed:
                    killed_count += 1
                else:
                    survived_mutants.append(
                        {
                            "file": cand.file_path,
                            "line": cand.lineno,
                            "mutation": cand.description,
                        }
                    )

            total_mutants = len(selected_candidates)
            kill_rate = (killed_count / total_mutants) if total_mutants > 0 else 1.0

            results[mod] = {
                "total": total_mutants,
                "killed": killed_count,
                "survived": len(survived_mutants),
                "kill_rate": kill_rate,
                "survivors": survived_mutants,
            }

            logger.info(
                "[%s] Mutants: %d, Killed: %d, Survived: %d, Kill Rate: %.1f%%",
                mod,
                total_mutants,
                killed_count,
                len(survived_mutants),
                kill_rate * 100.0,
            )

    # 2. Verify that the real tree was never modified
    final_hashes = compute_tree_hashes(root)
    assert (
        initial_hashes == final_hashes
    ), "FATAL: Real workspace files were modified during mutation testing!"
    logger.info("Real workspace hash integrity verified: 100%% invariant.")

    # 3. Append ledger records if requested
    if record_ledger:
        led = Ledger(ledger_path)
        for mod, res in results.items():
            mod_tag = mod.replace("/", ".")
            led.append(
                kind="audit_note",
                status="PASS" if res["kill_rate"] >= 0.80 else "FAIL",
                tag=f"mutation.{mod_tag}",
                payload={
                    "module": mod,
                    "kill_rate": float(res["kill_rate"]),
                    "total": int(res["total"]),
                    "killed": int(res["killed"]),
                    "survived": int(res["survived"]),
                },
            )

    return results


def print_summary_table(results: dict[str, Any]) -> None:
    """Print standard formatted summary table of mutation test results."""
    print("\n" + "=" * 78)
    print(f"{'Module':<40} {'Total':>7} {'Killed':>8} {'Survivors':>10} {'Kill Rate':>10}")
    print("-" * 78)
    for mod, data in results.items():
        print(
            f"{mod:<40} {data['total']:>7} {data['killed']:>8} {data['survived']:>10} {data['kill_rate']*100.0:>9.1f}%"
        )
    print("=" * 78)

    has_survivors = any(data["survivors"] for data in results.values())
    if has_survivors:
        print("\nSurviving Mutants by Module:")
        for mod, data in results.items():
            if data["survivors"]:
                print(f"  [{mod}] ({len(data['survivors'])} survivors):")
                for s in data["survivors"]:
                    print(f"    - {s['file']}:{s['line']} ({s['mutation']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="AST-based mutation testing harness.")
    parser.add_argument("--modules", nargs="*", default=None, help="Modules to mutate")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for mutation selection")
    parser.add_argument("--max-mutants", type=int, default=25, help="Max mutants per module")
    parser.add_argument("--timeout", type=int, default=120, help="Test timeout in seconds")
    parser.add_argument("--no-ledger", action="store_true", help="Do not write to ledger")
    args = parser.parse_args()

    results = run_mutation_tests(
        modules=args.modules,
        seed=args.seed,
        max_mutants_per_module=args.max_mutants,
        timeout=args.timeout,
        record_ledger=not args.no_ledger,
    )
    print_summary_table(results)


if __name__ == "__main__":
    main()
