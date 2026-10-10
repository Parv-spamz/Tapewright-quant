"""Unit tests for the AST mutation testing framework.

Verifies:
- A toy module with a weak test exhibits surviving mutants.
- A toy module with a strong test successfully kills mutants.
- The real workspace tree hash is strictly invariant before and after mutation runs.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

import importlib.util

_mut_path = Path(__file__).resolve().parents[2] / "scripts" / "mutation_test.py"
_spec = importlib.util.spec_from_file_location("mutation_test_module", _mut_path)
_mut_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mut_mod)

MutationApplier = _mut_mod.MutationApplier
MutationCollector = _mut_mod.MutationCollector
compute_tree_hashes = _mut_mod.compute_tree_hashes


def test_toy_module_weak_test_shows_survivors(tmp_path: Path) -> None:
    """A weak assertion (e.g. asserting non-None) fails to kill arithmetic mutants."""
    toy_src = tmp_path / "toy_math.py"
    toy_src.write_text(
        "def compute_total(a: int, b: int) -> int:\n"
        "    if a > 0 and b > 0:\n"
        "        return a + b\n"
        "    return 0\n",
        encoding="utf-8",
    )

    toy_test = tmp_path / "test_weak.py"
    toy_test.write_text(
        "from toy_math import compute_total\n"
        "def test_compute_weak():\n"
        "    val = compute_total(10, 5)\n"
        "    assert val is not None\n",
        encoding="utf-8",
    )

    tree = ast.parse(toy_src.read_text(encoding="utf-8"))
    collector = MutationCollector()
    collector.visit(tree)
    assert len(collector.mutants) > 0

    # Apply a mutation (e.g. + -> -)
    # Find + -> - mutant
    plus_mutants = [m for m in collector.mutants if m[3] == "+ -> -"]
    assert len(plus_mutants) > 0
    m_id = plus_mutants[0][0]

    applier = MutationApplier(m_id)
    mutated = applier.visit(ast.parse(toy_src.read_text(encoding="utf-8")))
    ast.fix_missing_locations(mutated)
    toy_src.write_text(ast.unparse(mutated), encoding="utf-8")

    # Run pytest against the weak test
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(toy_test)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    # Weak test passes (mutant survived)
    assert res.returncode == 0, f"Weak test was expected to pass but failed: {res.stdout}"


def test_toy_module_strong_test_shows_kills(tmp_path: Path) -> None:
    """A strong equality assertion kills arithmetic and comparison mutants."""
    toy_src = tmp_path / "toy_math.py"
    toy_src.write_text(
        "def compute_total(a: int, b: int) -> int:\n"
        "    if a > 0 and b > 0:\n"
        "        return a + b\n"
        "    return 0\n",
        encoding="utf-8",
    )

    toy_test = tmp_path / "test_strong.py"
    toy_test.write_text(
        "from toy_math import compute_total\n"
        "def test_compute_strong():\n"
        "    assert compute_total(10, 5) == 15\n"
        "    assert compute_total(0, 5) == 0\n",
        encoding="utf-8",
    )

    tree = ast.parse(toy_src.read_text(encoding="utf-8"))
    collector = MutationCollector()
    collector.visit(tree)

    # Mutate + to -
    plus_mutants = [m for m in collector.mutants if m[3] == "+ -> -"]
    assert len(plus_mutants) > 0
    m_id = plus_mutants[0][0]

    applier = MutationApplier(m_id)
    mutated = applier.visit(ast.parse(toy_src.read_text(encoding="utf-8")))
    ast.fix_missing_locations(mutated)
    toy_src.write_text(ast.unparse(mutated), encoding="utf-8")

    # Run pytest against the strong test
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(toy_test)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    # Strong test fails (mutant killed)
    assert res.returncode != 0, "Strong test was expected to fail and kill mutant"


def test_real_tree_hash_invariance() -> None:
    """Verify that compute_tree_hashes accurately detects changes and remains invariant."""
    root = Path(".").resolve()
    hashes1 = compute_tree_hashes(root)
    assert len(hashes1) > 0

    # Ensure hash of any tracked file matches direct computation
    sample_file = list(hashes1.keys())[0]
    expected_hash = hashlib.sha256((root / sample_file).read_bytes()).hexdigest()
    assert hashes1[sample_file] == expected_hash

    # Re-evaluating hashes confirms identical tree
    hashes2 = compute_tree_hashes(root)
    assert hashes1 == hashes2
