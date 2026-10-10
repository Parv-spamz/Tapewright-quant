import os
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    # tests/unit/test_scaffold.py -> parents: unit -> tests -> project_root
    return Path(__file__).resolve().parent.parent.parent


def test_every_package_dir_has_init() -> None:
    root_path = get_project_root()
    ignored_dir_names = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".pytest_cache",
        ".gemini",
        "logs",
        "scripts",
        "reports_out",
    }

    missing_inits: list[str] = []
    total_packages_checked = 0

    for current_dir, dirs, files in os.walk(root_path):
        rel_dir = Path(current_dir).relative_to(root_path)
        parts = rel_dir.parts

        # Skip root directory itself and any ignored directories
        if rel_dir == Path("."):
            continue
        if any(part in ignored_dir_names or part.startswith(".") for part in parts):
            continue
        # Skip runtime parquet data partitions inside parquet_store
        if "parquet_store" in parts and rel_dir != Path("data/caching/parquet_store"):
            continue

        total_packages_checked += 1
        init_file = Path(current_dir) / "__init__.py"
        if not init_file.is_file():
            missing_inits.append(str(rel_dir).replace("\\", "/"))

    assert total_packages_checked > 0, "No package directories were found to check."
    assert not missing_inits, f"The following directories are missing __init__.py: {missing_inits}"


def test_check_syntax_clean() -> None:
    root_path = get_project_root()
    syntax_script = root_path / "scripts" / "check_syntax.py"
    assert syntax_script.is_file(), f"Syntax script not found at {syntax_script}"

    result = subprocess.run(
        [sys.executable, str(syntax_script)],
        cwd=str(root_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"check_syntax.py failed with return code {result.returncode}.\n"
        f"Stdout: {result.stdout}\n"
        f"Stderr: {result.stderr}"
    )
    assert "0 failed" in result.stdout, f"Syntax failures detected: {result.stdout}"


def test_check_constraints_clean() -> None:
    root_path = get_project_root()
    constraints_script = root_path / "scripts" / "check_constraints.py"
    assert constraints_script.is_file(), f"Constraints script not found at {constraints_script}"

    result = subprocess.run(
        [sys.executable, str(constraints_script)],
        cwd=str(root_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"check_constraints.py failed with return code {result.returncode}.\n"
        f"Stdout: {result.stdout}\n"
        f"Stderr: {result.stderr}"
    )
    assert "0 violations" in result.stdout, f"Constraints violations detected: {result.stdout}"
