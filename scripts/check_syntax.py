import ast
import os
import sys
from pathlib import Path


def check_syntax(root_dir: str | Path | None = None) -> tuple[int, int]:
    if root_dir is None:
        root_path = Path(__file__).resolve().parent.parent
    else:
        root_path = Path(root_dir).resolve()

    ignored_dirs = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".gemini"}
    checked_count = 0
    failed_count = 0

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = Path(root) / file
            rel_path = file_path.relative_to(root_path)
            checked_count += 1

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source, filename=str(file_path))
            except SyntaxError as e:
                failed_count += 1
                line_no = e.lineno if e.lineno is not None else 0
                print(f"FAILED: {rel_path}:{line_no}: {e.msg}")
            except Exception as e:
                failed_count += 1
                print(f"FAILED: {rel_path}: {e}")

    print(f"{checked_count} files checked, {failed_count} failed")
    return checked_count, failed_count


def main() -> None:
    _, failed = check_syntax()
    if failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
