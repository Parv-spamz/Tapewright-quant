import os
from enum import Enum
from pathlib import Path


class RunMode(str, Enum):
    DEV = "DEV"
    PROD = "PROD"
    TEST = "TEST"


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """Parse a .env file by hand and merge into os.environ without overwriting existing vars.

    Skips blank lines and lines starting with '#'. Strips surrounding quotes.
    Returns a dictionary of key-value pairs loaded from the file.
    """
    env_file = Path(path)
    loaded: dict[str, str] = {}

    if not env_file.exists():
        return loaded

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Strip matching outer quotes
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                if len(value) >= 2:
                    value = value[1:-1]

            loaded[key] = value
            if key not in os.environ:
                os.environ[key] = value

    return loaded


def get_run_mode() -> RunMode:
    """Retrieve the current RunMode from TAPEWRIGHT_MODE environment variable.

    Defaults to RunMode.DEV if not set or empty.
    Raises ValueError for unknown run modes.
    """
    mode_str = os.environ.get("TAPEWRIGHT_MODE", "").strip().upper()
    if not mode_str:
        return RunMode.DEV

    try:
        return RunMode(mode_str)
    except ValueError:
        valid_modes = [m.value for m in RunMode]
        raise ValueError(
            f"Unknown run mode '{mode_str}'. Must be one of: {valid_modes}"
        )


def get_log_level() -> str:
    """Determine the active logging level.

    Priority:
    1. TAPEWRIGHT_LOG_LEVEL if explicitly defined.
    2. Derived from RunMode: DEV -> DEBUG, PROD -> INFO, TEST -> WARNING.
    """
    explicit_level = os.environ.get("TAPEWRIGHT_LOG_LEVEL", "").strip().upper()
    if explicit_level:
        return explicit_level

    mode = get_run_mode()
    mode_level_map = {
        RunMode.DEV: "DEBUG",
        RunMode.PROD: "INFO",
        RunMode.TEST: "WARNING",
    }
    return mode_level_map[mode]
