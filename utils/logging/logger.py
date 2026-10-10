import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.environments.env_loader import get_log_level

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "tapewright") -> logging.Logger:
    """Return a configured logger with console and rotating file handlers.

    Idempotent: Subsequent calls for the same logger name will not add duplicate handlers.
    Output format: time | level | name | message
    """
    logger = logging.getLogger(name)

    # Determine numeric log level
    level_name = get_log_level()
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers
    has_stream_handler = any(
        type(h) is logging.StreamHandler
        or (isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler))
        for h in logger.handlers
    )
    has_file_handler = any(
        isinstance(h, RotatingFileHandler) for h in logger.handlers
    )

    formatter = logging.Formatter(LOG_FORMAT)

    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_level)
        logger.addHandler(stream_handler)

    if not has_file_handler:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "tapewright.log"

        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid duplicate log outputs
    logger.propagate = False

    return logger
