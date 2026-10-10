import os
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    pool_size: int = 5
    connect_timeout: int = 10
    max_retries: int = 3
    retry_backoff_s: float = 1.0

    @classmethod
    def from_env(cls) -> "DBConfig":
        """Construct DBConfig from environment variables prefixed with TAPEWRIGHT_DB_."""
        host = os.environ.get("TAPEWRIGHT_DB_HOST", "localhost")
        port = int(os.environ.get("TAPEWRIGHT_DB_PORT", "3306"))
        user = os.environ.get("TAPEWRIGHT_DB_USER", "root")
        password = os.environ.get("TAPEWRIGHT_DB_PASSWORD", "")
        database = os.environ.get("TAPEWRIGHT_DB_NAME", "tapewright_db")
        pool_size = int(os.environ.get("TAPEWRIGHT_DB_POOL_SIZE", "5"))
        connect_timeout = int(os.environ.get("TAPEWRIGHT_DB_CONNECT_TIMEOUT", "10"))
        max_retries = int(os.environ.get("TAPEWRIGHT_DB_MAX_RETRIES", "3"))
        retry_backoff_s = float(os.environ.get("TAPEWRIGHT_DB_RETRY_BACKOFF_S", "1.0"))

        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            pool_size=pool_size,
            connect_timeout=connect_timeout,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )


def retry_with_backoff(
    fn: Callable[..., T],
    max_retries: int = 3,
    backoff_s: float = 1.0,
    exceptions: tuple[type[Exception], ...] | type[Exception] = (Exception,),
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute callable `fn` with exponential backoff on specified exceptions.

    Args:
        fn: Callable to execute.
        max_retries: Total number of attempts (must be >= 1).
        backoff_s: Base backoff time in seconds (multiplier is 2^attempt).
        exceptions: Tuple of exception classes to catch and retry.
        *args: Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        Result of fn(*args, **kwargs).

    Raises:
        The last exception encountered if all attempts fail.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    if not isinstance(exceptions, tuple):
        exceptions = (exceptions,)

    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:
            last_exception = exc
            if attempt == max_retries - 1:
                raise
            if backoff_s > 0:
                time.sleep(backoff_s * (2**attempt))

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected execution state in retry_with_backoff")
