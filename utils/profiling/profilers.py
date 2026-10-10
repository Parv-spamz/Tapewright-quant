import functools
import time
import tracemalloc
from typing import Any, Callable, TypeVar

from utils.logging.logger import get_logger

T = TypeVar("T")


def timed(fn: Callable | None = None, *, logger: Any = None) -> Callable:
    """Decorator to measure and log the execution time of a function in seconds.

    Can be used as `@timed` or `@timed(logger=custom_logger)`.
    """
    def decorator(func: Callable) -> Callable:
        log = logger if logger is not None else get_logger("profiler")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_s = time.perf_counter() - start_time
            log.info(f"{func.__name__} executed in {elapsed_s:.6f}s")
            return result

        return wrapper

    if fn is not None and callable(fn):
        return decorator(fn)
    return decorator


def measure_memory(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, int]:
    """Measure the peak memory consumption of callable `fn` in bytes using tracemalloc.

    Returns:
        tuple[T, int]: (result, peak_bytes)
    """
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()

    try:
        tracemalloc.reset_peak()
    except AttributeError:
        pass

    try:
        result = fn(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    return result, peak
