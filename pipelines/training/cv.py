"""Purged Cross-Validation and Walk-Forward splitters based on Marcos Lopez de Prado.

Eliminates information leakage from overlapping labels in financial time series.
"""

from __future__ import annotations

from typing import Iterator, Sequence
import numpy as np
import pandas as pd


def assert_no_leakage(
    train_idx: Sequence[int] | np.ndarray,
    test_idx: Sequence[int] | np.ndarray,
    t0: Sequence[Any] | np.ndarray | pd.Series,
    t1: Sequence[Any] | np.ndarray | pd.Series,
) -> None:
    """Assert that no training sample's [t0, t1] evaluation interval overlaps the test fold.

    Raises:
        AssertionError: If any sample in train_idx overlaps with test_idx intervals.
    """
    t0_arr = np.asarray(t0)
    t1_arr = np.asarray(t1)

    train_indices = np.asarray(train_idx, dtype=int)
    test_indices = np.asarray(test_idx, dtype=int)

    if len(train_indices) == 0 or len(test_indices) == 0:
        return

    train_t0 = t0_arr[train_indices]
    train_t1 = t1_arr[train_indices]

    test_min_t0 = np.min(t0_arr[test_indices])
    test_max_t1 = np.max(t1_arr[test_indices])

    # Overlap occurs iff train_t0 <= test_max_t1 AND train_t1 >= test_min_t0
    overlaps = (train_t0 <= test_max_t1) & (train_t1 >= test_min_t0)
    if np.any(overlaps):
        n_leak = int(np.sum(overlaps))
        raise AssertionError(
            f"Data leakage detected! {n_leak} training samples overlap the test evaluation window."
        )


class PurgedKFold:
    """Purged K-Fold Cross-Validator with embargo period for overlapping labels.

    Excludes from the training set any sample whose [t0, t1] interval overlaps
    with the test fold interval, plus an embargo period of embargo_pct * N samples
    immediately following the test fold.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        """Initialize PurgedKFold.

        Args:
            n_splits: Number of cross-validation folds (default 5).
            embargo_pct: Percentage of total sample count to embargo after test fold (default 0.01).
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.embargo_pct = max(0.0, embargo_pct)

    def split(
        self,
        X: Any,
        t0: Sequence[Any] | np.ndarray | pd.Series,
        t1: Sequence[Any] | np.ndarray | pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Args:
            X: Input dataset (used for sample count N).
            t0: Label start timestamps or bar indices for each sample.
            t1: Label end / touch timestamps or bar indices for each sample.

        Yields:
            tuple[np.ndarray, np.ndarray]: (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        t0_arr = np.asarray(t0)
        t1_arr = np.asarray(t1)

        embargo_samples = int(np.ceil(n_samples * self.embargo_pct))

        # Split test indices into n_splits contiguous chunks
        fold_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            test_start_idx = fold_bounds[i]
            test_end_idx = fold_bounds[i + 1]
            test_idx = indices[test_start_idx:test_end_idx]

            if len(test_idx) == 0:
                continue

            test_min_t0 = np.min(t0_arr[test_idx])
            test_max_t1 = np.max(t1_arr[test_idx])

            # Start with all non-test samples
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            # 1. Purge: remove training samples whose [t0, t1] overlaps test window
            overlap_mask = (t0_arr <= test_max_t1) & (t1_arr >= test_min_t0)
            train_mask[overlap_mask] = False

            # 2. Embargo: remove training samples starting within embargo after test_max_t1
            if embargo_samples > 0:
                # Find maximum index in test fold or index where test_max_t1 occurred
                embargo_limit_idx = min(n_samples, test_end_idx + embargo_samples)
                train_mask[test_end_idx:embargo_limit_idx] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx


class WalkForward:
    """Chronological Walk-Forward Cross-Validation with optional expanding window and purging."""

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step: int | None = None,
        expanding: bool = True,
        purge_horizon: int = 0,
    ) -> None:
        """Initialize WalkForward.

        Args:
            train_size: Initial number of samples in training window.
            test_size: Number of samples in out-of-sample testing window.
            step: Step size between splits (defaults to test_size).
            expanding: If True, expands training window; if False, rolls training window.
            purge_horizon: Bars to purge from the end of training set to avoid label overlap.
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step = step or test_size
        self.expanding = expanding
        self.purge_horizon = max(0, purge_horizon)

    def split(self, X: Any) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate chronological train and test index pairs.

        Args:
            X: Input dataset.

        Yields:
            tuple[np.ndarray, np.ndarray]: (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        curr_train_start = 0
        curr_train_end = self.train_size

        while curr_train_end + self.test_size <= n_samples:
            effective_train_end = max(curr_train_start, curr_train_end - self.purge_horizon)
            train_idx = indices[curr_train_start:effective_train_end]
            test_idx = indices[curr_train_end : curr_train_end + self.test_size]

            yield train_idx, test_idx

            if not self.expanding:
                curr_train_start += self.step
            curr_train_end += self.step
