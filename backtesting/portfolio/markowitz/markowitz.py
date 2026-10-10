"""Markowitz Modern Portfolio Theory (MPT) mean-variance optimization.

NOTE: Markowitz portfolio optimization is notoriously sensitive to input estimation noise.
Unconstrained mean-variance optimization on raw sample covariances frequently produces extreme,
unstable, and highly leveraged weights ("error maximizers", Michaud 1989).
Always feed a shrunk covariance matrix (e.g. Ledoit-Wolf shrinkage via `ledoit_wolf_cov`)
to ensure numerical stability and out-of-sample robustness.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def min_variance(
    cov: pd.DataFrame | np.ndarray,
    target_return: float | None = None,
    expected_returns: pd.Series | np.ndarray | None = None,
    long_only: bool = True,
) -> np.ndarray:
    """Find the global or target-constrained minimum variance portfolio weights.

    Minimizes: w^T * Sigma * w
    Subject to: sum(w) = 1, and optionally w^T * mu >= target_return.
    If long_only=True: w_i >= 0.

    Args:
        cov: Covariance matrix (N x N).
        target_return: Minimum portfolio expected return constraint (optional).
        expected_returns: Expected asset returns vector (required if target_return is set).
        long_only: If True, imposes non-negativity bounds (default True).

    Returns:
        np.ndarray: Optimal portfolio weights summing to 1.0.

    Raises:
        RuntimeError: If optimization fails to converge.
    """
    sigma = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    N = sigma.shape[0]

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        if expected_returns is None:
            raise ValueError("expected_returns must be provided when target_return is specified.")
        mu = expected_returns.values if isinstance(expected_returns, pd.Series) else np.asarray(expected_returns, dtype=float)
        constraints.append({"type": "ineq", "fun": lambda w: (w @ mu) - target_return})

    bounds = [(0.0, 1.0) for _ in range(N)] if long_only else [(-1.0, 2.0) for _ in range(N)]
    w0 = np.ones(N) / N

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not res.success:
        raise RuntimeError(f"SLSQP min_variance optimization failed to converge: {res.message}")

    return res.x


def max_sharpe(
    cov: pd.DataFrame | np.ndarray,
    expected_returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    long_only: bool = True,
) -> np.ndarray:
    """Find the tangency portfolio weights maximizing the Sharpe Ratio.

    Maximizes: (w^T * mu - rf) / sqrt(w^T * Sigma * w)
    Subject to: sum(w) = 1.
    If long_only=True: w_i >= 0.

    Args:
        cov: Covariance matrix (N x N).
        expected_returns: Expected return vector (N,).
        rf: Risk-free rate.
        long_only: If True, long-only non-negative weights (default True).

    Returns:
        np.ndarray: Optimal Tangency portfolio weights.

    Raises:
        RuntimeError: If optimization fails to converge.
    """
    sigma = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    mu = expected_returns.values if isinstance(expected_returns, pd.Series) else np.asarray(expected_returns, dtype=float)
    N = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        port_return = float(w @ mu)
        port_vol = np.sqrt(max(float(w @ sigma @ w), 1e-14))
        return -(port_return - rf) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(N)] if long_only else [(-1.0, 2.0) for _ in range(N)]
    w0 = np.ones(N) / N

    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not res.success:
        raise RuntimeError(f"SLSQP max_sharpe optimization failed to converge: {res.message}")

    return res.x


def efficient_frontier(
    cov: pd.DataFrame | np.ndarray,
    expected_returns: pd.Series | np.ndarray,
    n_points: int = 50,
    long_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Mean-Variance Efficient Frontier trace.

    Args:
        cov: Covariance matrix (N x N).
        expected_returns: Vector of expected asset returns (N,).
        n_points: Number of discrete frontier points to compute.
        long_only: If True, enforce non-negative weights.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (target_returns, frontier_volatilities, frontier_weights)
    """
    mu = expected_returns.values if isinstance(expected_returns, pd.Series) else np.asarray(expected_returns, dtype=float)
    sigma = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)

    # Global minimum variance return
    w_min = min_variance(sigma, long_only=long_only)
    min_ret = float(w_min @ mu)
    max_ret = float(np.max(mu)) if long_only else float(min_ret + 2.0 * np.std(mu))

    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier_vols = []
    frontier_weights = []

    for r_target in target_returns:
        try:
            w = min_variance(
                sigma,
                target_return=float(r_target),
                expected_returns=mu,
                long_only=long_only,
            )
            v = np.sqrt(max(float(w @ sigma @ w), 0.0))
            frontier_vols.append(v)
            frontier_weights.append(w)
        except RuntimeError as e:
            logger.warning("Optimization failed for target return %f: %s", float(r_target), e)
            continue

    return (
        np.array(target_returns[:len(frontier_vols)]),
        np.array(frontier_vols),
        np.array(frontier_weights),
    )
