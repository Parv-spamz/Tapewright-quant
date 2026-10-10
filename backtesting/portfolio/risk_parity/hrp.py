"""Hierarchical Risk Parity (HRP) and Equal Risk Contribution (ERC) allocation methods."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from backtesting.portfolio.black_litterman.bl import black_litterman
from backtesting.portfolio.markowitz.markowitz import max_sharpe, min_variance
from utils.math.linalg import ledoit_wolf_cov


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Quasi-diagonalize the hierarchical linkage tree to reorder assets along dendrogram leaves."""
    link_int = link.astype(int)
    num_items = link_int[-1, 3]
    sort_ix = pd.Series([link_int[-1, 0], link_int[-1, 1]])

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link_int[j, 0]
        df1 = pd.Series(link_int[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _get_cluster_var(cov: np.ndarray, c_items: list[int]) -> float:
    """Calculate the cluster variance using inverse-variance intra-cluster weights."""
    cov_c = cov[np.ix_(c_items, c_items)]
    diag = np.diag(cov_c)
    inv_diag = 1.0 / np.where(diag > 1e-12, diag, 1e-12)
    w = inv_diag / np.sum(inv_diag)
    return float(w @ cov_c @ w)


def hierarchical_risk_parity(cov: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Calculate Hierarchical Risk Parity (HRP) portfolio weights (Lopez de Prado, 2016).

    Three-stage algorithmic procedure:
        1. Tree Clustering: Computes correlation distance matrix d_ij = sqrt(0.5 * (1 - rho_ij))
           and single-linkage hierarchical dendrogram.
        2. Quasi-Diagonalization: Reorders covariance matrix placing similar clusters adjacent.
        3. Recursive Bisection: Allocates weights top-down using inverse-cluster variance.

    Args:
        cov: Covariance matrix (N x N).

    Returns:
        np.ndarray: Non-negative portfolio weights summing to 1.0.
    """
    sigma = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    N = sigma.shape[0]
    if N <= 1:
        return np.ones(N)

    # 1. Correlation and distance matrix
    diag_std = np.sqrt(np.maximum(np.diag(sigma), 1e-14))
    corr = sigma / np.outer(diag_std, diag_std)
    corr = np.clip(corr, -1.0, 1.0)

    dist = np.sqrt(0.5 * np.clip(1.0 - corr, 0.0, 2.0))
    np.fill_diagonal(dist, 0.0)

    # Single linkage clustering
    dist_condensed = squareform(dist, checks=False)
    link = linkage(dist_condensed, method="single")

    # 2. Quasi-diagonalization
    sort_ix = _get_quasi_diag(link)

    # 3. Recursive bisection
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]

    while len(c_items) > 0:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            c0 = c_items[i]
            c1 = c_items[i + 1]
            v0 = _get_cluster_var(sigma, c0)
            v1 = _get_cluster_var(sigma, c1)
            alpha = 1.0 - v0 / max(v0 + v1, 1e-14)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha

    w_sorted = w.sort_index().values
    return w_sorted / np.sum(w_sorted)


def equal_risk_contribution(cov: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Calculate Equal Risk Contribution (ERC) / Risk Parity portfolio weights via SLSQP.

    Minimizes: sum_{i,j} (RC_i - RC_j)^2
    where RC_i = w_i * (Sigma * w)_i / sqrt(w^T * Sigma * w)
    subject to sum(w) = 1, w_i >= 0.

    Args:
        cov: Asset covariance matrix (N x N).

    Returns:
        np.ndarray: Optimal ERC weights.
    """
    sigma = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    N = sigma.shape[0]
    if N <= 1:
        return np.ones(N)

    # Initial inverse-volatility starting weights
    diag_std = np.sqrt(np.maximum(np.diag(sigma), 1e-14))
    inv_std = 1.0 / diag_std
    w0 = inv_std / np.sum(inv_std)

    def objective(w: np.ndarray) -> float:
        port_vol = np.sqrt(max(float(w @ sigma @ w), 1e-14))
        rc = w * (sigma @ w) / port_vol
        # Scaled pairwise squared differences
        rc_diff = rc[:, None] - rc[None, :]
        return float(np.sum(rc_diff ** 2) * 1e4)

    bounds = [(1e-6, 1.0) for _ in range(N)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-16, "maxiter": 1000},
    )

    w = res.x
    return w / np.sum(w)


def allocate(
    returns_df: pd.DataFrame,
    method: str = "hrp",
    shrink: bool = True,
    **kwargs: Any,
) -> pd.Series:
    """Compute optimal portfolio allocation weights across assets.

    Supported methods:
        - 'hrp': Hierarchical Risk Parity (dendrogram clustering + recursive bisection)
        - 'erc': Equal Risk Contribution (numerical risk parity)
        - 'minvar': Minimum Variance (Markowitz quadratic optimization)
        - 'bl': Black-Litterman model with posterior Sharpe tangency optimization

    When shrink=True (default), always passes returns through Ledoit-Wolf shrinkage
    covariance (`ledoit_wolf_cov`) to suppress estimation error.

    Args:
        returns_df: DataFrame of periodic asset returns (time x assets).
        method: Allocation algorithm ('hrp', 'erc', 'minvar', 'bl').
        shrink: Whether to apply Ledoit-Wolf covariance shrinkage.
        **kwargs: Additional parameters passed to optimization methods.

    Returns:
        pd.Series: Allocated portfolio weights indexed by asset symbols.
    """
    if returns_df.empty:
        return pd.Series([], dtype=float)

    cols = returns_df.columns
    N = len(cols)

    # Compute covariance matrix
    if shrink:
        cov_df = ledoit_wolf_cov(returns_df)
    else:
        cov_df = returns_df.cov()

    method_lower = method.lower()
    if method_lower == "hrp":
        weights = hierarchical_risk_parity(cov_df)
    elif method_lower == "erc":
        weights = equal_risk_contribution(cov_df)
    elif method_lower == "minvar":
        long_only = kwargs.get("long_only", True)
        weights = min_variance(cov_df, long_only=long_only)
    elif method_lower == "bl":
        w_mkt = kwargs.get("w_mkt", np.ones(N) / N)
        P = kwargs.get("P")
        Q = kwargs.get("Q")
        tau = kwargs.get("tau", 0.05)
        delta = kwargs.get("delta", 2.5)
        omega = kwargs.get("omega")
        pi, mu_bl, cov_bl = black_litterman(
            cov=cov_df,
            w_mkt=w_mkt,
            P=P,
            Q=Q,
            tau=tau,
            delta=delta,
            omega=omega,
        )
        weights = max_sharpe(cov_bl, mu_bl, rf=kwargs.get("rf", 0.0))
    else:
        raise ValueError(f"Unknown allocation method: '{method}'. Choose from 'hrp', 'erc', 'minvar', 'bl'.")

    return pd.Series(weights, index=cols, name="weight")
