"""Black-Litterman portfolio optimization model combining market equilibrium with subjective views."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def black_litterman(
    cov: pd.DataFrame | np.ndarray,
    w_mkt: pd.Series | np.ndarray,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    tau: float = 0.05,
    delta: float = 2.5,
    omega: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Black-Litterman implied equilibrium, posterior returns, and posterior covariance.

    Formulas:
        1. Implied equilibrium returns:
           pi = delta * Sigma * w_mkt
        2. View uncertainty covariance (He & Litterman specification):
           Omega = diag(P * (tau * Sigma) * P^T)  (if omega is None)
        3. Posterior expected returns:
           M = [(tau * Sigma)^-1 + P^T * Omega^-1 * P]^-1
           mu_BL = M * [(tau * Sigma)^-1 * pi + P^T * Omega^-1 * Q]
        4. Posterior covariance:
           Sigma_BL = Sigma + M

    Args:
        cov: Asset covariance matrix Sigma (N x N).
        w_mkt: Benchmark/market capitalization weights vector (N,).
        P: View pick matrix (K x N), mapping views to asset portfolios.
        Q: View expected returns vector (K,).
        tau: Scalar tuning weight of prior equilibrium distribution (default 0.05).
        delta: Risk aversion parameter (default 2.5).
        omega: Optional diagonal view uncertainty matrix (K x K).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (pi, mu_BL, Sigma_BL)
    """
    sigma = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    w = w_mkt.values if isinstance(w_mkt, pd.Series) else np.asarray(w_mkt, dtype=float)
    N = sigma.shape[0]

    # 1. Implied equilibrium returns
    pi = float(delta) * (sigma @ w)

    # If no views supplied, posterior reduces strictly to equilibrium
    if P is None or Q is None or len(P) == 0:
        mu_bl = pi.copy()
        sigma_bl = sigma * (1.0 + float(tau))
        return pi, mu_bl, sigma_bl

    P_mat = np.asarray(P, dtype=float)
    Q_vec = np.asarray(Q, dtype=float)

    if P_mat.ndim == 1:
        P_mat = P_mat.reshape(1, -1)
    if Q_vec.ndim == 0:
        Q_vec = np.array([float(Q_vec)])

    tau_sigma = float(tau) * sigma

    # 2. View covariance Omega
    if omega is None:
        omega_mat = np.diag(np.diag(P_mat @ tau_sigma @ P_mat.T))
    else:
        omega_mat = np.asarray(omega, dtype=float)
        if omega_mat.ndim == 1:
            omega_mat = np.diag(omega_mat)

    # Inverses
    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega_mat)

    # 3. Posterior variance and expected returns
    M = np.linalg.inv(inv_tau_sigma + P_mat.T @ inv_omega @ P_mat)
    mu_bl = M @ (inv_tau_sigma @ pi + P_mat.T @ inv_omega @ Q_vec)

    # 4. Posterior asset covariance
    sigma_bl = sigma + M

    return pi, mu_bl, sigma_bl
