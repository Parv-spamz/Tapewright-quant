import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def is_psd(A: np.ndarray | pd.DataFrame, tol: float = 1e-8) -> bool:
    """Check whether matrix A is symmetric and positive semi-definite within tolerance."""
    mat = A.values if isinstance(A, pd.DataFrame) else np.asarray(A, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False

    # Check symmetry
    if not np.allclose(mat, mat.T, atol=tol):
        return False

    # Check eigenvalues of symmetrized matrix
    sym_mat = (mat + mat.T) / 2.0
    eigvals = np.linalg.eigvalsh(sym_mat)
    return bool(np.all(eigvals >= -tol))


def nearest_psd(A: np.ndarray | pd.DataFrame, tol: float = 1e-8) -> np.ndarray | pd.DataFrame:
    """Compute the nearest symmetric positive semi-definite matrix via spectral projection.

    Negative eigenvalues are clamped to 0. Symmetrizes before and after projection.
    Preserves pandas DataFrame indices/columns if input is a DataFrame.
    """
    is_df = isinstance(A, pd.DataFrame)
    mat = A.values if is_df else np.asarray(A, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be a 2D square matrix.")

    # 1. Symmetrize
    b = (mat + mat.T) / 2.0

    # 2. Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(b)

    # 3. Project eigenvalues onto non-negative cone
    eigvals_clamped = np.maximum(eigvals, 0.0)

    # 4. Reconstruct
    psd_mat = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

    # 5. Symmetrize again to purge numerical rounding asymmetries
    psd_mat = (psd_mat + psd_mat.T) / 2.0

    if is_df:
        return pd.DataFrame(psd_mat, index=A.index, columns=A.columns)
    return psd_mat


def ledoit_wolf_cov(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate the shrunk covariance matrix using Ledoit-Wolf shrinkage.

    Args:
        returns_df: DataFrame of asset returns (samples x assets).

    Returns:
        pd.DataFrame: Labeled covariance matrix (assets x assets).
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(returns_df)}")
    if returns_df.empty:
        raise ValueError("returns_df cannot be empty.")

    lw = LedoitWolf()
    lw.fit(returns_df.values)
    cov_arr = lw.covariance_

    return pd.DataFrame(cov_arr, index=returns_df.columns, columns=returns_df.columns)


def corr_from_cov(cov: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix.

    Preserves DataFrame indices/columns if input is a DataFrame.
    """
    is_df = isinstance(cov, pd.DataFrame)
    mat = cov.values if is_df else np.asarray(cov, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    diag = np.diag(mat)
    if np.any(diag <= 0):
        raise ValueError("Diagonal variances must be strictly positive.")

    std = np.sqrt(diag)
    outer_std = np.outer(std, std)
    corr = mat / outer_std

    # Clean numerical precision artifacts
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    if is_df:
        return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
    return corr
