"""Dynamic hedge ratio estimation using a state-space Kalman Filter.

Tracks time-varying beta and alpha for pairs trading and cointegrated spreads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class KalmanHedgeRatio:
    """Online, state-space Kalman Filter for dynamic hedge ratio and spread estimation.

    State vector: theta_t = [beta_t, alpha_t]^T
    Observation equation: y_t = x_t * beta_t + alpha_t + v_t, v_t ~ N(0, V_e)
    Transition equation: theta_t = theta_{t-1} + w_t, w_t ~ N(0, V_w)
    where V_w = delta / (1 - delta) * I_2.
    """

    def __init__(self, delta: float = 1e-4, ve: float = 1e-3) -> None:
        """Initialize Kalman Filter hyperparameters.

        Args:
            delta: Transition variance parameter in [0, 1) controlling parameter drift rate.
            ve: Observation error variance.
        """
        self.delta = delta
        self.ve = ve

    def fit(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Fit online Kalman Filter sequentially over the input series (x, y).

        Predict / Update cycle:
            R_t = P_{t-1} + V_w
            e_t = y_t - F_t * theta_{t-1}
            Q_t = F_t * R_t * F_t^T + V_e
            K_t = R_t * F_t^T / Q_t
            theta_t = theta_{t-1} + K_t * e_t
            P_t = R_t - K_t * (F_t * R_t)

        Args:
            x: Independent asset price series.
            y: Dependent asset price series.

        Returns:
            pd.DataFrame: Contains columns ['beta', 'alpha', 'spread', 'spread_std'].
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n = len(x_arr)

        if n == 0:
            return pd.DataFrame(columns=["beta", "alpha", "spread", "spread_std"])

        vw = (self.delta / (1.0 - self.delta)) * np.eye(2)
        p_cov = np.zeros((2, 2), dtype=float)
        theta = np.zeros(2, dtype=float)

        betas = np.zeros(n, dtype=float)
        alphas = np.zeros(n, dtype=float)
        spreads = np.zeros(n, dtype=float)
        spread_stds = np.zeros(n, dtype=float)

        for t in range(n):
            f_mat = np.array([x_arr[t], 1.0], dtype=float)

            # Predict state covariance
            r_cov = p_cov + vw

            # Measurement prediction error (spread)
            y_hat = float(np.dot(f_mat, theta))
            e_t = y_arr[t] - y_hat

            # Innovation variance
            q_t = float(np.dot(f_mat, np.dot(r_cov, f_mat)) + self.ve)

            # Kalman gain
            k_gain = np.dot(r_cov, f_mat) / q_t

            # Update state estimate
            theta = theta + k_gain * e_t

            # Update state covariance: P = R - K * (F @ R)
            p_cov = r_cov - np.outer(k_gain, np.dot(f_mat, r_cov))

            betas[t] = theta[0]
            alphas[t] = theta[1]
            spreads[t] = e_t
            spread_stds[t] = np.sqrt(q_t)

        index = x.index if isinstance(x, pd.Series) else None
        return pd.DataFrame(
            {
                "beta": betas,
                "alpha": alphas,
                "spread": spreads,
                "spread_std": spread_stds,
            },
            index=index,
        )
