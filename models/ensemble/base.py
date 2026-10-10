"""
Abstract base class for tabular supervised learning models in Tapewright-Quant.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


class BaseTabularModel:
    """Base specification for all tabular ensemble learners.

    Guarantees standard API across XGBoost, LightGBM, and CatBoost models:
    - fit(X, y, sample_weight=None)
    - predict(X)
    - predict_proba(X)
    - feature_importance()
    - save(path) / load(path)
    """

    def __init__(self) -> None:
        self.estimator_: Any = None
        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> BaseTabularModel:
        """Fit the model to training data with optional sample weights."""
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate point class or continuous predictions."""
        n = len(X)
        return np.zeros(n, dtype=int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate predicted class probabilities (2D array of shape [n_samples, n_classes])."""
        n = len(X)
        return np.full((n, 2), 0.5, dtype=float)

    def feature_importance(self) -> pd.Series:
        """Return model feature importances as a pandas Series sorted descending."""
        return pd.Series(dtype=float, name="importance")

    def save(self, path: str) -> None:
        """Serialize the fitted model instance to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> BaseTabularModel:
        """Deserialize a saved model instance from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
