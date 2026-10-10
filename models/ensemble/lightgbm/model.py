"""
Regularized LightGBM ensemble wrapper with causal time-ordered early stopping.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from models.ensemble.base import BaseTabularModel


class LightGBMModel(BaseTabularModel):
    """LightGBM tabular classifier with regularized tree defaults and time-ordered validation.

    Parameters
    ----------
    n_estimators : int, default 200
        Maximum number of boosting iterations.
    max_depth : int, default 4
        Maximum tree depth constraint.
    num_leaves : int, default 15
        Maximum tree leaves (controls capacity).
    learning_rate : float, default 0.05
        Shrinkage rate.
    subsample : float, default 0.8
        Row subsampling ratio.
    colsample_bytree : float, default 0.8
        Column subsampling ratio per tree.
    reg_alpha : float, default 0.1
        L1 regularization penalty.
    reg_lambda : float, default 1.0
        L2 regularization penalty.
    early_stopping_rounds : int, default 15
        Validation patience rounds before early stopping.
    val_ratio : float, default 0.15
        Fraction of data reserved for time-ordered validation tail.
    random_state : int, default 42
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        num_leaves: int = 15,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 15,
        val_ratio: float = 0.15,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.extra_kwargs = kwargs

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> LightGBMModel:
        """Fits LightGBM model with time-ordered validation early stopping."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            self.feature_names_ = [f"f{i}" for i in range(X_mat.shape[1])]

        y_vec = np.asarray(y)
        w_vec = np.asarray(sample_weight) if sample_weight is not None else None

        n_samples = len(X_mat)
        use_early_stopping = (
            self.val_ratio > 0.0
            and self.early_stopping_rounds > 0
            and n_samples >= 20
        )

        lgb_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "subsample_freq": 1,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "verbose": -1,
            **self.extra_kwargs,
        }

        if use_early_stopping:
            # TIME-ORDERED validation tail
            n_train = int(np.floor(n_samples * (1.0 - self.val_ratio)))
            n_train = max(n_train, 10)

            X_tr, y_tr = X_mat[:n_train], y_vec[:n_train]
            X_val, y_val = X_mat[n_train:], y_vec[n_train:]
            w_tr = w_vec[:n_train] if w_vec is not None else None

            if len(np.unique(y_val)) > 1:
                self.estimator_ = LGBMClassifier(**lgb_params)
                callbacks = [early_stopping(self.early_stopping_rounds, verbose=False)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.estimator_.fit(
                        X_tr,
                        y_tr,
                        sample_weight=w_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=callbacks,
                    )
                self.is_fitted_ = True
                return self

        # Fallback to standard full fit
        self.estimator_ = LGBMClassifier(**lgb_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.estimator_.fit(X_mat, y_vec, sample_weight=w_vec)

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted_ or self.estimator_ is None:
            raise ValueError("Model is not fitted yet.")
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.estimator_.predict(X_mat)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted_ or self.estimator_ is None:
            raise ValueError("Model is not fitted yet.")
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        proba = self.estimator_.predict_proba(X_mat)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def feature_importance(self) -> pd.Series:
        if not self.is_fitted_ or self.estimator_ is None:
            raise ValueError("Model is not fitted yet.")
        importances = self.estimator_.feature_importances_
        return pd.Series(
            importances, index=self.feature_names_, name="importance"
        ).sort_values(ascending=False)
