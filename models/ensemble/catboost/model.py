"""
Regularized CatBoost ensemble wrapper supporting native categorical features and causal early stopping.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from models.ensemble.base import BaseTabularModel


class CatBoostModel(BaseTabularModel):
    """CatBoost tabular classifier with native categorical support and regularized defaults.

    Parameters
    ----------
    iterations : int, default 200
        Maximum number of boosting iterations.
    depth : int, default 4
        Symmetric tree depth.
    learning_rate : float, default 0.05
        Gradient step size.
    l2_leaf_reg : float, default 3.0
        L2 regularization coefficient for leaf values.
    subsample : float, default 0.8
        Dataset subsampling ratio.
    early_stopping_rounds : int, default 15
        Validation patience rounds before early stopping.
    val_ratio : float, default 0.15
        Fraction of data reserved for time-ordered validation tail.
    cat_features : list of str or int, optional
        Categorical feature column names or indices (e.g. sector, exchange, asset_class).
    random_seed : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        iterations: int = 200,
        depth: int = 4,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        early_stopping_rounds: int = 15,
        val_ratio: float = 0.15,
        cat_features: Optional[Sequence[Union[str, int]]] = None,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.val_ratio = val_ratio
        self.cat_features = list(cat_features) if cat_features is not None else None
        self.random_seed = random_seed
        self.extra_kwargs = kwargs

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> CatBoostModel:
        """Fits CatBoost model with time-ordered validation early stopping."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_df = X.copy()
        else:
            X_mat = np.asarray(X)
            self.feature_names_ = [f"f{i}" for i in range(X_mat.shape[1])]
            X_df = pd.DataFrame(X_mat, columns=self.feature_names_)

        y_vec = np.asarray(y)
        w_vec = np.asarray(sample_weight) if sample_weight is not None else None

        n_samples = len(X_df)
        use_early_stopping = (
            self.val_ratio > 0.0
            and self.early_stopping_rounds > 0
            and n_samples >= 20
        )

        cb_params: dict[str, Any] = {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "subsample": self.subsample,
            "random_seed": self.random_seed,
            "allow_writing_files": False,
            "verbose": False,
            **self.extra_kwargs,
        }
        if self.cat_features is not None:
            cb_params["cat_features"] = self.cat_features

        if use_early_stopping:
            # TIME-ORDERED validation tail
            n_train = int(np.floor(n_samples * (1.0 - self.val_ratio)))
            n_train = max(n_train, 10)

            X_tr, y_tr = X_df.iloc[:n_train], y_vec[:n_train]
            X_val, y_val = X_df.iloc[n_train:], y_vec[n_train:]
            w_tr = w_vec[:n_train] if w_vec is not None else None

            if len(np.unique(y_val)) > 1:
                cb_params["early_stopping_rounds"] = self.early_stopping_rounds
                self.estimator_ = CatBoostClassifier(**cb_params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.estimator_.fit(
                        X_tr,
                        y_tr,
                        sample_weight=w_tr,
                        eval_set=(X_val, y_val),
                        verbose=False,
                    )
                self.is_fitted_ = True
                return self

        # Fallback to standard full fit
        self.estimator_ = CatBoostClassifier(**cb_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.estimator_.fit(X_df, y_vec, sample_weight=w_vec, verbose=False)

        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted_ or self.estimator_ is None:
            raise ValueError("Model is not fitted yet.")
        X_eval = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_)
        return self.estimator_.predict(X_eval)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted_ or self.estimator_ is None:
            raise ValueError("Model is not fitted yet.")
        X_eval = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_)
        proba = self.estimator_.predict_proba(X_eval)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def feature_importance(self) -> pd.Series:
        if not self.is_fitted_ or self.estimator_ is None:
            raise ValueError("Model is not fitted yet.")
        importances = self.estimator_.get_feature_importance()
        return pd.Series(
            importances, index=self.feature_names_, name="importance"
        ).sort_values(ascending=False)
