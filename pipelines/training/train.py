"""
Institutional tabular training pipeline with Lopez de Prado purged cross-validation,
meta-labeling, sample uniqueness weighting, and out-of-sample permutation importance.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from features.econometric.fractional_diff.ffd import frac_diff_ffd
from features.store.registry.registry import FeatureRegistry, register_default_features
from models.ensemble.base import BaseTabularModel
from pipelines.training.cv import PurgedKFold, assert_no_leakage
from pipelines.training.labeling import (
    cusum_filter,
    daily_vol,
    meta_labels,
    sample_weights,
    triple_barrier,
)


def bet_size(prob: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
    """Calculates discretionary bet size from secondary meta-label probability.

    Formula:
        m = 2 * prob - 1, clipped to [0, 1].

    Rationale:
    In de Prado's meta-labeling framework, a model predicting whether to take a primary trade
    has no conviction at prob <= 0.5 (bet size 0), increasing linearly up to maximum size 1.0
    at prob = 1.0.

    Parameters
    ----------
    prob : pd.Series, np.ndarray, or float
        Predicted probability of taking the trade in [0, 1].

    Returns
    -------
    pd.Series, np.ndarray, or float
        Bet size scalar or array clipped to [0.0, 1.0].
    """
    if isinstance(prob, pd.Series):
        return (2.0 * prob - 1.0).clip(lower=0.0, upper=1.0)
    arr = 2.0 * np.asarray(prob) - 1.0
    return np.clip(arr, 0.0, 1.0)


def build_dataset(
    close_or_panel: Union[pd.Series, pd.DataFrame],
    registry: Optional[FeatureRegistry] = None,
    events_method: str = "cusum",
    pt_sl: tuple[float, float] = (1.0, 1.0),
    horizon: int = 10,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Constructs point-in-time features, causal event labels, and uniqueness sample weights.

    Parameters
    ----------
    close_or_panel : pd.Series or pd.DataFrame
        Asset close price series or OHLCV DataFrame.
    registry : FeatureRegistry, optional
        Feature store registry instance. If None, default technical + FFD registry is used.
    events_method : str, default 'cusum'
        Event filter algorithm ('cusum' or 'all').
    pt_sl : tuple of float, default (1.0, 1.0)
        Profit-taking and stop-loss multiples of target volatility.
    horizon : int, default 10
        Vertical timeout barrier horizon in bars.

    Returns
    -------
    tuple of (X, y, t0, t1, weights)
        X : Feature DataFrame aligned to event timestamps t0.
        y : Binary or ternary triple-barrier target labels.
        t0 : Label start bar timestamps.
        t1 : Label barrier touch timestamps.
        weights : Normalized sample weights based on average uniqueness.
    """
    if isinstance(close_or_panel, pd.Series):
        df = pd.DataFrame(
            {
                "close": close_or_panel,
                "high": close_or_panel,
                "low": close_or_panel,
                "volume": 1000.0,
            }
        )
    else:
        df = close_or_panel.copy()
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 1000.0

    # 1. Feature construction via registry
    if registry is None:
        reg = FeatureRegistry()
        register_default_features(reg)
    else:
        reg = registry

    feat_dict: dict[str, pd.Series] = {}
    for name in reg.catalog()["name"]:
        try:
            res = reg.compute(name, df)
            if isinstance(res, pd.DataFrame):
                for col in res.columns:
                    feat_dict[f"{name}_{col}"] = res[col]
            else:
                feat_dict[name] = res
        except Exception:
            continue

    feat_df = pd.DataFrame(feat_dict, index=df.index)

    # Ensure fractional-differenced price feature is present
    ffd_thresh = 1e-3 if len(df) < 1500 else 1e-4
    feat_df["ffd_04"] = frac_diff_ffd(df["close"], d=0.4, threshold=ffd_thresh).shift(1)

    # Drop columns that are completely NaN (e.g. from insufficient history)
    feat_df = feat_df.dropna(axis=1, how="all")

    # 2. Causal event generation
    v = daily_vol(df["close"], span=50)
    if events_method == "cusum":
        events = cusum_filter(df["close"], threshold=v)
        if len(events) < 15:
            # Fallback to periodic sampling to ensure adequate sample size
            events = df.index[20:-horizon:3]
    else:
        events = df.index[20:-horizon]

    # 3. Triple-barrier labeling
    tb_df = triple_barrier(
        df,
        events=events,
        pt_sl=pt_sl,
        target=v,
        horizon_bars=horizon,
        vertical_label="sign",
    )
    if len(tb_df) == 0:
        raise ValueError("No events generated valid triple-barrier labels.")

    # 4. Sample weights by uniqueness
    weights_ser = sample_weights(tb_df, close_index=df.index, returns=tb_df["ret"])

    # 5. Alignment and dropping NaNs
    valid_t0 = tb_df["t0"].values
    X_sub = feat_df.loc[valid_t0].copy()
    y_sub = tb_df["label"].copy()
    t0_sub = tb_df["t0"].copy()
    t1_sub = tb_df["t1_touch"].copy()

    # Drop rows with NaN features or labels
    valid_mask = X_sub.notna().all(axis=1) & y_sub.notna() & weights_ser.notna()
    X = X_sub.loc[valid_mask].reset_index(drop=True)
    y = y_sub.loc[valid_mask].reset_index(drop=True)
    t0 = t0_sub.loc[valid_mask].reset_index(drop=True)
    t1 = t1_sub.loc[valid_mask].reset_index(drop=True)
    weights = weights_ser.loc[valid_mask].reset_index(drop=True)

    # Normalize weights so sum equals N
    if len(weights) > 0 and weights.sum() > 0:
        weights = weights * (float(len(weights)) / weights.sum())

    return X, y, t0, t1, weights


def train_with_purged_cv(
    model_factory: Callable[[], BaseTabularModel],
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    t0: Union[pd.Series, np.ndarray, Sequence[Any]],
    t1: Union[pd.Series, np.ndarray, Sequence[Any]],
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
    cv: Optional[PurgedKFold] = None,
) -> dict[str, Any]:
    """Trains tabular models across PurgedKFold splits with strict leakage assertion.

    Parameters
    ----------
    model_factory : callable
        Factory returning a fresh BaseTabularModel instance.
    X : pd.DataFrame
        Input features.
    y : pd.Series or np.ndarray
        Classification targets.
    t0 : array-like
        Start timestamps of evaluation windows.
    t1 : array-like
        End / barrier-touch timestamps of evaluation windows.
    weights : array-like, optional
        Per-sample uniqueness weights.
    cv : PurgedKFold, optional
        Purged CV splitter. If None, 5-fold PurgedKFold(embargo_pct=0.01) is used.

    Returns
    -------
    dict
        Dictionary containing fold_metrics, oof_predictions, oof_probabilities,
        mean_auc, mean_accuracy, mean_log_loss, and list of trained models.
    """
    splitter = cv if cv is not None else PurgedKFold(n_splits=5, embargo_pct=0.01)

    t0_arr = np.asarray(t0)
    t1_arr = np.asarray(t1)
    y_arr = np.asarray(y)
    w_arr = np.asarray(weights) if weights is not None else None

    n_samples = len(X)
    oof_predictions = np.full(n_samples, np.nan)
    oof_probabilities = np.full(n_samples, np.nan)

    fold_metrics: list[dict[str, Any]] = []
    models: list[BaseTabularModel] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, t0_arr, t1_arr)):
        # Mandatory leakage gate: verifies zero overlap
        assert_no_leakage(train_idx, test_idx, t0_arr, t1_arr)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        X_tr = X.iloc[train_idx]
        y_tr = y_arr[train_idx]
        w_tr = w_arr[train_idx] if w_arr is not None else None

        X_te = X.iloc[test_idx]
        y_te = y_arr[test_idx]

        model = model_factory()
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        models.append(model)

        preds = model.predict(X_te)
        probas = model.predict_proba(X_te)

        if probas.ndim == 2:
            prob_pos = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
        else:
            prob_pos = probas

        oof_predictions[test_idx] = preds
        oof_probabilities[test_idx] = prob_pos

        # Calculate fold metrics
        acc = float(accuracy_score(y_te, preds))
        try:
            auc = float(roc_auc_score(y_te, prob_pos))
        except Exception:
            auc = 0.5

        try:
            ll = float(log_loss(y_te, probas))
        except Exception:
            ll = 0.693

        fold_metrics.append(
            {
                "fold": fold,
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "auc": auc,
                "accuracy": acc,
                "log_loss": ll,
            }
        )

    aucs = [m["auc"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]
    losses = [m["log_loss"] for m in fold_metrics]

    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    mean_acc = float(np.mean(accs)) if accs else 0.0
    mean_ll = float(np.mean(losses)) if losses else 0.0

    return {
        "fold_metrics": fold_metrics,
        "oof_predictions": pd.Series(oof_predictions, index=X.index, name="oof_pred"),
        "oof_probabilities": pd.Series(oof_probabilities, index=X.index, name="oof_prob"),
        "mean_auc": mean_auc,
        "mean_accuracy": mean_acc,
        "mean_log_loss": mean_ll,
        "models": models,
    }


def meta_label_pipeline(
    primary_side: Union[pd.Series, np.ndarray, Sequence[float], float],
    X: pd.DataFrame,
    triple_barrier_df: pd.DataFrame,
    model_factory: Callable[[], BaseTabularModel],
    cv: Optional[PurgedKFold] = None,
) -> dict[str, Any]:
    """Executes Lopez de Prado meta-labeling pipeline for secondary position sizing.

    Parameters
    ----------
    primary_side : array-like or scalar
        Primary trading model trade directions (+1 long, -1 short).
    X : pd.DataFrame
        Predictive feature set.
    triple_barrier_df : pd.DataFrame
        Triple-barrier evaluation DataFrame with 't0', 't1_touch', and 'ret'.
    model_factory : callable
        Factory producing fresh secondary ensemble models.
    cv : PurgedKFold, optional
        Cross-validator.

    Returns
    -------
    dict
        Dictionary containing cv_results, oof_probabilities, bet_sizes, and oof_auc.
    """
    meta_y = meta_labels(triple_barrier_df, primary_side)
    t0 = triple_barrier_df["t0"].values
    t1 = triple_barrier_df["t1_touch"].values

    weights = sample_weights(triple_barrier_df, close_index=triple_barrier_df["t0"])

    cv_results = train_with_purged_cv(
        model_factory=model_factory,
        X=X,
        y=meta_y,
        t0=t0,
        t1=t1,
        weights=weights,
        cv=cv,
    )

    oof_probs = cv_results["oof_probabilities"]
    sizes = bet_size(oof_probs)

    valid_mask = oof_probs.notna()
    try:
        oof_auc = float(roc_auc_score(meta_y[valid_mask], oof_probs[valid_mask]))
    except Exception:
        oof_auc = cv_results["mean_auc"]

    return {
        "cv_results": cv_results,
        "oof_probabilities": oof_probs,
        "bet_sizes": sizes,
        "oof_auc": oof_auc,
    }


def walk_forward_signals(
    model_factory: Callable[[], BaseTabularModel],
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    t0: Union[pd.Series, np.ndarray],
    t1: Union[pd.Series, np.ndarray],
    n_splits: int = 5,
    min_train_samples: int = 20,
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> dict[str, Any]:
    """Generates walk-forward out-of-sample trading signals with strict purge verification.

    For every prediction block, asserts and records in metadata that:
        max(t1_train) < min(t0_test)

    Parameters
    ----------
    model_factory : callable
        Model generator function.
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target values.
    t0 : array-like
        Start timestamps.
    t1 : array-like
        End touch timestamps.
    n_splits : int, default 5
        Number of walk-forward out-of-sample blocks.
    min_train_samples : int, default 20
        Minimum historical observations required for the initial training block.
    weights : array-like, optional
        Sample weights.

    Returns
    -------
    dict
        Dictionary containing signals, probabilities, and metadata verification records.
    """
    t0_arr = np.asarray(t0)
    t1_arr = np.asarray(t1)
    y_arr = np.asarray(y)
    w_arr = np.asarray(weights) if weights is not None else None

    n = len(X)
    signals = pd.Series(np.nan, index=X.index, name="signal")
    probabilities = pd.Series(np.nan, index=X.index, name="prob")
    metadata: list[dict[str, Any]] = []

    block_bounds = np.linspace(min_train_samples, n, n_splits + 1, dtype=int)

    for k in range(n_splits):
        test_start = block_bounds[k]
        test_end = block_bounds[k + 1]
        test_idx = np.arange(test_start, test_end)

        if len(test_idx) == 0:
            continue

        min_t0_test = np.min(t0_arr[test_idx])

        # Purged training set: strictly samples whose evaluation finished before test block starts
        train_idx = np.where(t1_arr < min_t0_test)[0]

        if len(train_idx) < 10:
            continue

        max_t1_train = np.max(t1_arr[train_idx])

        # Verification assertion: strictly no leakage
        assert max_t1_train < min_t0_test, (
            f"Leakage detected in walk-forward block {k}: max t1 train ({max_t1_train}) >= min t0 test ({min_t0_test})"
        )

        metadata.append(
            {
                "block": k,
                "train_samples": len(train_idx),
                "test_samples": len(test_idx),
                "max_t1_train": max_t1_train,
                "min_t0_test": min_t0_test,
                "is_purged": bool(max_t1_train < min_t0_test),
            }
        )

        model = model_factory()
        w_sub = w_arr[train_idx] if w_arr is not None else None
        model.fit(X.iloc[train_idx], y_arr[train_idx], sample_weight=w_sub)

        preds = model.predict(X.iloc[test_idx])
        probas = model.predict_proba(X.iloc[test_idx])
        prob_pos = probas[:, 1] if probas.ndim == 2 and probas.shape[1] > 1 else probas.flatten()

        signals.iloc[test_idx] = preds
        probabilities.iloc[test_idx] = prob_pos

    return {
        "signals": signals,
        "probabilities": probabilities,
        "metadata": metadata,
    }


def cross_sectional_ranker_dataset(
    panel: pd.DataFrame,
    horizon: int = 1,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[int]]:
    """Builds cross-sectional ranking dataset with per-date normalized rank targets.

    Parameters
    ----------
    panel : pd.DataFrame
        Multi-asset historical panel containing columns ['date', 'symbol', 'close'].
    horizon : int, default 1
        Forward return horizon bars.
    feature_cols : list of str, optional
        Feature column names to include.

    Returns
    -------
    tuple of (X, y_rank, dates, groups)
        X : Feature DataFrame.
        y_rank : Normalized percentile ranks in [0, 1] per date.
        dates : Timestamp series.
        groups : Group sizes per date (for CatBoost / LightGBM ranker objectives).
    """
    df = panel.copy()
    if "date" not in df.columns or "symbol" not in df.columns:
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "date", df.columns[1]: "symbol"}, inplace=True)
        else:
            raise ValueError("Panel DataFrame must have 'date' and 'symbol' columns.")

    # Compute forward returns per symbol
    df["fwd_ret"] = df.groupby("symbol")["close"].shift(-horizon) / df["close"] - 1.0

    # Rank forward returns per date
    df["rank_target"] = df.groupby("date")["fwd_ret"].rank(pct=True)

    df_clean = df.dropna(subset=["rank_target"]).sort_values("date").reset_index(drop=True)

    if feature_cols is None:
        feat_candidates = [c for c in df_clean.columns if c not in ("date", "symbol", "close", "fwd_ret", "rank_target")]
        if not feat_candidates:
            # Generate a baseline return feature if no extra features supplied
            df_clean["ret_1"] = df_clean.groupby("symbol")["close"].pct_change()
            feat_candidates = ["ret_1"]
        feature_cols = feat_candidates

    X = df_clean[feature_cols].fillna(0.0)
    y_rank = df_clean["rank_target"]
    dates = df_clean["date"]
    groups = df_clean.groupby("date", sort=False).size().tolist()

    return X, y_rank, dates, groups


def permutation_importance_oos(
    model: BaseTabularModel,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    n_repeats: int = 5,
    seed: int = 42,
    metric: str = "auc",
) -> pd.Series:
    """Calculates true out-of-sample permutation feature importance.

    Unlike in-sample tree impurity importances which severely overstate noise features,
    OOS permutation measures the drop in holdout performance when each feature's
    values are randomly permuted.

    Parameters
    ----------
    model : BaseTabularModel
        Fitted tabular model.
    X_test : pd.DataFrame
        Holdout test feature set.
    y_test : array-like
        Holdout test target values.
    n_repeats : int, default 5
        Number of permutation shuffles per feature.
    seed : int, default 42
        Random seed for shuffling.
    metric : str, default 'auc'
        Evaluation metric ('auc' or 'accuracy').

    Returns
    -------
    pd.Series
        Feature importances sorted descending.
    """
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_test)

    def _eval(X_eval: pd.DataFrame) -> float:
        if metric == "auc":
            probas = model.predict_proba(X_eval)
            prob_pos = probas[:, 1] if probas.ndim == 2 and probas.shape[1] > 1 else probas.flatten()
            try:
                return float(roc_auc_score(y_arr, prob_pos))
            except Exception:
                return 0.5
        else:
            preds = model.predict(X_eval)
            return float(accuracy_score(y_arr, preds))

    base_score = _eval(X_test)
    importances: dict[str, float] = {}

    for col in X_test.columns:
        drops: list[float] = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_score = _eval(X_perm)
            drops.append(base_score - perm_score)
        importances[col] = float(np.mean(drops))

    return pd.Series(importances, name="oos_importance").sort_values(ascending=False)
