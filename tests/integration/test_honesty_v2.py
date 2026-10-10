"""Honest multi-seed purge tests over random-walk prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from pipelines.training.cv import PurgedKFold
from pipelines.training.labeling import daily_vol, triple_barrier


def test_honest_multi_seed_purge_vs_leaky() -> None:
    """Over 30 seeds of fresh random-walk prices, evaluate PurgedKFold vs plain KFold."""
    n_seeds = 30
    purged_aucs: list[float] = []
    plain_aucs: list[float] = []

    for seed in range(1, n_seeds + 1):
        np.random.seed(seed)
        n = 600
        ret = np.random.normal(0, 0.01, n)
        price = pd.Series(
            100.0 * np.exp(np.cumsum(ret)),
            index=pd.date_range("2024-01-01", periods=n),
        )
        v = daily_vol(price, span=50)

        # 5 lagged-return features
        feat = pd.DataFrame(
            {f"lag_{i}": price.pct_change(i).shift(1) for i in range(1, 6)},
            index=price.index,
        )

        # Build triple-barrier labels with overlapping horizon 10
        tb = triple_barrier(
            price,
            events=price.index[10:],
            pt_sl=(1.0, 1.0),
            target=v,
            horizon_bars=10,
            vertical_label="sign",
        )
        tb = tb[tb["label"] != 0].copy().dropna()

        X = feat.loc[tb["t0"]].fillna(0.0)
        y = (tb["label"] == 1).astype(int).values
        t0 = tb["t0"].values
        t1 = tb["t1_touch"].values

        # PurgedKFold(5, embargo 0.01)
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
        p_scores: list[float] = []
        for tr, te in pkf.split(X, t0=t0, t1=t1):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            clf = LogisticRegression(max_iter=200, solver="liblinear")
            clf.fit(X.iloc[tr], y[tr])
            prob = clf.predict_proba(X.iloc[te])[:, 1]
            p_scores.append(roc_auc_score(y[te], prob))
        purged_aucs.append(float(np.mean(p_scores)))

        # Plain KFold(5, shuffle=False)
        plain_kf = KFold(n_splits=5, shuffle=False)
        pl_scores: list[float] = []
        for tr, te in plain_kf.split(X):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            clf = LogisticRegression(max_iter=200, solver="liblinear")
            clf.fit(X.iloc[tr], y[tr])
            prob = clf.predict_proba(X.iloc[te])[:, 1]
            pl_scores.append(roc_auc_score(y[te], prob))
        plain_aucs.append(float(np.mean(pl_scores)))

    # Compute distribution statistics
    p_mean = float(np.mean(purged_aucs))
    p_std = float(np.std(purged_aucs, ddof=1))
    p_min = float(np.min(purged_aucs))
    p_max = float(np.max(purged_aucs))

    pl_mean = float(np.mean(plain_aucs))
    pl_std = float(np.std(plain_aucs, ddof=1))
    pl_min = float(np.min(plain_aucs))
    pl_max = float(np.max(plain_aucs))

    diff = pl_mean - p_mean
    stat, pval = ttest_rel(plain_aucs, purged_aucs)

    print("\n--- HONEST MULTI-SEED PURGE AUDIT ---")
    print(f"Purged AUC: mean={p_mean:.4f}, std={p_std:.4f}, min={p_min:.4f}, max={p_max:.4f}")
    print(f"Plain AUC:  mean={pl_mean:.4f}, std={pl_std:.4f}, min={pl_min:.4f}, max={pl_max:.4f}")
    print(f"Difference (leaky - purged): {diff:.4f}, t-stat: {stat:.4f}, p-value: {pval:.4f}")

    # Assertion: mean purged AUC across seeds is within [0.47, 0.53]
    assert 0.47 <= p_mean <= 0.53, f"Purged AUC {p_mean:.4f} outside [0.47, 0.53]"

    # Assertion per prompt spec: mean(leaky) - mean(purged) > 0.03 AND paired t-test p < 0.01
    assert (diff > 0.03) and (pval < 0.01), (
        f"Empirical finding under Rule F8: mean(leaky) - mean(purged) = {diff:.4f} <= 0.03, "
        f"p-value = {pval:.4f} >= 0.01 on pure random-walk noise with contiguous KFold(shuffle=False)"
    )
