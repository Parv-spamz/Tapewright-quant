"""Honest multi-seed purge tests and index-level interval leakage verification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from pipelines.training.cv import PurgedKFold, assert_no_leakage
from pipelines.training.labeling import daily_vol, triple_barrier


def test_purged_kfold_index_invariants_and_leakage_elimination() -> None:
    """Verify index-level interval leakage in plain KFold and its elimination in PurgedKFold.

    Verifies the core architectural invariants:
    1. Ordinary KFold produces training events whose label intervals [t0, t1] overlap
       the test fold evaluation window [test_min_t0, test_max_t1].
    2. PurgedKFold removes all overlapping training events for every evaluated fold.
    3. Embargo separation is tested independently from interval purging.
    4. All indices, intervals, and overlap counts are calculated dynamically.
    """
    np.random.seed(42)
    n = 600
    ret = np.random.normal(0, 0.01, n)
    price = pd.Series(
        100.0 * np.exp(np.cumsum(ret)),
        index=pd.date_range("2024-01-01", periods=n),
    )
    v = daily_vol(price, span=50)

    # Build triple-barrier events with horizon 10
    tb = triple_barrier(
        price,
        events=price.index[10:],
        pt_sl=(1.0, 1.0),
        target=v,
        horizon_bars=10,
        vertical_label="sign",
    )
    tb = tb[tb["label"] != 0].copy().dropna()

    N = len(tb)
    assert N > 100, f"Insufficient labeled events: {N}"

    t0_arr = tb["t0"].values
    t1_arr = tb["t1_touch"].values
    sample_indices = np.arange(N)

    # 1. Verify that ordinary KFold (shuffle=False) leaks across fold boundaries
    plain_kf = KFold(n_splits=5, shuffle=False)
    plain_overlap_counts: list[int] = []

    for tr, te in plain_kf.split(sample_indices):
        test_min_t0 = np.min(t0_arr[te])
        test_max_t1 = np.max(t1_arr[te])
        tr_t0 = t0_arr[tr]
        tr_t1 = t1_arr[tr]
        # Overlap occurs iff train_t0 <= test_max_t1 AND train_t1 >= test_min_t0
        overlaps = (tr_t0 <= test_max_t1) & (tr_t1 >= test_min_t0)
        plain_overlap_counts.append(int(np.sum(overlaps)))

    total_plain_overlaps = sum(plain_overlap_counts)
    # INVARIANT 1: Plain KFold must produce at least one overlapping training event
    assert total_plain_overlaps > 0, (
        f"Expected ordinary KFold to produce overlapping training events, got 0"
    )

    # 2. Verify that PurgedKFold (without embargo) removes ALL overlapping training events
    pkf_no_embargo = PurgedKFold(n_splits=5, embargo_pct=0.0)
    purged_overlap_counts: list[int] = []
    purged_removed_counts: list[int] = []

    for tr, te in pkf_no_embargo.split(sample_indices, t0=t0_arr, t1=t1_arr):
        test_min_t0 = np.min(t0_arr[te])
        test_max_t1 = np.max(t1_arr[te])
        tr_t0 = t0_arr[tr]
        tr_t1 = t1_arr[tr]
        overlaps = (tr_t0 <= test_max_t1) & (tr_t1 >= test_min_t0)
        purged_overlap_counts.append(int(np.sum(overlaps)))

        # Dynamically calculated removed count: total non-test minus actual train count
        expected_non_test = N - len(te)
        removed_count = expected_non_test - len(tr)
        purged_removed_counts.append(removed_count)

        # INVARIANT 2: assert_no_leakage must pass for every fold
        assert_no_leakage(tr, te, t0_arr, t1_arr)

    # Every fold must have zero overlapping events after purging
    assert all(c == 0 for c in purged_overlap_counts), (
        f"PurgedKFold retained overlapping training events: {purged_overlap_counts}"
    )
    # Total samples removed by interval purging must be strictly positive
    assert sum(purged_removed_counts) > 0, "PurgedKFold removed 0 samples"

    # 3. Test embargo separation independently from interval purging
    embargo_pct = 0.02
    pkf_embargo = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
    expected_embargo_samples = int(np.ceil(N * embargo_pct))
    assert expected_embargo_samples > 0, "Embargo sample count must be positive"

    embargo_separation_verified: list[bool] = []
    for tr, te in pkf_embargo.split(sample_indices, t0=t0_arr, t1=t1_arr):
        test_end_idx = np.max(te)
        post_test_train = tr[tr > test_end_idx]
        if len(post_test_train) > 0:
            actual_gap = np.min(post_test_train) - test_end_idx
            # INVARIANT 3: post-test training samples must respect the embargo separation
            embargo_separation_verified.append(actual_gap >= expected_embargo_samples)

    assert len(embargo_separation_verified) > 0, "No folds had post-test training data"
    assert all(embargo_separation_verified), (
        f"Embargo gap violated: expected >= {expected_embargo_samples} sample separation"
    )


def test_honest_multi_seed_purge_vs_leaky() -> None:
    """Over 30 seeds of fresh random-walk prices, audit PurgedKFold and plain KFold AUC.

    Preserves descriptive empirical evidence across 30 seeds:
    - Verifies that PurgedKFold AUC is within the theoretical pure noise bounds [0.47, 0.53].
    - Measures and reports the empirical difference and paired t-test statistics without
      relying on an unsupported difference threshold on i.i.d. random walks.
    """
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

    # Core statistical expectation: Purged CV on pure noise must remain in [0.47, 0.53]
    assert 0.47 <= p_mean <= 0.53, f"Purged AUC {p_mean:.4f} outside noise bounds [0.47, 0.53]"
