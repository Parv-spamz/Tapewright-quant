"""
Training pipeline package for leakage-safe quantitative and ML modeling.
"""

from pipelines.training.cv import (
    PurgedKFold,
    WalkForward,
    assert_no_leakage,
)
from pipelines.training.labeling import (
    average_uniqueness,
    cusum_filter,
    daily_vol,
    meta_labels,
    sample_weights,
    triple_barrier,
)
from pipelines.training.train import (
    bet_size,
    build_dataset,
    cross_sectional_ranker_dataset,
    meta_label_pipeline,
    permutation_importance_oos,
    train_with_purged_cv,
    walk_forward_signals,
)

__all__ = [
    # CV
    "PurgedKFold",
    "WalkForward",
    "assert_no_leakage",
    # Labeling
    "daily_vol",
    "cusum_filter",
    "triple_barrier",
    "meta_labels",
    "average_uniqueness",
    "sample_weights",
    # Training Pipeline
    "bet_size",
    "build_dataset",
    "train_with_purged_cv",
    "meta_label_pipeline",
    "walk_forward_signals",
    "cross_sectional_ranker_dataset",
    "permutation_importance_oos",
]
