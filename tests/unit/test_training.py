"""
Unit tests for tabular ensemble models, purged cross-validation honesty, meta-labeling,
walk-forward signal gating, and model registry persistence.
"""

from __future__ import annotations

import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from database.connection.manager import ConnectionManager
from database.migrations.runner import apply_migrations
from database.repositories.market_data_repo.model_registry_repo import ModelRegistryRepo
from models.ensemble.catboost import CatBoostModel
from models.ensemble.lightgbm import LightGBMModel
from models.ensemble.xgboost import XGBoostModel
from models.registry import register_model
from pipelines.training.cv import PurgedKFold
from pipelines.training.labeling import daily_vol, triple_barrier
from pipelines.training.train import (
    bet_size,
    build_dataset,
    cross_sectional_ranker_dataset,
    meta_label_pipeline,
    permutation_importance_oos,
    train_with_purged_cv,
    walk_forward_signals,
)


class TestTabularEnsembles:
    """Tests for XGBoost, LightGBM, and CatBoost models and out-of-sample importance."""

    def test_top_feature_importance_predict_shapes_and_save_load_round_trip(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 4), columns=["f0", "f1", "f2", "f3"])
        # Target y depends exclusively on f0
        prob = 1.0 / (1.0 + np.exp(-3.5 * X["f0"]))
        y = (np.random.rand(n) < prob).astype(int)

        X_tr, y_tr = X.iloc[:140], y.iloc[:140]
        X_te, y_te = X.iloc[140:], y.iloc[140:]

        models_to_test = [
            ("XGBoost", XGBoostModel(n_estimators=30, max_depth=3, random_state=42)),
            ("LightGBM", LightGBMModel(n_estimators=30, max_depth=3, random_state=42)),
            ("CatBoost", CatBoostModel(iterations=30, depth=3, random_seed=42)),
        ]

        for name, model in models_to_test:
            model.fit(X_tr, y_tr)

            # Top OOS permutation feature importance must be f0
            pi = permutation_importance_oos(model, X_te, y_te, n_repeats=5, seed=42)
            assert pi.index[0] == "f0", f"{name} top feature was {pi.index[0]} instead of f0"

            # Predict shapes
            preds = model.predict(X_te)
            probas = model.predict_proba(X_te)
            assert preds.shape == (60,), f"{name} predict shape mismatch: {preds.shape}"
            assert probas.shape == (60, 2), f"{name} predict_proba shape mismatch: {probas.shape}"

            # Save / Load round trip produces identical predictions
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                path = f.name
            model.save(path)
            loaded_model = type(model).load(path)

            np.testing.assert_allclose(
                model.predict_proba(X_te),
                loaded_model.predict_proba(X_te),
                err_msg=f"{name} save/load round trip yielded different predictions!",
            )

    def test_sample_weights_change_model_fit(self):
        np.random.seed(42)
        n = 120
        X = pd.DataFrame(np.random.randn(n, 3), columns=["x1", "x2", "x3"])
        y = np.random.randint(0, 2, n)

        # Strongly skewed weights
        w = np.ones(n, dtype=float)
        w[:25] = 100.0

        m_unweighted = XGBoostModel(n_estimators=20, max_depth=3, random_state=42).fit(X, y)
        m_weighted = XGBoostModel(n_estimators=20, max_depth=3, random_state=42).fit(
            X, y, sample_weight=w
        )

        p_unw = m_unweighted.predict_proba(X)[:, 1]
        p_w = m_weighted.predict_proba(X)[:, 1]

        max_diff = float(np.max(np.abs(p_unw - p_w)))
        assert max_diff > 0.05, f"Sample weights did not alter fit (max diff = {max_diff:.4f})"


class TestPurgedCVHonesty:
    """Verifies that Purged K-Fold prevents label-overlap leakage on pure noise."""

    def test_purged_cv_honesty_on_pure_noise_vs_leaky_kfold(self):
        np.random.seed(42)
        n = 600
        # Pure noise random walk prices
        ret = np.random.randn(n) * 0.01
        price = pd.Series(100.0 * np.exp(np.cumsum(ret)))
        v = daily_vol(price, span=50)

        events = price.index[20:-20]
        tb = triple_barrier(
            price,
            events=events,
            pt_sl=(1.0, 1.0),
            target=v,
            horizon_bars=15,
            vertical_label="sign",
        )
        tb = tb[tb["label"] != 0].copy().reset_index(drop=True)

        # Overlapping rolling average features (which leak forward information across overlapping label spans)
        feat = pd.DataFrame(
            {
                "roll_10": price.rolling(10).mean().shift(1),
                "roll_15": price.rolling(15).mean().shift(1),
            }
        )
        X = feat.loc[tb["t0"]].reset_index(drop=True)
        y = (tb["label"] == 1).astype(int).values
        t0 = tb["t0"].values
        t1 = tb["t1_touch"].values

        # 1. Purged K-Fold
        purged_cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        purged_res = train_with_purged_cv(
            model_factory=lambda: XGBoostModel(
                n_estimators=40, max_depth=5, val_ratio=0.0, random_state=42
            ),
            X=X,
            y=y,
            t0=t0,
            t1=t1,
            cv=purged_cv,
        )
        mean_purged_auc = purged_res["mean_auc"]

        # 2. Deliberately leaky plain KFold with shuffling
        plain_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        plain_aucs: list[float] = []
        for tr, te in plain_cv.split(X):
            m = XGBoostModel(
                n_estimators=40, max_depth=5, val_ratio=0.0, random_state=42
            )
            m.fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            plain_aucs.append(roc_auc_score(y[te], p))
        mean_plain_auc = float(np.mean(plain_aucs))

        # REPORT HONEST FINDINGS:
        # Purged CV on pure noise must be within [0.45, 0.55]
        assert 0.45 <= mean_purged_auc <= 0.55, (
            f"Purged CV AUC ({mean_purged_auc:.4f}) outside pure noise expectation [0.45, 0.55]"
        )

        # Plain KFold exhibits leakage inflation > purged AUC
        assert mean_plain_auc > mean_purged_auc + 0.05, (
            f"Plain KFold AUC ({mean_plain_auc:.4f}) not visibly higher than Purged CV ({mean_purged_auc:.4f})"
        )

    def test_planted_signal_recovers_purged_auc_above_threshold(self):
        np.random.seed(42)
        n = 600
        # Construct price series where feature at t0 causally drives forward returns
        signal = np.random.randn(n)
        ret = np.zeros(n)
        for t in range(1, n):
            ret[t] = 0.012 * signal[t - 1] + np.random.randn() * 0.005

        price = pd.Series(100.0 * np.exp(np.cumsum(ret)))
        v = daily_vol(price, span=30)
        events = price.index[20:-20]
        tb = triple_barrier(
            price,
            events=events,
            pt_sl=(1.0, 1.0),
            target=v,
            horizon_bars=5,
            vertical_label="sign",
        )
        tb = tb[tb["label"] != 0].copy().reset_index(drop=True)

        X = pd.DataFrame({"signal": signal[tb["t0"]]})
        y = (tb["label"] == 1).astype(int).values
        t0 = tb["t0"].values
        t1 = tb["t1_touch"].values

        res = train_with_purged_cv(
            model_factory=lambda: XGBoostModel(n_estimators=30, max_depth=3, random_state=42),
            X=X,
            y=y,
            t0=t0,
            t1=t1,
        )
        mean_auc = res["mean_auc"]
        assert mean_auc > 0.55, f"Expected planted signal Purged AUC > 0.55, got {mean_auc:.4f}"


class TestMetaLabelingAndWalkForward:
    """Tests for secondary meta-labeling position sizing and walk-forward verification."""

    def test_meta_label_pipeline_oracle_flag_and_bet_sizing(self):
        np.random.seed(42)
        n = 500
        primary_side = np.random.choice([-1, 1], size=n)
        # Primary model right 55% of the time
        is_right = np.random.rand(n) < 0.55
        realized_ret = np.where(is_right, 0.02 * primary_side, -0.02 * primary_side)

        t0 = np.arange(n)
        t1 = t0 + 2
        tb_df = pd.DataFrame({"t0": t0, "t1_touch": t1, "ret": realized_ret})

        # Feature that flags when the primary model is right
        feat_flag = np.where(is_right, 1.0, 0.0) + np.random.randn(n) * 0.02
        X = pd.DataFrame({"oracle_feature": feat_flag})

        meta_res = meta_label_pipeline(
            primary_side=primary_side,
            X=X,
            triple_barrier_df=tb_df,
            model_factory=lambda: XGBoostModel(n_estimators=30, max_depth=2, random_state=42),
        )

        oof_auc = meta_res["oof_auc"]
        assert oof_auc > 0.90, f"Expected meta model OOF AUC > 0.9, got {oof_auc:.4f}"

        # Bet sizes must be in [0, 1]
        sizes = meta_res["bet_sizes"]
        valid_sizes = sizes.dropna()
        assert (valid_sizes >= 0.0).all() and (valid_sizes <= 1.0).all()
        assert bet_size(0.4) == 0.0
        assert bet_size(1.0) == 1.0
        assert abs(bet_size(0.75) - 0.5) < 1e-12

    def test_walk_forward_signals_strict_no_leakage_assertion(self):
        np.random.seed(42)
        n = 160
        X = pd.DataFrame(np.random.randn(n, 3), columns=["x1", "x2", "x3"])
        y = np.random.randint(0, 2, n)
        t0 = np.arange(n)
        t1 = t0 + 3

        wf_res = walk_forward_signals(
            model_factory=lambda: XGBoostModel(n_estimators=10, max_depth=2, random_state=42),
            X=X,
            y=y,
            t0=t0,
            t1=t1,
            n_splits=4,
            min_train_samples=30,
        )

        assert len(wf_res["metadata"]) == 4
        for meta in wf_res["metadata"]:
            # Every signal's training set max(t1) < prediction block's first timestamp
            assert meta["max_t1_train"] < meta["min_t0_test"], (
                f"Walk-forward leakage in block {meta['block']}: "
                f"max(t1_train)={meta['max_t1_train']} >= min(t0_test)={meta['min_t0_test']}"
            )
            assert meta["is_purged"] is True

    def test_build_dataset_and_cross_sectional_ranker(self):
        # 1. build_dataset integration
        np.random.seed(42)
        price = pd.Series(100.0 * np.exp(np.cumsum(np.random.randn(300) * 0.01)))
        X, y, t0, t1, w = build_dataset(price, horizon=5)
        assert len(X) == len(y) == len(t0) == len(t1) == len(w) > 0
        assert "ffd_04" in X.columns
        assert abs(w.sum() - len(w)) < 1e-6

        # 2. cross_sectional_ranker_dataset
        dates = pd.date_range("2026-01-01", periods=10)
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
        rows = []
        for d in dates:
            for s in symbols:
                rows.append({"date": d, "symbol": s, "close": 100.0 + np.random.randn(), "f1": np.random.randn()})
        panel = pd.DataFrame(rows)
        X_rank, y_rank, d_out, groups = cross_sectional_ranker_dataset(panel, horizon=1)
        assert (y_rank >= 0.0).all() and (y_rank <= 1.0).all()
        assert len(groups) == 9
        assert groups[0] == 4


class TestModelRegistryPersistence:
    """Tests for model registration and persistence into SQLite database."""

    def test_registry_round_trip_on_sqlite(self):
        ConnectionManager.reset_instance()
        cm = ConnectionManager.get_instance(backend="sqlite", sqlite_path=":memory:")
        with cm.connection() as conn:
            apply_migrations(conn)

        repo = ModelRegistryRepo(cm)

        metrics = {"purged_auc": 0.635, "accuracy": 0.58, "log_loss": 0.67}
        path = "/artifacts/models/xgboost_v1.bin"

        model_id = register_model(
            name="tapewright_xgboost_alpha",
            version="1.0.0",
            metrics=metrics,
            path=path,
            repo=repo,
            family="ensemble",
        )

        latest = repo.get_latest("tapewright_xgboost_alpha")
        assert latest is not None
        assert latest["model_id"] == model_id
        assert latest["model_name"] == "tapewright_xgboost_alpha"
        assert latest["version"] == "1.0.0"
        assert latest["metrics"]["purged_auc"] == 0.635
        assert latest["hyperparameters"]["path"] == path

        ConnectionManager.reset_instance()
