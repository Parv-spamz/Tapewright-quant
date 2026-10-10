"""Unit tests for labeling, purged cross-validation, feature registry, and sentiment.

Verifies:
1. Triple-barrier labeling on hand-built price paths (upper hit, lower hit, vertical timeout, tie-breaking)
2. Causal labeling invariance: changing data before t0 does not alter outcome
3. Symmetric CUSUM event filter on constant drift and dynamic Series threshold
4. Average label uniqueness and sample weights on hand-computed overlapping intervals
5. PurgedKFold cross-validation, assert_no_leakage verification, and comparison against leaking KFold
6. WalkForward chronological non-overlapping window splits
7. Point-in-time feature registry: lag enforcement, truncation invariance, negative lag guard
8. Lexicon sentiment polarity scoring with negator flipping
9. Text embedding dimensionality, determinism, and cosine similarity ranking
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from data.ingestion.market_feeds.equities.synthetic import make_ohlcv
from features.sentiment.embeddings.embed import (
    HashingTfidfEmbedder,
    TorchTextEmbedder,
)
from features.sentiment.lexicon_scoring.lexicon import LexiconScorer
from features.store.registry.registry import (
    FeatureRegistry,
    register_default_features,
)
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


class TestTripleBarrierLabeling:
    """Tests for Lopez de Prado triple barrier method."""

    def test_triple_barrier_on_hand_built_price_path(self) -> None:
        """Hand-constructed path: one hits PT first, one SL first, one times out."""
        dates = pd.date_range("2024-01-01", periods=15)
        # Event 1 at bar 0: hits +5% profit-take at bar 2 (105.0)
        # Event 2 at bar 4: hits -5% stop-loss at bar 6 (95.0)
        # Event 3 at bar 8: flat path, times out at bar 8 + 5 = 13 (vertical timeout)
        prices = [
            100.0, 101.0, 105.0, 103.0,  # bars 0, 1, 2, 3
            100.0,  99.0,  95.0,  97.0,  # bars 4, 5, 6, 7
            100.0, 100.1, 100.2, 100.1, 100.0, 100.1, 100.0,  # bars 8..14
        ]
        close = pd.Series(prices, index=dates)
        events = [dates[0], dates[4], dates[8]]

        res = triple_barrier(
            close=close,
            events=events,
            pt_sl=(1.0, 1.0),
            target=0.03,  # PT=+3%, SL=-3%
            horizon_bars=5,
            vertical_label="zero",
        )

        assert len(res) == 3

        # Event 1: hits PT at bar 2
        ev1 = res.iloc[0]
        assert ev1["t0"] == dates[0]
        assert ev1["t1_touch"] == dates[2]
        assert ev1["barrier"] == "pt"
        assert ev1["label"] == 1
        assert ev1["ret"] == pytest.approx(0.05)

        # Event 2: hits SL at bar 6
        ev2 = res.iloc[1]
        assert ev2["t0"] == dates[4]
        assert ev2["t1_touch"] == dates[6]
        assert ev2["barrier"] == "sl"
        assert ev2["label"] == -1
        assert ev2["ret"] == pytest.approx(-0.05)

        # Event 3: times out at bar 13
        ev3 = res.iloc[2]
        assert ev3["t0"] == dates[8]
        assert ev3["t1_touch"] == dates[13]
        assert ev3["barrier"] == "vertical"
        assert ev3["label"] == 0
        assert ev3["ret"] == pytest.approx(0.001)

    def test_triple_barrier_simultaneous_breach_order_resolution(self) -> None:
        """When both thresholds are reached in the same bar, documented rule: PT wins."""
        dates = pd.date_range("2024-01-01", periods=5)
        # Target = 0.0, meaning return 0 breaches both up (>=0) and dn (<=0) simultaneously
        close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=dates)
        # Using pt_sl=(0.0, 0.0) with target=0.01 breaches both immediately at bar 1
        res = triple_barrier(
            close=close,
            events=[dates[0]],
            pt_sl=(0.0, 0.0),
            target=0.01,
            horizon_bars=3,
        )
        assert res.iloc[0]["barrier"] == "pt"
        assert res.iloc[0]["label"] == 1

    def test_labels_use_only_data_after_t0(self) -> None:
        """Changing data strictly before t0 does NOT alter the label or touch."""
        dates = pd.date_range("2024-01-01", periods=10)
        prices_orig = [10.0, 20.0, 30.0, 100.0, 102.0, 105.0, 103.0, 100.0, 99.0, 98.0]
        prices_modified = [999.0, 1.0, 500.0, 100.0, 102.0, 105.0, 103.0, 100.0, 99.0, 98.0]

        c_orig = pd.Series(prices_orig, index=dates)
        c_mod = pd.Series(prices_modified, index=dates)
        events = [dates[3]]  # event at index 3 (value 100.0)

        res_orig = triple_barrier(c_orig, events=events, pt_sl=(1.0, 1.0), target=0.03, horizon_bars=4)
        res_mod = triple_barrier(c_mod, events=events, pt_sl=(1.0, 1.0), target=0.03, horizon_bars=4)

        pd.testing.assert_frame_equal(res_orig, res_mod)

    def test_meta_labels_binary_correctness(self) -> None:
        tb_df = pd.DataFrame({
            "t0": [1, 2, 3],
            "ret": [0.05, -0.02, 0.01],
        })
        # If model side is [1, 1, -1]
        ml = meta_labels(tb_df, side=[1, 1, -1])
        # Sample 1: ret=0.05 * side=1 > 0 -> 1
        # Sample 2: ret=-0.02 * side=1 < 0 -> 0
        # Sample 3: ret=0.01 * side=-1 < 0 -> 0
        assert list(ml.values) == [1, 0, 0]


class TestCusumFilter:
    """Tests for symmetric CUSUM filter event detection."""

    def test_cusum_constant_drift_hand_indices(self) -> None:
        """On constant ~2% drift with threshold=0.03, triggers at bar 2 and bar 4."""
        dates = pd.date_range("2024-01-01", periods=5)
        prices = [100.0, 102.0, 104.04, 106.1208, 108.243216]
        close = pd.Series(prices, index=dates)

        events = cusum_filter(close, threshold=0.03)
        assert list(events) == [dates[2], dates[4]]

    def test_cusum_series_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5)
        close = pd.Series([100.0, 102.0, 104.04, 106.1208, 108.243216], index=dates)
        vol_th = pd.Series([0.05, 0.05, 0.01, 0.05, 0.05], index=dates)

        events = cusum_filter(close, threshold=vol_th)
        # At bar 2 (2024-01-03), threshold is 0.01 < return 0.02, so it triggers
        assert dates[2] in events


class TestUniquenessAndSampleWeights:
    """Tests for Lopez de Prado label uniqueness and sample weights."""

    def test_average_uniqueness_hand_example(self) -> None:
        """3 labels with intervals [0,2], [1,3], [4,5] -> u = [2/3, 2/3, 1.0]."""
        idx = pd.date_range("2024-01-01", periods=6)
        df_t = pd.DataFrame({
            "t0": [idx[0], idx[1], idx[4]],
            "t1_touch": [idx[2], idx[3], idx[5]],
        })

        u = average_uniqueness(df_t, idx)
        expected_u = [2.0 / 3.0, 2.0 / 3.0, 1.0]
        np.testing.assert_allclose(u.values, expected_u)

        w = sample_weights(df_t, idx)
        assert abs(np.sum(w) - 3.0) < 1e-9


class TestPurgedCrossValidation:
    """Tests for PurgedKFold and leakage assertions."""

    def test_purged_kfold_no_leakage_and_sklearn_failure(self) -> None:
        N = 300
        horizon = 5
        t0 = np.arange(N)
        t1 = t0 + horizon

        # 1. Plain sklearn KFold FAILS assert_no_leakage
        plain_kf = KFold(n_splits=5)
        train_plain, test_plain = next(plain_kf.split(np.arange(N)))
        with pytest.raises(AssertionError, match="Data leakage detected"):
            assert_no_leakage(train_plain, test_plain, t0, t1)

        # 2. PurgedKFold passes assert_no_leakage for every fold
        embargo_pct = 0.01
        pkf = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
        fold_count = 0

        for train_idx, test_idx in pkf.split(np.arange(N), t0=t0, t1=t1):
            fold_count += 1
            assert_no_leakage(train_idx, test_idx, t0, t1)

            # Verify embargo size: test_end + int(ceil(N * embargo_pct))
            expected_embargo = int(np.ceil(N * embargo_pct))
            test_end = np.max(test_idx)
            # No training sample can start in (test_end, test_end + expected_embargo]
            embargo_breaches = (train_idx > test_end) & (train_idx <= test_end + expected_embargo)
            assert not np.any(embargo_breaches)

        assert fold_count == 5

    def test_walk_forward_chronological_and_non_overlapping(self) -> None:
        N = 200
        wf = WalkForward(train_size=100, test_size=20, step=20, expanding=True)
        splits = list(wf.split(np.arange(N)))
        assert len(splits) == 5

        for train_idx, test_idx in splits:
            assert np.max(train_idx) < np.min(test_idx)


class TestFeatureRegistryPointInTime:
    """Tests for FeatureRegistry point-in-time calculation and lag invariants."""

    def test_registry_point_in_time_and_lags(self) -> None:
        df = make_ohlcv("AAPL", start="2023-01-01", n_days=60, seed=42)
        reg = FeatureRegistry()
        register_default_features(reg)

        # Negative lag raises ValueError
        with pytest.raises(ValueError, match="lag cannot be negative"):
            reg.register("invalid", lambda d: d["close"], lag=-1)

        # lag=1 output at t equals raw feature at t-1
        reg.register("raw_close", lambda d: d["close"], lag=0)
        reg.register("lag1_close", lambda d: d["close"], lag=1)

        raw_c = reg.compute("raw_close", df)
        lag1_c = reg.compute("lag1_close", df)
        assert lag1_c.iloc[10] == raw_c.iloc[9]

        # point_in_time(asof=t) equals full-data feature at t (truncation invariance)
        asof_t = df.index[30]
        pit_sma = reg.point_in_time("sma_20", df, asof=asof_t)
        full_sma = reg.compute("sma_20", df)
        assert pit_sma.loc[asof_t] == full_sma.loc[asof_t]

        # Catalog
        cat = reg.catalog()
        assert "sma_20" in cat["name"].values
        assert "ffd_04" in cat["name"].values


class TestSentimentLexiconAndEmbeddings:
    """Tests for LexiconScorer and HashingTfidfEmbedder."""

    def test_lexicon_scorer_polarity_and_negation(self) -> None:
        ls = LexiconScorer()
        assert ls.score("profit growth strong") > 0.0
        assert ls.score("loss decline weak") < 0.0
        assert ls.score("not strong") < 0.0

    def test_embedder_properties_and_similarity(self) -> None:
        embedder = HashingTfidfEmbedder(n_features=64)

        texts = [
            "company reports higher quarterly profit and solid revenue growth",
            "company reports strong quarterly profit and robust revenue expansion",
            "heavy rainfall causes severe flooding in remote northern valley",
        ]

        vecs = embedder.fit_transform(texts)
        assert vecs.shape == (3, 64)

        # Deterministic
        vecs_repeat = embedder.transform(texts)
        np.testing.assert_allclose(vecs, vecs_repeat)

        # Similar headlines have higher cosine similarity than unrelated headline
        sim_similar = float(np.dot(vecs[0], vecs[1]))
        sim_unrelated = float(np.dot(vecs[0], vecs[2]))

        print(f"\n[Embeddings Cosine Similarity]: Similar={sim_similar:.4f}, Unrelated={sim_unrelated:.4f}")
        assert sim_similar > sim_unrelated
        assert sim_similar > 0.40

    def test_torch_text_embedder_import_fallback(self) -> None:
        """TorchTextEmbedder raises clear ImportError if transformers missing."""
        import importlib.util
        if importlib.util.find_spec("transformers") is not None:
            assert True
        else:
            with pytest.raises(ImportError, match="requires 'transformers' library"):
                TorchTextEmbedder()
