"""Tests for the IsolationForest anomaly detector wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from log_rca.ml import AnomalyDetector, AnomalyScore


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_fit_requires_min_rows(rng: np.random.Generator):
    detector = AnomalyDetector()
    too_few = rng.normal(size=(5, 3))
    with pytest.raises(ValueError, match="at least 10"):
        detector.fit(too_few)


def test_score_requires_fit():
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError, match="before fit"):
        detector.score(np.zeros((1, 3)))


def test_anomalous_rows_get_lower_scores(rng: np.random.Generator):
    # 100 'normal' rows around origin
    normal = rng.normal(loc=0, scale=1, size=(100, 5))
    # 5 outliers far away
    outliers = rng.normal(loc=10, scale=1, size=(5, 5))
    X = np.vstack([normal, outliers])

    detector = AnomalyDetector().fit(normal)
    scores = detector.score(X)
    # outliers (rows 100-104) should be the lowest-scoring rows
    order = np.argsort(scores)
    assert set(order[:5].tolist()) == {100, 101, 102, 103, 104}


def test_rank_returns_topk_in_ascending_score_order(rng: np.random.Generator):
    normal = rng.normal(size=(50, 4))
    detector = AnomalyDetector().fit(normal)
    X = np.vstack([normal, rng.normal(loc=8, size=(3, 4))])
    top = detector.rank(X, top_k=3)
    assert len(top) == 3
    assert all(isinstance(t, AnomalyScore) for t in top)
    # scores should be in ascending order
    assert top[0].score <= top[1].score <= top[2].score
    # the outliers we added should be the top picks
    assert {top[0].row_index, top[1].row_index, top[2].row_index} == {50, 51, 52}


def test_save_and_load_roundtrip(rng: np.random.Generator, tmp_path: Path):
    X = rng.normal(size=(20, 3))
    detector = AnomalyDetector().fit(X)
    path = tmp_path / "model.pkl"
    detector.save(path)

    loaded = AnomalyDetector.load(path)
    np.testing.assert_array_equal(detector.score(X), loaded.score(X))


def test_save_unfitted_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="unfitted"):
        AnomalyDetector().save(tmp_path / "m.pkl")
