"""Tests for the FailureClassifier wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from log_rca.ml import FailureClassifier


@pytest.fixture
def two_class_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X_a = rng.normal(loc=0, scale=0.5, size=(40, 4))
    X_b = rng.normal(loc=5, scale=0.5, size=(40, 4))
    X = np.vstack([X_a, X_b])
    y = np.array(["A"] * 40 + ["B"] * 40)
    return X, y


def test_fit_predict_basic(two_class_data):
    X, y = two_class_data
    clf = FailureClassifier().fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape
    assert (pred == y).mean() == 1.0   # trivially separable


def test_predict_requires_fit():
    clf = FailureClassifier()
    with pytest.raises(RuntimeError, match="before fit"):
        clf.predict(np.zeros((1, 4)))


def test_fit_validates_input_length():
    clf = FailureClassifier()
    with pytest.raises(ValueError, match="length mismatch"):
        clf.fit(np.zeros((5, 3)), np.array(["a", "b"]))


def test_cross_val_predict_returns_oof_labels(two_class_data):
    X, y = two_class_data
    pred = FailureClassifier().cross_val_predict(X, y, n_splits=4)
    assert pred.shape == y.shape
    # Trivially separable -> 100% accuracy
    assert (pred == y).mean() == 1.0


def test_save_load_roundtrip(two_class_data, tmp_path: Path):
    X, y = two_class_data
    clf = FailureClassifier().fit(X, y)
    path = tmp_path / "rf.pkl"
    clf.save(path)
    loaded = FailureClassifier.load(path)
    np.testing.assert_array_equal(clf.predict(X), loaded.predict(X))


def test_save_unfitted_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="unfitted"):
        FailureClassifier().save(tmp_path / "x.pkl")
