"""RandomForest classifier for failure-mode prediction.

Wraps ``sklearn.ensemble.RandomForestClassifier`` with a small surface:

- ``fit(X, y)`` — train on labelled feature vectors
- ``predict(X)`` — predict labels for new feature vectors
- ``cross_val_predict(X, y, n_splits)`` — out-of-fold predictions for
  honest per-class metrics
- ``save`` / ``load`` — persist the trained model to disk

Pure ML; featurisation lives in the pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


class FailureClassifier:
    """RandomForest wrapper for the Phase 3 multi-class classifier."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        random_state: int = 42,
        class_weight: str | dict = "balanced",
    ):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight,
        )
        self._random_state = random_state
        self._fitted = False

    # ----- fit / predict -----

    def fit(self, X: np.ndarray, y: np.ndarray) -> FailureClassifier:
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FailureClassifier.predict called before fit()")
        return np.asarray(self._model.predict(X))

    # ----- evaluation -----

    def cross_val_predict(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
    ) -> np.ndarray:
        """Stratified k-fold out-of-fold predictions for honest metrics."""
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self._random_state,
        )
        y_arr = np.asarray(y)
        y_pred = np.empty_like(y_arr)
        for train_idx, test_idx in skf.split(X, y_arr):
            clf = RandomForestClassifier(
                n_estimators=self._model.n_estimators,
                random_state=self._random_state,
                n_jobs=-1,
                class_weight=self._model.class_weight,
            )
            clf.fit(X[train_idx], y_arr[train_idx])
            y_pred[test_idx] = clf.predict(X[test_idx])
        return y_pred

    # ----- persistence -----

    def save(self, path: Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted FailureClassifier")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, path: Path) -> FailureClassifier:
        with path.open("rb") as f:
            model = pickle.load(f)
        obj = cls.__new__(cls)
        obj._model = model
        obj._random_state = getattr(model, "random_state", 42)
        obj._fitted = True
        return obj
