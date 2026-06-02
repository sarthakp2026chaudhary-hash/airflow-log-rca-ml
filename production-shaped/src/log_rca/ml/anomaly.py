"""IsolationForest-based anomaly detector for per-run feature vectors.

Pure-ML wrapper; featurisation lives in the pipeline. The detector
exposes ``fit`` (on SUCCESS-only rows) + ``score`` (on all rows). Lower
score = more anomalous.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class AnomalyScore:
    """One scored row: same shape as input row index."""

    row_index: int
    score: float


class AnomalyDetector:
    """One-class anomaly detector for per-run feature vectors."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        contamination: float | str = "auto",
        random_state: int = 42,
    ):
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            max_samples="auto",
        )
        self._fitted = False

    # ----- fit / score -----

    def fit(self, X_success: np.ndarray) -> AnomalyDetector:
        """Fit on rows known to be normal (SUCCESS runs only)."""
        if len(X_success) < 10:
            raise ValueError(
                f"Need at least 10 SUCCESS rows to fit; got {len(X_success)}"
            )
        self._model.fit(X_success)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return decision-function scores. Lower = more anomalous."""
        if not self._fitted:
            raise RuntimeError("AnomalyDetector.score called before fit()")
        return np.asarray(self._model.decision_function(X), dtype=np.float64)

    def rank(self, X: np.ndarray, top_k: int) -> list[AnomalyScore]:
        """Return the ``top_k`` most-anomalous rows in ascending score order."""
        scores = self.score(X)
        order = np.argsort(scores)[:top_k]
        return [AnomalyScore(row_index=int(i), score=float(scores[i])) for i in order]

    # ----- persistence -----

    def save(self, path: Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted AnomalyDetector")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, path: Path) -> AnomalyDetector:
        with path.open("rb") as f:
            model = pickle.load(f)
        obj = cls.__new__(cls)
        obj._model = model
        obj._fitted = True
        return obj
