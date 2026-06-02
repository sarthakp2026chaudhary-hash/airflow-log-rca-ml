"""Phase 1+ ML components.

- ``clustering``:     Drain3-backed template miner (fit/transform/save/load).
- ``discrimination``: Fisher's-exact-based discovery of failure-correlated
  templates (Phase 1 RCA report).
- ``anomaly``:        IsolationForest per-run anomaly detection (Phase 2).
- ``classification``: RandomForest failure-mode classifier (Phase 3).
"""

from log_rca.ml.anomaly import AnomalyDetector, AnomalyScore
from log_rca.ml.classification import FailureClassifier
from log_rca.ml.clustering import MinedLine, TemplateMiner
from log_rca.ml.discrimination import (
    GENERIC_FAILURE_PATTERNS,
    Discriminator,
    is_generic_template,
)

__all__ = [
    "TemplateMiner",
    "MinedLine",
    "Discriminator",
    "is_generic_template",
    "GENERIC_FAILURE_PATTERNS",
    "AnomalyDetector",
    "AnomalyScore",
    "FailureClassifier",
]
