"""Phase 1+ ML components.

- ``clustering``: Drain3-backed template miner with fit/transform/save/load.
- ``discrimination``: Fisher's-exact-based discovery of failure-correlated
  templates (used in the Phase 1 RCA report).
"""

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
]
