"""Drain3-backed template miner.

Wraps ``drain3.TemplateMiner`` with a small fit/transform/save/load surface
so the rest of the pipeline does not depend on Drain3's API directly.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from drain3 import TemplateMiner as _DrainMiner
from drain3.template_miner_config import TemplateMinerConfig


@dataclass(frozen=True)
class MinedLine:
    """One log line after Drain3 has assigned it a cluster."""

    cluster_id: int
    template: str
    message: str


class TemplateMiner:
    """Stateful Drain3 wrapper.

    Usage:
        miner = TemplateMiner()
        for msg in messages:
            mined = miner.add(msg)
            ...
        miner.save(path)         # persists state
        miner2 = TemplateMiner.load(path)
    """

    def __init__(self, config: TemplateMinerConfig | None = None):
        if config is None:
            config = TemplateMinerConfig()
            config.profiling_enabled = False
        self._inner = _DrainMiner(config=config)

    # ----- fit/transform -----

    def add(self, message: str) -> MinedLine:
        """Feed one message; return its mined-line record."""
        result = self._inner.add_log_message(message)
        return MinedLine(
            cluster_id=int(result["cluster_id"]),
            template=str(result["template_mined"]),
            message=message,
        )

    def fit(self, messages: Iterable[str]) -> list[MinedLine]:
        """Feed many messages, returning the per-line results."""
        out: list[MinedLine] = []
        for m in messages:
            if not m.strip():
                continue
            out.append(self.add(m))
        return out

    # ----- introspection -----

    def templates(self) -> dict[int, str]:
        """``{cluster_id: template_string}`` for every cluster discovered."""
        return {
            int(c.cluster_id): c.get_template()
            for c in self._inner.drain.clusters
        }

    def cluster_count(self) -> int:
        return len(self._inner.drain.clusters)

    # ----- persistence -----

    def save(self, path: Path) -> None:
        """Persist the Drain3 state to ``path`` (pickle)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._inner, f)

    @classmethod
    def load(cls, path: Path) -> TemplateMiner:
        with path.open("rb") as f:
            inner = pickle.load(f)
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj
