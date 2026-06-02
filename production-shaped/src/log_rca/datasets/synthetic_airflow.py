"""Loader for dataset 1 — synthetic Airflow logs produced by ``log_rca.datagen``.

The loader walks the fake-GCS bucket and yields per-line records in the
Airflow schema. It also exposes ``load_truth()`` for the ground-truth
labels used by later phases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# Same line shape as the generator emits:
#   [2026-05-22T13:35:10.731+0000] {taskinstance.py:1216} INFO - <message>
_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+\{(?P<source>[^}]+)\}\s+(?P<level>\w+)\s+-\s+(?P<message>.*)$"
)


@dataclass(frozen=True)
class SyntheticAirflowRecord:
    """One line from an Airflow task log + the run's ground-truth labels."""

    dag_id: str
    run_id: str
    task_id: str
    attempt: int
    ts: str
    source: str
    level: str
    message: str
    outcome: str          # SUCCESS / FAILED / UNKNOWN (if no truth row)
    failure_mode: str     # empty for SUCCESS


@dataclass(frozen=True)
class TruthRow:
    dag_id: str
    run_id: str
    outcome: str
    failure_mode: str
    task_count: int


class SyntheticAirflowLoader:
    """Walks ``<bucket_root>/airflow-logs`` and yields ``SyntheticAirflowRecord``s."""

    def __init__(
        self,
        bucket_root: Path,
        logs_prefix: str = "airflow-logs",
        truth_file: str = "_truth.jsonl",
    ):
        self.bucket_root = Path(bucket_root).resolve()
        self.logs_prefix = logs_prefix
        self.truth_file = truth_file

    # ----- properties -----

    @property
    def log_root(self) -> Path:
        return self.bucket_root / self.logs_prefix

    @property
    def truth_path(self) -> Path:
        return self.bucket_root / self.truth_file

    # ----- public API -----

    def load_truth(self) -> dict[str, TruthRow]:
        """``{run_id: TruthRow}`` parsed from ``_truth.jsonl``."""
        out: dict[str, TruthRow] = {}
        if not self.truth_path.exists():
            return out
        with self.truth_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                out[rec["run_id"]] = TruthRow(
                    dag_id=rec["dag_id"],
                    run_id=rec["run_id"],
                    outcome=rec["outcome"],
                    failure_mode=rec.get("failure_mode", ""),
                    task_count=rec.get("task_count", 0),
                )
        return out

    def iter_log_files(self) -> Iterator[Path]:
        if not self.log_root.exists():
            return
        yield from self.log_root.rglob("attempt=*.log")

    def load_records(self) -> Iterator[SyntheticAirflowRecord]:
        truth = self.load_truth()
        for path in self.iter_log_files():
            meta = _parse_path(path)
            tr = truth.get(meta.get("run_id", ""))
            outcome = tr.outcome if tr else "UNKNOWN"
            failure_mode = tr.failure_mode if tr else ""
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for raw in text.splitlines():
                m = _LINE_RE.match(raw)
                if m is None:
                    # Continuation line (traceback body). Keep the message,
                    # blank the structured fields.
                    yield SyntheticAirflowRecord(
                        dag_id=meta.get("dag_id", ""),
                        run_id=meta.get("run_id", ""),
                        task_id=meta.get("task_id", ""),
                        attempt=int(meta.get("attempt", "1")),
                        ts="", source="", level="",
                        message=raw,
                        outcome=outcome,
                        failure_mode=failure_mode,
                    )
                    continue
                yield SyntheticAirflowRecord(
                    dag_id=meta.get("dag_id", ""),
                    run_id=meta.get("run_id", ""),
                    task_id=meta.get("task_id", ""),
                    attempt=int(meta.get("attempt", "1")),
                    ts=m["ts"], source=m["source"], level=m["level"],
                    message=m["message"],
                    outcome=outcome,
                    failure_mode=failure_mode,
                )

    def summary(self) -> dict[str, object]:
        truth = self.load_truth()
        files = list(self.iter_log_files())
        lines = 0
        by_level: dict[str, int] = {}
        for rec in self.load_records():
            lines += 1
            if rec.level:
                by_level[rec.level] = by_level.get(rec.level, 0) + 1
        by_failure: dict[str, int] = {}
        for tr in truth.values():
            if tr.outcome == "FAILED":
                code = tr.failure_mode or "UNKNOWN"
                by_failure[code] = by_failure.get(code, 0) + 1
        return {
            "files": len(files),
            "lines": lines,
            "runs": len(truth),
            "successes": sum(1 for t in truth.values() if t.outcome == "SUCCESS"),
            "failures": sum(1 for t in truth.values() if t.outcome == "FAILED"),
            "levels": by_level,
            "failure_modes": by_failure,
        }


# ----- helpers -----

def _parse_path(p: Path) -> dict[str, str]:
    """Extract ``dag_id``, ``run_id``, ``task_id``, ``attempt`` from the path."""
    out: dict[str, str] = {}
    for part in p.parts:
        if part.startswith(("dag_id=", "run_id=", "task_id=")):
            k, _, v = part.partition("=")
            out[k] = v
        elif part.startswith("attempt=") and part.endswith(".log"):
            out["attempt"] = part[len("attempt="):-len(".log")]
    return out
