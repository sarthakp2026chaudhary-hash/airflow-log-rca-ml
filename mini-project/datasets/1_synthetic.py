"""
Dataset 1 loader — Synthetic Airflow logs (mini-project)
=========================================================

Walks ``fake_gcs_bucket/`` (produced by ``mini-project/generate_logs.py``)
and yields one record per LOG LINE, keeping the natural Airflow schema.

Run standalone:
    python mini-project/datasets/1_synthetic.py

Prints a summary table; later phases import ``load_records`` from this file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
BUCKET = REPO_ROOT / "fake_gcs_bucket"
LOG_ROOT = BUCKET / "airflow-logs"
TRUTH_PATH = BUCKET / "_truth.jsonl"

# Matches:  [2026-05-22T13:35:10.731+0000] {taskinstance.py:1216} INFO - message
LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+\{(?P<source>[^}]+)\}\s+(?P<level>\w+)\s+-\s+(?P<message>.*)$"
)


def load_truth() -> dict[str, dict]:
    """Return ``{run_id: truth_record}`` from ``fake_gcs_bucket/_truth.jsonl``."""
    out: dict[str, dict] = {}
    if not TRUTH_PATH.exists():
        return out
    with TRUTH_PATH.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            out[rec["run_id"]] = rec
    return out


def iter_log_files() -> Iterator[Path]:
    if not LOG_ROOT.exists():
        return
    yield from LOG_ROOT.rglob("attempt=*.log")


def _parse_path(p: Path) -> dict[str, str]:
    """Extract ``dag_id``, ``run_id``, ``task_id``, ``attempt`` from the Airflow path."""
    out: dict[str, str] = {}
    for part in p.parts:
        if "=" in part and part.startswith(("dag_id=", "run_id=", "task_id=")):
            k, _, v = part.partition("=")
            out[k] = v
        elif part.startswith("attempt=") and part.endswith(".log"):
            out["attempt"] = part[len("attempt="):-len(".log")]
    return out


def load_records() -> Iterator[dict]:
    """Yield one record per log LINE.

    Schema (natural to Airflow):
        dag_id, run_id, task_id, attempt (int), ts (str), source (str),
        level (str), message (str), outcome (str), failure_mode (str)
    """
    truth = load_truth()
    for path in iter_log_files():
        path_meta = _parse_path(path)
        run_truth = truth.get(path_meta.get("run_id", ""), {})
        outcome = run_truth.get("outcome", "UNKNOWN")
        failure_mode = run_truth.get("failure_mode", "")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            m = LINE_RE.match(raw)
            if m is None:
                # Continuation lines (tracebacks etc.) carry the previous timestamp;
                # for Phase 0 we just yield them with empty ts/source.
                yield {
                    "dag_id": path_meta.get("dag_id", ""),
                    "run_id": path_meta.get("run_id", ""),
                    "task_id": path_meta.get("task_id", ""),
                    "attempt": int(path_meta.get("attempt", "1")),
                    "ts": "",
                    "source": "",
                    "level": "",
                    "message": raw,
                    "outcome": outcome,
                    "failure_mode": failure_mode,
                }
                continue
            yield {
                "dag_id": path_meta.get("dag_id", ""),
                "run_id": path_meta.get("run_id", ""),
                "task_id": path_meta.get("task_id", ""),
                "attempt": int(path_meta.get("attempt", "1")),
                "ts": m["ts"],
                "source": m["source"],
                "level": m["level"],
                "message": m["message"],
                "outcome": outcome,
                "failure_mode": failure_mode,
            }


def summarise() -> None:
    if not BUCKET.exists():
        print(f"ERROR: {BUCKET} does not exist.", file=sys.stderr)
        print("Run:  python mini-project/generate_logs.py", file=sys.stderr)
        sys.exit(1)

    truth = load_truth()
    files = list(iter_log_files())
    lines = 0
    by_level: dict[str, int] = {}
    by_failure: dict[str, int] = {}
    for rec in load_records():
        lines += 1
        if rec["level"]:
            by_level[rec["level"]] = by_level.get(rec["level"], 0) + 1

    for rec in truth.values():
        if rec["outcome"] == "FAILED":
            code = rec["failure_mode"] or "UNKNOWN"
            by_failure[code] = by_failure.get(code, 0) + 1

    print(f"Dataset 1: Synthetic Airflow logs")
    print(f"  bucket:     {BUCKET}")
    print(f"  log files:  {len(files):,}")
    print(f"  log lines:  {lines:,}")
    print(f"  DAG runs:   {len(truth):,}")
    print(f"  outcomes:   SUCCESS={sum(1 for r in truth.values() if r['outcome']=='SUCCESS')}, "
          f"FAILED={sum(1 for r in truth.values() if r['outcome']=='FAILED')}")
    print(f"  log levels seen: {dict(sorted(by_level.items()))}")
    print(f"  failure-mode breakdown (from _truth.jsonl):")
    for code in sorted(by_failure):
        print(f"    {code:<18} {by_failure[code]:>4}")


if __name__ == "__main__":
    summarise()
