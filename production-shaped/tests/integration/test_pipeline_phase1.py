"""End-to-end Phase 1 pipeline test against a tiny synthetic bucket."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from log_rca.pipeline.phase1 import run


N_SUCCESS = 8
N_FAILED = 6


def _make_bucket(tmp_path: Path) -> Path:
    """Hand-craft a bucket with N_SUCCESS + N_FAILED runs (failures = OOM).

    We need >2v2 to clear Fisher-exact significance at p<0.05.
    OOM lines are unique to failures so Phase 1 must surface them.
    """
    root = tmp_path / "bucket"
    truth = []
    for i in range(N_SUCCESS):
        run_id = f"scheduled__ok-{i}"
        log = (
            root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
            / "task_id=extract" / "attempt=1.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"[2026-01-01T00:00:{i:02d}.000+0000] {{a.py:1}} INFO - Fetched {i*1000} rows\n"
            f"[2026-01-01T00:01:00.000+0000] {{a.py:2}} INFO - Marking task as SUCCESS\n",
            encoding="utf-8",
        )
        truth.append({
            "dag_id": "demo", "run_id": run_id,
            "outcome": "SUCCESS", "failure_mode": "", "task_count": 1,
        })

    for i in range(N_FAILED):
        run_id = f"scheduled__fail-{i}"
        log = (
            root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
            / "task_id=extract" / "attempt=1.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"[2026-01-02T00:00:{i:02d}.000+0000] {{a.py:1}} INFO - Fetched {i*1000} rows\n"
            f"[2026-01-02T00:00:30.000+0000] {{a.py:9}} ERROR - Task failed with exception\n"
            "MemoryError\n"
            f"[2026-01-02T00:01:00.000+0000] {{a.py:2}} INFO - Marking task as FAILED\n",
            encoding="utf-8",
        )
        truth.append({
            "dag_id": "demo", "run_id": run_id,
            "outcome": "FAILED", "failure_mode": "OOM", "task_count": 1,
        })

    (root / "_truth.jsonl").write_text(
        "\n".join(json.dumps(t) for t in truth) + "\n",
        encoding="utf-8",
    )
    return root


def test_phase1_runs_end_to_end_and_surfaces_oom(tmp_path: Path):
    bucket = _make_bucket(tmp_path)
    report = tmp_path / "report.md"
    stats = run(bucket_root=bucket, report_path=report)

    assert report.exists()
    body = report.read_text(encoding="utf-8")
    # MemoryError appears only in failed runs => must surface as a discriminator
    assert "MemoryError" in body
    # generic markers should be deprioritised but listed in the generic section
    assert "Task failed with exception" in body

    assert stats["runs"] == N_SUCCESS + N_FAILED
    assert stats["failed_runs"] == N_FAILED
    assert stats["dags_with_both_outcomes"] == 1
    assert stats["lines_processed"] > 0
    assert stats["templates_discovered"] >= 2


def test_phase1_raises_when_truth_missing(tmp_path: Path):
    bucket = tmp_path / "empty"
    bucket.mkdir()
    with pytest.raises(FileNotFoundError, match="No ground truth"):
        run(bucket_root=bucket, report_path=tmp_path / "r.md")


def test_phase1_saves_drain3_state_when_requested(tmp_path: Path):
    bucket = _make_bucket(tmp_path)
    state = tmp_path / "drain.pkl"
    run(bucket_root=bucket, report_path=tmp_path / "r.md", drain3_state_path=state)
    assert state.exists()
    assert state.stat().st_size > 0
