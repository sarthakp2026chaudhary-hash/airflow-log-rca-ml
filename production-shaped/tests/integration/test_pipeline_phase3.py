"""End-to-end Phase 3 pipeline test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from log_rca.pipeline.phase3 import run


def _make_bucket_two_modes(tmp_path: Path) -> Path:
    """Build a bucket with two failure modes (OOM, CONN_REFUSED) plus
    successes. Templates are distinctive so the classifier should hit 100%.
    """
    root = tmp_path / "bucket"
    truth = []

    # 8 SUCCESS (won't be used by the classifier)
    for i in range(8):
        run_id = f"scheduled__ok-{i}"
        log = (
            root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
            / "task_id=t" / "attempt=1.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"[2026-01-01T00:00:{i:02d}.000+0000] {{a.py:1}} INFO - Marking task as SUCCESS\n",
            encoding="utf-8",
        )
        truth.append({
            "dag_id": "demo", "run_id": run_id,
            "outcome": "SUCCESS", "failure_mode": "", "task_count": 1,
        })

    # 8 OOM failures
    for i in range(8):
        run_id = f"scheduled__oom-{i}"
        log = (
            root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
            / "task_id=t" / "attempt=1.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"[2026-01-02T00:00:{i:02d}.000+0000] {{a.py:1}} ERROR - Task failed with exception\n"
            "MemoryError: out of memory\n"
            "Killed\n",
            encoding="utf-8",
        )
        truth.append({
            "dag_id": "demo", "run_id": run_id,
            "outcome": "FAILED", "failure_mode": "OOM", "task_count": 1,
        })

    # 8 CONN_REFUSED failures
    for i in range(8):
        run_id = f"scheduled__cr-{i}"
        log = (
            root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
            / "task_id=t" / "attempt=1.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"[2026-01-03T00:00:{i:02d}.000+0000] {{a.py:1}} ERROR - Task failed with exception\n"
            "psycopg2.OperationalError: connection to server failed\n"
            "Connection refused\n",
            encoding="utf-8",
        )
        truth.append({
            "dag_id": "demo", "run_id": run_id,
            "outcome": "FAILED", "failure_mode": "CONN_REFUSED", "task_count": 1,
        })

    (root / "_truth.jsonl").write_text(
        "\n".join(json.dumps(t) for t in truth) + "\n",
        encoding="utf-8",
    )
    return root


def test_phase3_classifies_distinct_modes_correctly(tmp_path: Path):
    bucket = _make_bucket_two_modes(tmp_path)
    report = tmp_path / "p3.md"
    stats = run(bucket_root=bucket, report_path=report, n_splits=4)
    assert report.exists()
    assert stats["n_samples"] == 16     # 8 OOM + 8 CONN_REFUSED
    assert stats["n_classes"] == 2
    assert stats["accuracy"] >= 0.9     # distinctive templates -> easy


def test_phase3_raises_when_no_failures(tmp_path: Path):
    root = tmp_path / "bucket"
    truth = [{
        "dag_id": "d", "run_id": "scheduled__s1",
        "outcome": "SUCCESS", "failure_mode": "", "task_count": 1,
    }]
    log = (root / "airflow-logs" / "dag_id=d" / "run_id=scheduled__s1"
           / "task_id=t" / "attempt=1.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("[2026-01-01T00:00:00.000+0000] {a.py:1} INFO - ok\n", encoding="utf-8")
    (root / "_truth.jsonl").write_text(
        "\n".join(json.dumps(t) for t in truth) + "\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No FAILED runs"):
        run(bucket_root=root, report_path=tmp_path / "r.md")


def test_phase3_saves_model(tmp_path: Path):
    bucket = _make_bucket_two_modes(tmp_path)
    model_path = tmp_path / "rf.pkl"
    run(bucket_root=bucket, report_path=tmp_path / "r.md",
        n_splits=4, model_save_path=model_path)
    assert model_path.exists() and model_path.stat().st_size > 0
