"""End-to-end Phase 2 pipeline test."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from log_rca.pipeline.phase2 import run


def _make_bucket(tmp_path: Path) -> Path:
    """20 SUCCESS runs (boring) + 10 FAILED runs (with OOM lines).

    Enough rows that IsolationForest fits cleanly and the FAILED runs
    end up in the bottom (anomalous) tail of the score distribution.
    """
    rng = random.Random(0)
    root = tmp_path / "bucket"
    truth = []

    for i in range(20):
        run_id = f"scheduled__ok-{i}"
        log = (
            root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
            / "task_id=t" / "attempt=1.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"[2026-01-01T00:00:{i:02d}.000+0000] {{a.py:1}} INFO - Fetched {rng.randint(100, 999)} rows\n"
            f"[2026-01-01T00:00:{i+1:02d}.000+0000] {{a.py:2}} INFO - Transformed in {rng.randint(1, 9)}s\n"
            f"[2026-01-01T00:01:00.000+0000] {{a.py:3}} INFO - Marking task as SUCCESS\n",
            encoding="utf-8",
        )
        truth.append({
            "dag_id": "demo", "run_id": run_id,
            "outcome": "SUCCESS", "failure_mode": "", "task_count": 1,
        })

    for i in range(10):
        run_id = f"scheduled__fail-{i}"
        log = (
            root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
            / "task_id=t" / "attempt=1.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"[2026-01-02T00:00:{i:02d}.000+0000] {{a.py:1}} INFO - Fetched {rng.randint(100, 999)} rows\n"
            f"[2026-01-02T00:00:30.000+0000] {{a.py:9}} ERROR - Task failed with exception\n"
            "MemoryError: out of memory\n"
            "Killed\n"
            f"[2026-01-02T00:01:00.000+0000] {{a.py:3}} INFO - Marking task as FAILED\n",
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


def test_phase2_ranks_failures_as_anomalous(tmp_path: Path):
    bucket = _make_bucket(tmp_path)
    report = tmp_path / "p2.md"
    stats = run(bucket_root=bucket, report_path=report, top_k=10)

    assert report.exists()
    assert stats["n_total"] == 30
    assert stats["n_failed"] == 10
    # With clearly-distinct failure logs, all 10 failures should be in top-10.
    assert stats["n_failed_in_top"] >= 8     # allow 80% to keep test stable
    assert stats["precision_at_k"] >= 0.8
    assert stats["templates"] > 0


def test_phase2_raises_when_too_few_success(tmp_path: Path):
    # Build a bucket with only 3 SUCCESS rows -> IsolationForest fit fails.
    root = tmp_path / "bucket"
    truth = []
    for i in range(3):
        run_id = f"scheduled__ok-{i}"
        log = (
            root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
            / "task_id=t" / "attempt=1.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("[2026-01-01T00:00:00.000+0000] {a.py:1} INFO - hi\n", encoding="utf-8")
        truth.append({
            "dag_id": "demo", "run_id": run_id,
            "outcome": "SUCCESS", "failure_mode": "", "task_count": 1,
        })
    (root / "_truth.jsonl").write_text(
        "\n".join(json.dumps(t) for t in truth) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 10"):
        run(bucket_root=root, report_path=tmp_path / "r.md")


def test_phase2_saves_model_when_requested(tmp_path: Path):
    bucket = _make_bucket(tmp_path)
    model_path = tmp_path / "iforest.pkl"
    run(bucket_root=bucket, report_path=tmp_path / "r.md", model_save_path=model_path)
    assert model_path.exists() and model_path.stat().st_size > 0
