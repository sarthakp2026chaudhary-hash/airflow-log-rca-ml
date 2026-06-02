"""End-to-end Phase 4 pipeline test (stub backend; no API calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from log_rca.pipeline.phase4 import run


def _make_bucket(tmp_path: Path) -> Path:
    """Tiny bucket with 5 successes + 3 OOM failures."""
    root = tmp_path / "bucket"
    truth = []
    for i in range(5):
        run_id = f"scheduled__ok-{i}"
        log = (root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
               / "task_id=t" / "attempt=1.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("[2026-01-01T00:00:00.000+0000] {a:1} INFO - ok\n", encoding="utf-8")
        truth.append({"dag_id": "demo", "run_id": run_id, "outcome": "SUCCESS",
                       "failure_mode": "", "task_count": 1})
    for i in range(3):
        run_id = f"scheduled__fail-{i}"
        log = (root / "airflow-logs" / "dag_id=demo" / f"run_id={run_id}"
               / "task_id=t" / "attempt=1.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "[2026-01-02T00:00:00.000+0000] {a:1} ERROR - Task failed with exception\n"
            "MemoryError: out of memory\n", encoding="utf-8",
        )
        truth.append({"dag_id": "demo", "run_id": run_id, "outcome": "FAILED",
                       "failure_mode": "OOM", "task_count": 1})
    (root / "_truth.jsonl").write_text(
        "\n".join(json.dumps(t) for t in truth) + "\n", encoding="utf-8",
    )
    return root


def test_phase4_runs_in_stub_mode(tmp_path: Path):
    bucket = _make_bucket(tmp_path)
    report = tmp_path / "p4.md"
    cache = tmp_path / "cache.json"
    stats = run(
        bucket_root=bucket, report_path=report,
        backend_name="stub", n=3, cache_path=cache,
    )
    assert report.exists()
    body = report.read_text(encoding="utf-8")
    assert "Stub mode" in body
    assert "### Root cause" in body
    assert "OOM" in body
    assert stats["backend"] == "stub"
    assert stats["n_results"] == 3
    # All 3 failures share (dag, mode, templates) so 2 of them hit the
    # in-batch cache after the first one populates it — this is correct.
    assert stats["cache_hits"] == 2
    assert cache.exists()


def test_phase4_second_run_hits_cache_on_disk(tmp_path: Path):
    bucket = _make_bucket(tmp_path)
    cache = tmp_path / "cache.json"
    run(bucket_root=bucket, report_path=tmp_path / "r1.md",
        backend_name="stub", n=3, cache_path=cache)
    # Re-run from a fresh on-disk cache -> all 3 hit (loaded from json)
    stats = run(bucket_root=bucket, report_path=tmp_path / "r2.md",
                backend_name="stub", n=3, cache_path=cache)
    assert stats["cache_hits"] == 3


def test_phase4_all_flag_processes_every_failure(tmp_path: Path):
    bucket = _make_bucket(tmp_path)
    stats = run(bucket_root=bucket, report_path=tmp_path / "r.md",
                backend_name="stub", n=None,
                cache_path=tmp_path / "c.json")
    assert stats["n_results"] == 3


def test_phase4_raises_when_no_failures(tmp_path: Path):
    root = tmp_path / "bucket"
    truth = [{"dag_id": "d", "run_id": "scheduled__s",
              "outcome": "SUCCESS", "failure_mode": "", "task_count": 1}]
    log = (root / "airflow-logs" / "dag_id=d" / "run_id=scheduled__s"
           / "task_id=t" / "attempt=1.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("[2026-01-01T00:00:00.000+0000] {a:1} INFO - ok\n", encoding="utf-8")
    (root / "_truth.jsonl").write_text("\n".join(json.dumps(t) for t in truth) + "\n",
                                       encoding="utf-8")
    with pytest.raises(ValueError, match="No FAILED runs"):
        run(bucket_root=root, report_path=tmp_path / "r.md",
            backend_name="stub", cache_path=tmp_path / "c.json")
