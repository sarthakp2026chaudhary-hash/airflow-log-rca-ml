"""Tests for the SyntheticAirflowLoader (dataset 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from log_rca.datasets import SyntheticAirflowLoader, SyntheticAirflowRecord


@pytest.fixture
def bucket_with_one_run(tmp_path: Path) -> Path:
    """Hand-craft a minimal bucket with one SUCCESS task log and one truth row."""
    root = tmp_path / "bucket"
    log = (
        root / "airflow-logs" / "dag_id=etl_customer_daily"
        / "run_id=scheduled__2026-01-15T10-30-00+0000"
        / "task_id=extract" / "attempt=1.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "[2026-01-15T10:30:15.123+0000] {taskinstance.py:1216} INFO - Dependencies all met\n"
        "[2026-01-15T10:30:16.000+0000] {taskinstance.py:1408} INFO - "
        "Marking task as SUCCESS. dag_id=etl_customer_daily, task_id=extract\n",
        encoding="utf-8",
    )
    (root / "_truth.jsonl").write_text(
        json.dumps({
            "dag_id": "etl_customer_daily",
            "run_id": "scheduled__2026-01-15T10-30-00+0000",
            "outcome": "SUCCESS",
            "failure_mode": "",
            "task_count": 1,
        }) + "\n",
        encoding="utf-8",
    )
    return root


def test_load_truth_parses_jsonl(bucket_with_one_run: Path):
    loader = SyntheticAirflowLoader(bucket_with_one_run)
    truth = loader.load_truth()
    assert "scheduled__2026-01-15T10-30-00+0000" in truth
    tr = truth["scheduled__2026-01-15T10-30-00+0000"]
    assert tr.outcome == "SUCCESS"
    assert tr.dag_id == "etl_customer_daily"
    assert tr.task_count == 1


def test_iter_log_files_finds_log(bucket_with_one_run: Path):
    loader = SyntheticAirflowLoader(bucket_with_one_run)
    files = list(loader.iter_log_files())
    assert len(files) == 1
    assert files[0].name == "attempt=1.log"


def test_load_records_yields_typed_records(bucket_with_one_run: Path):
    loader = SyntheticAirflowLoader(bucket_with_one_run)
    records = list(loader.load_records())
    assert len(records) == 2
    for r in records:
        assert isinstance(r, SyntheticAirflowRecord)
        assert r.dag_id == "etl_customer_daily"
        assert r.run_id == "scheduled__2026-01-15T10-30-00+0000"
        assert r.task_id == "extract"
        assert r.attempt == 1
        assert r.outcome == "SUCCESS"
        assert r.failure_mode == ""
    assert records[0].level == "INFO"
    assert "Dependencies all met" in records[0].message
    assert "Marking task as SUCCESS" in records[1].message


def test_unknown_outcome_when_no_truth_row(tmp_path: Path):
    root = tmp_path / "bucket"
    log = (
        root / "airflow-logs" / "dag_id=x" / "run_id=scheduled__orphan"
        / "task_id=y" / "attempt=1.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "[2026-01-01T00:00:00.000+0000] {a.py:1} INFO - hello\n",
        encoding="utf-8",
    )
    loader = SyntheticAirflowLoader(root)
    rec = next(iter(loader.load_records()))
    assert rec.outcome == "UNKNOWN"
    assert rec.failure_mode == ""


def test_missing_bucket_returns_empty_iter(tmp_path: Path):
    loader = SyntheticAirflowLoader(tmp_path / "does-not-exist")
    assert list(loader.iter_log_files()) == []
    assert list(loader.load_records()) == []
    assert loader.load_truth() == {}


def test_summary_counts_runs_and_failures(bucket_with_one_run: Path):
    loader = SyntheticAirflowLoader(bucket_with_one_run)
    s = loader.summary()
    assert s["files"] == 1
    assert s["lines"] == 2
    assert s["runs"] == 1
    assert s["successes"] == 1
    assert s["failures"] == 0


def test_continuation_lines_kept_with_blank_fields(tmp_path: Path):
    """Non-matching lines (traceback bodies) should still yield records."""
    root = tmp_path / "bucket"
    log = (
        root / "airflow-logs" / "dag_id=d" / "run_id=scheduled__t"
        / "task_id=t" / "attempt=1.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "[2026-01-01T00:00:00.000+0000] {a.py:1} ERROR - Task failed with exception\n"
        "Traceback (most recent call last):\n"
        "MemoryError\n",
        encoding="utf-8",
    )
    loader = SyntheticAirflowLoader(root)
    records = list(loader.load_records())
    assert len(records) == 3
    assert records[1].level == ""
    assert records[1].message == "Traceback (most recent call last):"
    assert records[2].message == "MemoryError"
