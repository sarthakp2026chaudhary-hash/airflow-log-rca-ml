"""Tests for log-line and per-attempt body generation."""

from __future__ import annotations

import re
from datetime import datetime

import pytest

from log_rca.datagen.failure_modes import by_code
from log_rca.datagen.log_format import emit_task_log, line, ts


def test_ts_has_airflow_format():
    t = datetime(2026, 6, 2, 10, 30, 15, 123_456)
    assert ts(t) == "[2026-06-02T10:30:15.123+0000]"


def test_line_format():
    t = datetime(2026, 6, 2, 10, 30, 15, 0)
    out = line(t, "taskinstance.py:1216", "INFO", "hello")
    assert out == "[2026-06-02T10:30:15.000+0000] {taskinstance.py:1216} INFO - hello"


def test_emit_success_log_marks_success(rng, fake, fixed_start):
    body = emit_task_log(
        dag_id="etl_customer_daily",
        run_id="scheduled__2026-01-15T10:30:00+00:00",
        task_id="extract_crm",
        attempt=1,
        start=fixed_start,
        duration_s=30,
        outcome="SUCCESS",
        failure=None,
        rng=rng,
        fake=fake,
    )
    assert "Marking task as SUCCESS" in body
    assert "Marking task as FAILED" not in body
    assert "Starting attempt 1 of 3" in body
    # every line should start with a timestamp bracket
    for ln in body.strip().splitlines():
        assert ln.startswith("[2026"), f"unexpected line: {ln!r}"


def test_emit_failed_log_contains_traceback(rng, fake, fixed_start):
    failure = by_code("OOM")
    assert failure is not None
    body = emit_task_log(
        dag_id="risk_pnl_hourly",
        run_id="scheduled__2026-01-15T10:30:00+00:00",
        task_id="compute_pnl",
        attempt=2,
        start=fixed_start,
        duration_s=60,
        outcome="FAILED",
        failure=failure,
        rng=rng,
        fake=fake,
    )
    assert "Marking task as FAILED" in body
    assert "MemoryError" in body
    assert "exit status 137" in body


def test_emit_failed_requires_failure(rng, fake, fixed_start):
    with pytest.raises(ValueError, match="FAILED outcome requires"):
        emit_task_log(
            dag_id="x", run_id="r", task_id="t", attempt=1,
            start=fixed_start, duration_s=10,
            outcome="FAILED", failure=None,
            rng=rng, fake=fake,
        )


def test_emit_log_renders_template_placeholders(rng, fake, fixed_start):
    """Failure-mode error lines with ``{dag_id}`` etc. should be substituted."""
    failure = by_code("CONN_REFUSED")
    assert failure is not None
    body = emit_task_log(
        dag_id="loan_originations_etl",
        run_id="scheduled__2026-01-15T10:30:00+00:00",
        task_id="extract",
        attempt=1,
        start=fixed_start,
        duration_s=20,
        outcome="FAILED",
        failure=failure,
        rng=rng,
        fake=fake,
    )
    # No unrendered placeholders should remain.
    assert "{dag_id}" not in body
    assert "{task_id}" not in body
    # The dag_id and task_id should appear at least once.
    assert "loan_originations_etl" in body
    assert "extract" in body
    # And the failure-specific message must be there.
    assert "Connection refused" in body
    # IP placeholders should have been replaced with digits.
    assert re.search(r"10\.0\.\d+\.\d+", body)
