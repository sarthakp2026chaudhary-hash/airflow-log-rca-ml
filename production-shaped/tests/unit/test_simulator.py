"""Tests for the DagRun simulator + writer."""

from __future__ import annotations

import json
import random

from log_rca.datagen.dag_catalogue import DAGS
from log_rca.datagen.simulator import simulate_run
from log_rca.datagen.writer import write_run, write_truth


def test_simulate_run_always_succeeds_when_failure_rate_zero(fixed_start):
    rng = random.Random(0)
    dr = simulate_run(
        dag_id="etl_customer_daily",
        tasks=DAGS["etl_customer_daily"],
        run_start=fixed_start,
        failure_rate=0.0,
        rng=rng,
    )
    assert dr.overall_outcome == "SUCCESS"
    assert dr.failure_mode == ""
    assert len(dr.tasks) == len(DAGS["etl_customer_daily"])
    for t in dr.tasks:
        assert t.outcome == "SUCCESS"
        assert t.attempts == 1


def test_simulate_run_always_fails_when_failure_rate_one(fixed_start):
    rng = random.Random(0)
    dr = simulate_run(
        dag_id="risk_pnl_hourly",
        tasks=DAGS["risk_pnl_hourly"],
        run_start=fixed_start,
        failure_rate=1.0,
        rng=rng,
    )
    assert dr.overall_outcome == "FAILED"
    assert dr.failure_mode != ""
    # the simulator stops at the failing task -- so fewer tasks recorded
    assert any(t.outcome == "FAILED" for t in dr.tasks)


def test_run_id_format(fixed_start):
    rng = random.Random(0)
    dr = simulate_run(
        dag_id="kyc_refresh",
        tasks=DAGS["kyc_refresh"],
        run_start=fixed_start,
        failure_rate=0.0,
        rng=rng,
    )
    assert dr.run_id == "scheduled__2026-01-15T10-30-00+0000"


def test_write_run_creates_files_under_logs_prefix(storage, rng, fake, fixed_start):
    dr = simulate_run(
        dag_id="etl_customer_daily",
        tasks=DAGS["etl_customer_daily"],
        run_start=fixed_start,
        failure_rate=0.0,
        rng=rng,
    )
    count = write_run(
        dr=dr, storage=storage, logs_prefix="airflow-logs", rng=rng, fake=fake,
    )
    assert count == len(dr.tasks)  # all SUCCESS, one attempt each
    keys = sorted(storage.iter_keys("airflow-logs"))
    assert len(keys) == count
    for k in keys:
        assert k.startswith("airflow-logs/dag_id=etl_customer_daily/")
        assert k.endswith(".log")


def test_write_run_emits_extra_attempts_on_failure(storage, fake, fixed_start):
    # Use a fixed seed that gives a multi-attempt failure
    rng = random.Random(7)
    dr = simulate_run(
        dag_id="card_txn_settlement",
        tasks=DAGS["card_txn_settlement"],
        run_start=fixed_start,
        failure_rate=1.0,
        rng=rng,
    )
    write_run(dr=dr, storage=storage, logs_prefix="airflow-logs", rng=rng, fake=fake)
    # Find the failing task and check attempt files match
    failing_task = next(t for t in dr.tasks if t.outcome == "FAILED")
    attempts_present = sum(
        1 for k in storage.iter_keys(
            f"airflow-logs/dag_id={dr.dag_id}/run_id={dr.run_id}/task_id={failing_task.task_id}"
        )
    )
    assert attempts_present == failing_task.attempts


def test_write_truth_appends_json_line(storage, rng, fixed_start):
    dr = simulate_run(
        dag_id="aml_monitor_daily",
        tasks=DAGS["aml_monitor_daily"],
        run_start=fixed_start,
        failure_rate=1.0,
        rng=rng,
    )
    write_truth(dr=dr, storage=storage, truth_key="_truth.jsonl")
    body = storage.read_text("_truth.jsonl").strip()
    rec = json.loads(body)
    assert rec["dag_id"] == "aml_monitor_daily"
    assert rec["outcome"] == "FAILED"
    assert rec["failure_mode"]
    assert isinstance(rec["task_count"], int)
