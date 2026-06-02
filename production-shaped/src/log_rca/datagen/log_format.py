"""Airflow-style log-line builders + per-attempt log body.

Pure functions; no I/O. ``emit_task_log`` returns the full body of one
``attempt=<n>.log`` file as a single string.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from faker import Faker

from log_rca.datagen.failure_modes import FailureMode


def ts(dt: datetime) -> str:
    """Airflow-style timestamp: ``[2026-06-02T10:30:15.123+0000]``."""
    return f"[{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{dt.microsecond // 1000:03d}+0000]"


def line(t: datetime, src: str, level: str, msg: str) -> str:
    return f"{ts(t)} {{{src}}} {level} - {msg}"


# ─── task-specific flavour lines ────────────────────────────────────────────

def task_flavour(task_id: str, rng: random.Random, fake: Faker) -> list[str]:
    """3-5 INFO lines that read like the task is doing real work."""
    tid = task_id.lower()
    if any(k in tid for k in ("extract", "pull", "fetch", "consume")):
        pool = [
            f"Opening connection to source: {fake.hostname()}",
            f"Fetched {rng.randint(1000, 5_000_000):,} rows from source table",
            f"Read {rng.randint(10, 9999):,} MB from upstream",
            "Source connection closed cleanly",
        ]
    elif any(k in tid for k in ("validate", "screen")):
        pool = [
            f"Loaded validation rules: {rng.randint(20, 200)} active",
            f"Validated batch: {rng.randint(900, 1000)}/{rng.randint(1000, 1001)} passed",
            "Quarantine bucket empty -- proceeding",
        ]
    elif any(k in tid for k in (
        "transform", "enrich", "feature", "compute", "join",
        "rebalance", "match", "settle", "reconcile", "attribute",
    )):
        pool = [
            f"Transform stage starting with {rng.randint(4, 64)} workers",
            f"Cached intermediate dataset: {rng.randint(100, 9999)} MB",
            f"Joined {rng.randint(2, 8)} datasets",
        ]
    elif any(k in tid for k in (
        "load", "write", "persist", "post", "submit", "route", "execute",
    )):
        pool = [
            f"Target table: project-prod."
            f"{rng.choice(['risk','finance','crm','ml'])}.tbl_{fake.word()}",
            f"Inserting {rng.randint(1000, 500_000):,} rows",
            "Commit successful",
        ]
    elif any(k in tid for k in ("score", "model", "rule")):
        pool = [
            f"Loaded model artifact: gs://models/{fake.word()}_v{rng.randint(1,20)}.joblib",
            f"Scored {rng.randint(1000, 200_000):,} examples",
            f"Mean prediction: {rng.uniform(0.01, 0.99):.4f}",
        ]
    else:
        pool = [
            f"Sending notification to channel #{fake.word()}-ops",
            "Notification delivered (status=200)",
            f"Step {rng.randint(1, 9)} starting",
            f"Processed batch of {rng.randint(100, 10_000)} items",
            "Step finished cleanly",
        ]
    return rng.sample(pool, k=min(rng.randint(3, 5), len(pool)))


# ─── full attempt body ─────────────────────────────────────────────────────

def emit_task_log(
    *,
    dag_id: str,
    run_id: str,
    task_id: str,
    attempt: int,
    start: datetime,
    duration_s: int,
    outcome: str,
    failure: FailureMode | None,
    rng: random.Random,
    fake: Faker,
) -> str:
    """Build the body of one ``attempt=<n>.log`` file.

    ``outcome`` is "SUCCESS" or "FAILED". When FAILED, ``failure`` is required.
    """
    if outcome == "FAILED" and failure is None:
        raise ValueError("FAILED outcome requires a FailureMode")

    lines: list[str] = []
    t = start

    def step(seconds: float = 0.1) -> datetime:
        nonlocal t
        t = t + timedelta(seconds=seconds)
        return t

    lines.append(line(step(), "taskinstance.py:1216", "INFO",
        f"Dependencies all met for <TaskInstance: {dag_id}.{task_id} {run_id} [queued]>"))
    lines.append(line(step(0.05), "taskinstance.py:1416", "INFO",
        f"Starting attempt {attempt} of 3"))
    lines.append(line(step(0.02), "taskinstance.py:1437", "INFO",
        f"Executing <Task(PythonOperator): {task_id}> on {start.isoformat()}"))
    lines.append(line(step(0.03), "standard_task_runner.py:55", "INFO",
        f"Started process {rng.randint(1000, 99999)} to run task"))
    lines.append(line(step(0.05), "logging_mixin.py:137", "INFO",
        f"Running task on host airflow-worker-{rng.randint(1,8)}.composer.internal"))

    for msg in task_flavour(task_id, rng, fake):
        lines.append(line(step(rng.uniform(0.5, 3.0)), "logging_mixin.py:137", "INFO", msg))

    t = start + timedelta(seconds=max(duration_s - 1, 1))

    if outcome == "FAILED":
        assert failure is not None
        lines.append(line(step(), "taskinstance.py:1851", "ERROR",
            "Task failed with exception"))
        for raw in failure.error_lines:
            lines.append(raw.format(
                dag_id=dag_id, task_id=task_id,
                lineno=rng.randint(20, 200),
                bucket=f"bank-data-{rng.choice(['prod','staging'])}-{rng.randint(1,9)}",
                obj=f"warehouse/{dag_id}/{run_id.split('__')[1][:10]}/"
                    f"part-{rng.randint(0,99):05d}.parquet",
                shard=rng.randint(1, 6),
                ip=rng.randint(0, 255), ip2=rng.randint(0, 255),
            ))
        lines.append(line(step(0.05), "taskinstance.py:1408", "INFO",
            f"Marking task as FAILED. dag_id={dag_id}, task_id={task_id}, "
            f"execution_date={start.isoformat()}, start_date={start.isoformat()}, "
            f"end_date={t.isoformat()}"))
    else:
        lines.append(line(step(), "python.py:152", "INFO",
            f"Done. Returned value was: {{'rows': {rng.randint(1000, 5_000_000)}}}"))
        lines.append(line(step(0.05), "taskinstance.py:1408", "INFO",
            f"Marking task as SUCCESS. dag_id={dag_id}, task_id={task_id}, "
            f"execution_date={start.isoformat()}, start_date={start.isoformat()}, "
            f"end_date={t.isoformat()}"))

    return "\n".join(lines) + "\n"
