"""Persist a simulated DagRun to a ``LogStorage`` backend."""

from __future__ import annotations

import json
import random
from datetime import timedelta

from faker import Faker

from log_rca.datagen.failure_modes import RETRY_PLACEHOLDER
from log_rca.datagen.log_format import emit_task_log
from log_rca.datagen.simulator import DagRun
from log_rca.storage import LogStorage


def write_run(
    *,
    dr: DagRun,
    storage: LogStorage,
    logs_prefix: str,
    rng: random.Random,
    fake: Faker,
) -> int:
    """Write every attempt of every task in ``dr``. Returns the file count."""
    count = 0
    for task in dr.tasks:
        for attempt in range(1, task.attempts + 1):
            is_final = attempt == task.attempts
            outcome = task.outcome if is_final else "FAILED"
            failure = task.failure if is_final else RETRY_PLACEHOLDER
            body = emit_task_log(
                dag_id=dr.dag_id,
                run_id=dr.run_id,
                task_id=task.task_id,
                attempt=attempt,
                start=task.start + timedelta(seconds=(attempt - 1) * 5),
                duration_s=task.duration_s,
                outcome=outcome,
                failure=failure,
                rng=rng,
                fake=fake,
            )
            key = (
                f"{logs_prefix}/"
                f"dag_id={dr.dag_id}/"
                f"run_id={dr.run_id}/"
                f"task_id={task.task_id}/"
                f"attempt={attempt}.log"
            )
            storage.write_text(key, body)
            count += 1
    return count


def write_truth(*, dr: DagRun, storage: LogStorage, truth_key: str) -> None:
    """Append one ground-truth JSONL line for this run."""
    storage.append_line(truth_key, json.dumps({
        "dag_id": dr.dag_id,
        "run_id": dr.run_id,
        "outcome": dr.overall_outcome,
        "failure_mode": dr.failure_mode,
        "task_count": len(dr.tasks),
    }))
