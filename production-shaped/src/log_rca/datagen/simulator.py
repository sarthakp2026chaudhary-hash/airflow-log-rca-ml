"""DagRun-level simulation: decide what fails, where, and how long it takes.

The simulator stays pure — it builds typed ``DagRun`` objects in memory.
The actual disk I/O is owned by a ``LogStorage`` backend (see ``writer``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from log_rca.datagen.failure_modes import (
    FAILURE_MODES,
    FailureMode,
    weights,
)


@dataclass
class TaskOutcome:
    task_id: str
    attempts: int
    start: datetime
    duration_s: int
    outcome: str
    failure: FailureMode | None


@dataclass
class DagRun:
    dag_id: str
    run_id: str
    run_start: datetime
    tasks: list[TaskOutcome] = field(default_factory=list)
    overall_outcome: str = "SUCCESS"
    failure_mode: str = ""


def simulate_run(
    *,
    dag_id: str,
    tasks: list[str],
    run_start: datetime,
    failure_rate: float,
    rng: random.Random,
) -> DagRun:
    """Build a typed DagRun. No I/O."""
    # Real Airflow run_ids contain ':' (e.g. scheduled__2026-01-15T10:30:00+00:00)
    # but ':' is illegal in Windows NTFS paths, so we use '-' here. The shape
    # is otherwise identical: parse-able by splitting on 'scheduled__'.
    run_id = f"scheduled__{run_start.strftime('%Y-%m-%dT%H-%M-%S')}+0000"
    dr = DagRun(dag_id=dag_id, run_id=run_id, run_start=run_start)

    will_fail = rng.random() < failure_rate
    fail_at_idx = rng.randrange(len(tasks)) if will_fail else -1
    chosen_failure: FailureMode | None = (
        rng.choices(FAILURE_MODES, weights=weights(), k=1)[0]
        if will_fail else None
    )

    t_cursor = run_start
    for i, task_id in enumerate(tasks):
        dur = rng.choices(
            population=[rng.randint(5, 60), rng.randint(60, 300), rng.randint(300, 1800)],
            weights=[0.6, 0.3, 0.1],
            k=1,
        )[0]
        if i == fail_at_idx:
            attempts = rng.randint(1, 3)
            dr.tasks.append(
                TaskOutcome(task_id, attempts, t_cursor, dur, "FAILED", chosen_failure)
            )
            dr.overall_outcome = "FAILED"
            dr.failure_mode = chosen_failure.code  # type: ignore[union-attr]
            break
        dr.tasks.append(TaskOutcome(task_id, 1, t_cursor, dur, "SUCCESS", None))
        t_cursor = t_cursor + timedelta(seconds=dur + rng.randint(1, 10))
    return dr
