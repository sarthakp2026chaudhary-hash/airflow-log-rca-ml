"""
generate_logs.py — Phase 0 of airflow-log-rca-ml (mini-project)
================================================================

Generates a synthetic corpus of Airflow task logs that mimics what a large
bank's Cloud Composer setup would dump into GCS. The output layout mirrors
the real Airflow + Composer convention:

    fake_gcs_bucket/
      airflow-logs/
        dag_id=<dag>/
          run_id=scheduled__<iso8601>/
            task_id=<task>/
              attempt=<n>.log
      _truth.jsonl            <- ground-truth labels (used in Phase 3)

Read top-to-bottom. The script is intentionally a single file with sectioned
comment banners. The production-shaped sibling splits the same logic across
src/log_rca/datagen/ if you want to see what the "engineered" version looks
like.

Design choices, briefly
-----------------------
* ~500 DAG runs across the last 14 days, ~30 % fail.
* 20 distinct bank-flavoured DAGs (etl_customer_daily, risk_pnl_hourly, ...).
* Failures drawn from a fixed taxonomy (OOM, BQ_QUOTA, SCHEMA_MISMATCH, ...)
  so later ML phases have a known signal to find.
* Log lines follow the real Airflow format:
    [2026-06-02T10:30:15.123+0000] {taskinstance.py:1216} INFO - <message>
* Failures emit a multi-line traceback right before the "Marking task as
  FAILED" footer. Retries before the final failure carry a generic transient
  error (mimics how flaky tasks behave on Composer).
* Seeded RNG -> the corpus is deterministic for a given SEED.

Run
---
    python generate_logs.py

Output is ~2,000+ log files; takes 5-15 s on a laptop.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from faker import Faker
except ImportError:
    print(
        "ERROR: 'faker' not installed. Run:  pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

SEED = 42
DAYS_BACK = 14
TOTAL_RUNS = 500
FAILURE_RATE = 0.30
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "fake_gcs_bucket"

fake = Faker()
Faker.seed(SEED)
rng = random.Random(SEED)


# ════════════════════════════════════════════════════════════════════════════
# DAG catalogue — 20 bank-flavoured DAGs and their tasks
# ════════════════════════════════════════════════════════════════════════════

DAGS: dict[str, list[str]] = {
    "etl_customer_daily":         ["extract_crm", "validate", "transform", "load_bq", "notify"],
    "risk_pnl_hourly":            ["fetch_positions", "fetch_marks", "compute_pnl", "persist"],
    "kyc_refresh":                ["pull_kyc", "screen_sanctions", "score", "alert_ops"],
    "mortgage_scoring":           ["extract_apps", "feature_eng", "score_model", "write_decisions"],
    "fraud_detection_streaming":  ["consume_kafka", "enrich", "score", "publish_alerts"],
    "regulatory_basel_report":    ["aggregate_positions", "compute_rwa", "generate_xbrl", "submit"],
    "card_txn_settlement":        ["pull_txns", "match", "settle", "reconcile"],
    "branch_metrics_daily":       ["extract_branches", "aggregate", "publish_dashboard"],
    "loan_originations_etl":      ["extract", "validate", "enrich_credit_bureau", "load_bq"],
    "aml_monitor_daily":          ["fetch_txns", "rule_engine", "score_ml", "create_cases"],
    "treasury_cash_position":     ["pull_balances", "fx_convert", "aggregate", "report"],
    "wealth_portfolio_rebalance": ["fetch_holdings", "rebalance", "execute_orders"],
    "credit_card_rewards":        ["pull_spend", "compute_rewards", "post_to_accounts"],
    "marketing_campaign_attrib":  ["pull_clicks", "join_conversions", "attribute", "publish"],
    "deposit_interest_accrual":   ["pull_balances", "compute_accrual", "post_gl"],
    "swift_message_parser":       ["consume", "parse", "validate", "route"],
    "collateral_valuation":       ["fetch_collateral", "mark_to_market", "haircut", "persist"],
    "stress_test_scenarios":      ["load_scenarios", "run_models", "aggregate", "report"],
    "customer_360_refresh":       ["pull_sources", "resolve_entities", "build_profile"],
    "operational_loss_etl":       ["pull_incidents", "classify", "aggregate", "report"],
}


# ════════════════════════════════════════════════════════════════════════════
# Failure taxonomy — what each crash looks like in the log
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FailureMode:
    code: str               # short label, e.g. "OOM"
    weight: float           # relative frequency among failures
    error_lines: list[str]  # lines emitted just before the FAILED footer.
                            # Templated with .format(dag_id=, task_id=, ...)


FAILURE_MODES: list[FailureMode] = [
    FailureMode("OOM", 0.18, [
        "[ERROR] - Task process exceeded memory limit (8192 MiB)",
        "Killed",
        "MemoryError",
        "subprocess.CalledProcessError: Command 'spark-submit --master yarn ...' "
        "returned non-zero exit status 137.",
    ]),
    FailureMode("TASK_TIMEOUT", 0.14, [
        "[ERROR] - Task exceeded timeout of 3600 seconds",
        "Traceback (most recent call last):",
        "  File \"/opt/airflow/dags/{dag_id}.py\", line {lineno}, in {task_id}",
        "    result = run_query(query)",
        "airflow.exceptions.AirflowTaskTimeout: Timeout",
    ]),
    FailureMode("GCS_PERMISSION", 0.10, [
        "Traceback (most recent call last):",
        "  File \"/usr/local/lib/python3.10/site-packages/google/cloud/storage/blob.py\","
        " line 1234, in download_to_filename",
        "    self._do_download(transport, file_obj, download_url, headers, start, end, raw_download)",
        "google.api_core.exceptions.Forbidden: 403 GET "
        "https://storage.googleapis.com/{bucket}/{obj}: ",
        "Caller does not have storage.objects.get access to the Google Cloud Storage object.",
    ]),
    FailureMode("BQ_QUOTA", 0.13, [
        "Traceback (most recent call last):",
        "  File \"/usr/local/lib/python3.10/site-packages/google/cloud/bigquery/client.py\","
        " line 3210, in query",
        "    return _job_helpers.query_jobs_query(self, request, retry, timeout, job_retry)",
        "google.api_core.exceptions.Forbidden: 403 quotaExceeded: Exceeded rate limits: ",
        "too many concurrent queries for this project_and_region. For more information, see "
        "https://cloud.google.com/bigquery/docs/troubleshoot-quotas",
    ]),
    FailureMode("SCHEMA_MISMATCH", 0.12, [
        "Traceback (most recent call last):",
        "  File \"/opt/airflow/dags/{dag_id}.py\", line {lineno}, in {task_id}",
        "    table.write(df)",
        "  File \"/usr/local/lib/python3.10/site-packages/pyarrow/parquet/core.py\","
        " line 2123, in write_table",
        "    writer.write_table(table)",
        "pyarrow.lib.ArrowInvalid: Schema mismatch: expected column 'customer_id' "
        "to be int64, got string",
    ]),
    FailureMode("UPSTREAM_MISSING", 0.10, [
        "[INFO] - Dependency 'Trigger Rule' FAILED: Task's trigger rule 'all_success' "
        "requires all upstream tasks to have succeeded, but found 1 non-success(es).",
        "airflow.exceptions.AirflowSkipException: Skipping because upstream task failed",
    ]),
    FailureMode("CONN_REFUSED", 0.13, [
        "Traceback (most recent call last):",
        "  File \"/opt/airflow/dags/{dag_id}.py\", line {lineno}, in {task_id}",
        "    conn = psycopg2.connect(dsn)",
        "psycopg2.OperationalError: connection to server at \"db-prod-{shard}.internal\" "
        "(10.0.{ip}.{ip2}), port 5432 failed: ",
        "Connection refused",
        "        Is the server running on that host and accepting TCP/IP connections?",
    ]),
    FailureMode("KEY_ERROR", 0.10, [
        "Traceback (most recent call last):",
        "  File \"/opt/airflow/dags/{dag_id}.py\", line {lineno}, in {task_id}",
        "    val = payload['customer']['account']['balance']",
        "KeyError: 'balance'",
    ]),
]
_FM_WEIGHTS = [m.weight for m in FAILURE_MODES]

# Used for non-final attempts of a failing task (the "first try" before a retry).
_RETRY_PLACEHOLDER = FailureMode("__retry", 0, [
    "[ERROR] - Task instance is in a temporary failed state and will be retried",
    "Exception: transient error, will retry",
])


# ════════════════════════════════════════════════════════════════════════════
# Log line helpers
# ════════════════════════════════════════════════════════════════════════════

def _ts(dt: datetime) -> str:
    """Airflow-style timestamp: [2026-06-02T10:30:15.123+0000]"""
    return f"[{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{dt.microsecond // 1000:03d}+0000]"


def _line(t: datetime, src: str, level: str, msg: str) -> str:
    return f"{_ts(t)} {{{src}}} {level} - {msg}"


def _task_flavour(task_id: str) -> list[str]:
    """A few task-specific INFO lines so successful logs are not empty noise."""
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
    elif any(k in tid for k in ("transform", "enrich", "feature", "compute", "join", "rebalance", "match", "settle", "reconcile", "attribute")):
        pool = [
            f"Transform stage starting with {rng.randint(4, 64)} workers",
            f"Cached intermediate dataset: {rng.randint(100, 9999)} MB",
            f"Joined {rng.randint(2, 8)} datasets",
        ]
    elif any(k in tid for k in ("load", "write", "persist", "post", "submit", "route", "execute")):
        pool = [
            f"Target table: project-prod.{rng.choice(['risk','finance','crm','ml'])}"
            f".tbl_{fake.word()}",
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


# ════════════════════════════════════════════════════════════════════════════
# Per-attempt log body
# ════════════════════════════════════════════════════════════════════════════

def emit_task_log(
    dag_id: str,
    run_id: str,
    task_id: str,
    attempt: int,
    start: datetime,
    duration_s: int,
    outcome: str,                       # "SUCCESS" or "FAILED"
    failure: FailureMode | None,
) -> str:
    """Build the body of one attempt=<n>.log file."""
    lines: list[str] = []
    t = start

    def step(seconds: float = 0.1) -> datetime:
        nonlocal t
        t = t + timedelta(seconds=seconds)
        return t

    # Boilerplate header lines that Airflow always emits.
    lines.append(_line(step(), "taskinstance.py:1216", "INFO",
        f"Dependencies all met for <TaskInstance: {dag_id}.{task_id} {run_id} [queued]>"))
    lines.append(_line(step(0.05), "taskinstance.py:1416", "INFO",
        f"Starting attempt {attempt} of 3"))
    lines.append(_line(step(0.02), "taskinstance.py:1437", "INFO",
        f"Executing <Task(PythonOperator): {task_id}> on {start.isoformat()}"))
    lines.append(_line(step(0.03), "standard_task_runner.py:55", "INFO",
        f"Started process {rng.randint(1000, 99999)} to run task"))
    lines.append(_line(step(0.05), "logging_mixin.py:137", "INFO",
        f"Running task on host airflow-worker-{rng.randint(1,8)}.composer.internal"))

    # Task-flavoured body lines.
    for msg in _task_flavour(task_id):
        lines.append(_line(step(rng.uniform(0.5, 3.0)), "logging_mixin.py:137", "INFO", msg))

    # Jump to near the end of the task.
    t = start + timedelta(seconds=max(duration_s - 1, 1))

    if outcome == "FAILED":
        assert failure is not None
        lines.append(_line(step(), "taskinstance.py:1851", "ERROR",
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
        lines.append(_line(step(0.05), "taskinstance.py:1408", "INFO",
            f"Marking task as FAILED. dag_id={dag_id}, task_id={task_id}, "
            f"execution_date={start.isoformat()}, start_date={start.isoformat()}, "
            f"end_date={t.isoformat()}"))
    else:
        lines.append(_line(step(), "python.py:152", "INFO",
            f"Done. Returned value was: {{'rows': {rng.randint(1000, 5_000_000)}}}"))
        lines.append(_line(step(0.05), "taskinstance.py:1408", "INFO",
            f"Marking task as SUCCESS. dag_id={dag_id}, task_id={task_id}, "
            f"execution_date={start.isoformat()}, start_date={start.isoformat()}, "
            f"end_date={t.isoformat()}"))

    return "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════════════════
# DAG-run simulation
# ════════════════════════════════════════════════════════════════════════════

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


def simulate_run(dag_id: str, tasks: list[str], run_start: datetime) -> DagRun:
    # Real Airflow run_ids contain ':' (e.g. scheduled__2026-01-15T10:30:00+00:00)
    # but ':' is illegal in Windows NTFS paths, so we use '-' here. The shape
    # is otherwise identical: parse-able by splitting on 'scheduled__'.
    run_id = f"scheduled__{run_start.strftime('%Y-%m-%dT%H-%M-%S')}+0000"
    dr = DagRun(dag_id=dag_id, run_id=run_id, run_start=run_start)

    will_fail = rng.random() < FAILURE_RATE
    fail_at_idx = rng.randrange(len(tasks)) if will_fail else -1
    chosen_failure = (
        rng.choices(FAILURE_MODES, weights=_FM_WEIGHTS, k=1)[0]
        if will_fail else None
    )

    t_cursor = run_start
    for i, task_id in enumerate(tasks):
        # Most tasks short, some long, a few very long.
        dur = rng.choices(
            population=[rng.randint(5, 60), rng.randint(60, 300), rng.randint(300, 1800)],
            weights=[0.6, 0.3, 0.1],
            k=1,
        )[0]
        if i == fail_at_idx:
            attempts = rng.randint(1, 3)   # mimic Airflow retry behaviour
            dr.tasks.append(
                TaskOutcome(task_id, attempts, t_cursor, dur, "FAILED", chosen_failure)
            )
            dr.overall_outcome = "FAILED"
            dr.failure_mode = chosen_failure.code  # type: ignore[union-attr]
            break
        dr.tasks.append(TaskOutcome(task_id, 1, t_cursor, dur, "SUCCESS", None))
        t_cursor = t_cursor + timedelta(seconds=dur + rng.randint(1, 10))
    return dr


# ════════════════════════════════════════════════════════════════════════════
# Write the run to disk in fake-GCS layout
# ════════════════════════════════════════════════════════════════════════════

def write_run(dr: DagRun, root: Path) -> int:
    count = 0
    for task in dr.tasks:
        for attempt in range(1, task.attempts + 1):
            is_final = attempt == task.attempts
            outcome = task.outcome if is_final else "FAILED"
            failure = task.failure if is_final else _RETRY_PLACEHOLDER
            body = emit_task_log(
                dag_id=dr.dag_id,
                run_id=dr.run_id,
                task_id=task.task_id,
                attempt=attempt,
                start=task.start + timedelta(seconds=(attempt - 1) * 5),
                duration_s=task.duration_s,
                outcome=outcome,
                failure=failure,
            )
            path = (
                root / "airflow-logs"
                / f"dag_id={dr.dag_id}"
                / f"run_id={dr.run_id}"
                / f"task_id={task.task_id}"
                / f"attempt={attempt}.log"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            count += 1
    return count


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Generating ~{TOTAL_RUNS} DAG runs into {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    truth_path = OUTPUT_ROOT / "_truth.jsonl"

    dag_names = list(DAGS.keys())
    now = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    earliest = now - timedelta(days=DAYS_BACK)

    total_files = 0
    fail_counts: dict[str, int] = {}
    success_count = 0

    with truth_path.open("w", encoding="utf-8") as truth_file:
        for i in range(TOTAL_RUNS):
            dag_id = rng.choice(dag_names)
            tasks = DAGS[dag_id]
            offset = rng.random() * DAYS_BACK * 86400
            run_start = earliest + timedelta(seconds=offset)
            run_start = run_start.replace(microsecond=rng.randint(0, 999_999))

            dr = simulate_run(dag_id, tasks, run_start)
            total_files += write_run(dr, OUTPUT_ROOT)

            truth_file.write(json.dumps({
                "dag_id": dr.dag_id,
                "run_id": dr.run_id,
                "outcome": dr.overall_outcome,
                "failure_mode": dr.failure_mode,
                "task_count": len(dr.tasks),
            }) + "\n")

            if dr.overall_outcome == "FAILED":
                fail_counts[dr.failure_mode] = fail_counts.get(dr.failure_mode, 0) + 1
            else:
                success_count += 1

            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{TOTAL_RUNS} runs ({total_files} log files so far)")

    print(f"\nDone. {total_files} log files written under {OUTPUT_ROOT / 'airflow-logs'}")
    print(f"Ground truth: {truth_path}")
    print(f"\nOutcome breakdown: SUCCESS={success_count}, "
          f"FAILED={sum(fail_counts.values())}")
    print("Failure-mode breakdown:")
    for code in sorted(fail_counts):
        print(f"  {code:<18} {fail_counts[code]:>4}")


if __name__ == "__main__":
    main()
