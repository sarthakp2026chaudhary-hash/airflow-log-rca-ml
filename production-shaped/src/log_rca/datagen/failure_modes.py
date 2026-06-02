"""Failure-mode taxonomy.

Each ``FailureMode`` describes one way a real Composer DAG can crash:
the short code we use in ground-truth labels, how often it happens
relative to other failures, and the lines that appear in the log right
before the FAILED footer.

The error lines are ``str.format`` templates with these placeholders
available: ``dag_id``, ``task_id``, ``lineno``, ``bucket``, ``obj``,
``shard``, ``ip``, ``ip2``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureMode:
    code: str
    weight: float
    error_lines: list[str]


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

# Returned for non-final attempts of a failing task (mimics retry-then-fail).
RETRY_PLACEHOLDER = FailureMode("__retry", 0.0, [
    "[ERROR] - Task instance is in a temporary failed state and will be retried",
    "Exception: transient error, will retry",
])


def weights() -> list[float]:
    return [m.weight for m in FAILURE_MODES]


def by_code(code: str) -> FailureMode | None:
    for m in FAILURE_MODES:
        if m.code == code:
            return m
    return None
