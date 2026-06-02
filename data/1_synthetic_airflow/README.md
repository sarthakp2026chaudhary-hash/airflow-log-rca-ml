# Dataset 1 — Synthetic Airflow logs

This folder is intentionally almost empty. The actual log files are **regenerated on demand** into `<repo>/fake_gcs_bucket/` (gitignored) rather than committed.

## Regenerate

From the repo root:

```bash
python mini-project/generate_logs.py
```

About 5–15 s. Produces ~1,835 log files across 500 DAG runs.

## On-disk layout (after regeneration)

```
fake_gcs_bucket/
├── _truth.jsonl                    # one JSON line per DAG run; ground-truth labels
└── airflow-logs/
    └── dag_id=<dag>/
        └── run_id=scheduled__<iso>/
            └── task_id=<task>/
                └── attempt=<n>.log
```

## `_truth.jsonl` columns

| Column | Type | Description |
|---|---|---|
| `dag_id` | str | one of 20 bank-flavoured DAGs (`etl_customer_daily`, `risk_pnl_hourly`, …) |
| `run_id` | str | `scheduled__YYYY-MM-DDTHH-MM-SS+0000` |
| `outcome` | str | `SUCCESS` or `FAILED` |
| `failure_mode` | str | empty for SUCCESS; one of 8 codes for FAILED |
| `task_count` | int | how many tasks ran before success or first failure |

## Failure-mode taxonomy

| Code | Weight | What it looks like in the log |
|---|---:|---|
| `OOM` | 0.18 | `Killed`, exit code 137, `MemoryError` |
| `TASK_TIMEOUT` | 0.14 | `airflow.exceptions.AirflowTaskTimeout` |
| `BQ_QUOTA` | 0.13 | `quotaExceeded: Exceeded rate limits` |
| `CONN_REFUSED` | 0.13 | `psycopg2.OperationalError: connection refused` |
| `SCHEMA_MISMATCH` | 0.12 | `pyarrow.lib.ArrowInvalid: Schema mismatch` |
| `GCS_PERMISSION` | 0.10 | `403 Forbidden: storage.objects.get` |
| `UPSTREAM_MISSING` | 0.10 | `airflow.exceptions.AirflowSkipException` |
| `KEY_ERROR` | 0.10 | Python `KeyError` traceback |

Generator: [`mini-project/generate_logs.py`](../../mini-project/generate_logs.py) (mini) and [`production-shaped/src/log_rca/datagen/`](../../production-shaped/src/log_rca/datagen/) (engineered).

## Why both this dataset and LogHub?

LogHub gives us real-world logs from real distributed systems — credibility. The synthetic dataset gives us **clean per-run labels with a known failure-mode taxonomy** — necessary for training and scoring the Phase 3 classifier. We use both, never mixed.
