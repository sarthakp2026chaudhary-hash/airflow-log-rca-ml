"""Airflow DAG that runs the synthetic log generator daily.

Drop this file into an Airflow ``dags/`` folder (or Composer's GCS-backed
DAG bucket) and it will appear in the UI. In a real bank setup this DAG
would be replaced by the actual production DAGs whose logs we are
analysing — the *generator* is only here because we are mimicking the
ecosystem locally.

Imports are guarded so the file can be imported (and lightly parsed by
Airflow's DAG processor) without ``apache-airflow`` installed in dev
environments.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover -- airflow is an optional extra
    DAG = None  # type: ignore[assignment]
    PythonOperator = None  # type: ignore[assignment]


def _run_generator() -> None:
    """Wrapper so the import is lazy at task-execution time."""
    from log_rca.datagen.cli import main as run

    rc = run([])
    if rc != 0:
        raise RuntimeError(f"log-rca-gen exited with {rc}")


if DAG is not None:
    with DAG(
        dag_id="generate_synthetic_logs",
        description="Phase 0 of airflow-log-rca-ml — populates fake_gcs_bucket/",
        start_date=datetime(2026, 1, 1),
        schedule="@daily",
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "rca-platform",
            "retries": 1,
            "retry_delay": timedelta(minutes=5),
        },
        tags=["rca", "phase-0", "synthetic"],
    ) as dag:
        PythonOperator(
            task_id="generate_logs",
            python_callable=_run_generator,
        )
