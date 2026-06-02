"""Airflow DAG that runs the daily RCA pipeline.

Today: wires Phase 1 (template mining + RCA report) as a single task.
Later: Phase 2 (anomaly detection), Phase 3 (failure classification),
and Phase 4 (LLM RCA summarisation) will land as downstream tasks.

Imports are guarded so the file is importable in dev environments where
``apache-airflow`` is not installed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover -- airflow is an optional extra
    DAG = None  # type: ignore[assignment]
    PythonOperator = None  # type: ignore[assignment]


def _run_phase1() -> None:
    """Run Phase 1 against the synthetic dataset."""
    from log_rca.config import load_settings
    from log_rca.pipeline.phase1 import run

    settings = load_settings()
    bucket = settings.storage.bucket_root
    report = Path("reports") / "phase1_clusters.md"
    stats = run(bucket_root=bucket, report_path=report)
    if stats["failed_runs"] == 0:
        print("WARNING: no FAILED runs found — Phase 1 RCA is degenerate.")


def _run_phase2() -> None:
    """Run Phase 2 (anomaly detection) against the synthetic dataset."""
    from log_rca.config import load_settings
    from log_rca.pipeline.phase2 import run

    settings = load_settings()
    bucket = settings.storage.bucket_root
    report = Path("reports") / "phase2_anomalies.md"
    stats = run(bucket_root=bucket, report_path=report, top_k=25)
    if stats["precision_at_k"] < 0.3:
        print(f"WARNING: Phase 2 precision@25 is {stats['precision_at_k']:.0%}; "
              "model may need retuning.")


def _run_phase3() -> None:
    """Run Phase 3 (RandomForest failure-mode classifier)."""
    from log_rca.config import load_settings
    from log_rca.pipeline.phase3 import run

    settings = load_settings()
    bucket = settings.storage.bucket_root
    report = Path("reports") / "phase3_classifier.md"
    model = Path("reports") / "phase3_model.pkl"
    stats = run(
        bucket_root=bucket, report_path=report,
        model_save_path=model, n_splits=5,
    )
    if stats["accuracy"] < 0.6:
        print(f"WARNING: Phase 3 accuracy is {stats['accuracy']:.0%}; "
              "failure-mode templates may be overlapping.")


if DAG is not None:
    with DAG(
        dag_id="rca_pipeline",
        description="Daily RCA pipeline (Phases 1-3 today; phase 4 to come)",
        start_date=datetime(2026, 1, 1),
        schedule="@daily",
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "rca-platform",
            "retries": 1,
            "retry_delay": timedelta(minutes=5),
        },
        tags=["rca", "phase-1", "phase-2", "phase-3", "synthetic"],
    ) as dag:
        phase1 = PythonOperator(
            task_id="phase1_template_clustering",
            python_callable=_run_phase1,
        )
        phase2 = PythonOperator(
            task_id="phase2_anomaly_detection",
            python_callable=_run_phase2,
        )
        phase3 = PythonOperator(
            task_id="phase3_classify_failures",
            python_callable=_run_phase3,
        )
        phase1 >> phase2 >> phase3
        # Future:
        #   phase4 = PythonOperator(task_id="phase4_llm_rca_summaries", ...)
        #   phase3 >> phase4
