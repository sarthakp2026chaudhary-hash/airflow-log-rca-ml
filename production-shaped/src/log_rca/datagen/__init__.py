"""Phase 0 — synthetic Airflow log generator.

Same semantics as ``mini-project/generate_logs.py`` but split into modules:

- ``dag_catalogue``: the 20 bank-flavoured DAGs and their tasks
- ``failure_modes``: the failure-mode taxonomy (OOM, BQ_QUOTA, ...)
- ``log_format``: Airflow-style log-line builders
- ``simulator``: DagRun-level orchestration
- ``cli``: ``log-rca-gen`` console-script entrypoint
"""
