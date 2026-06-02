# mini-project

Learning-grade implementation of the airflow-log-rca-ml pipeline. **One Python file per phase**, heavy comments, no package structure — meant to be read top-to-bottom.

Compare with `../production-shaped/` for the engineered version of the same pipeline.

## Setup

```bash
cd mini-project
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

## Run

```bash
# Phase 0 — generate synthetic Airflow logs into ../fake_gcs_bucket/
python generate_logs.py

# Phase 1 — mine templates and produce first RCA report  (coming next commit)
python phase1_clustering.py

# Phase 2 — anomaly detection (sketch)
python phase2_anomaly.py

# Phase 3 — failure-mode classifier (sketch)
python phase3_classify.py

# Phase 4 — LLM RCA summarisation (sketch, needs ANTHROPIC_API_KEY)
python phase4_llm_rca.py
```

## What lands where

- Logs → `../fake_gcs_bucket/airflow-logs/dag_id=.../run_id=.../task_id=.../attempt=N.log`
- Ground-truth labels → `../fake_gcs_bucket/_truth.jsonl`
- Reports → `mini-project/reports/`

## Status

| Phase | Status |
|------:|--------|
| 0     | shipped |
| 1     | coming next |
| 2     | coming |
| 3     | coming |
| 4     | coming |
