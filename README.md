# airflow-log-rca-ml

A **local mimic** of the Airflow-on-GCP log-analysis pipeline used at large banks to do **root-cause analysis (RCA)** on crashing DAGs.

Real setup (the one this mimics):

- Airflow runs on **Google Cloud Composer**.
- Hundreds of DAGs run every day; a meaningful chunk fail.
- Task logs are streamed to a **GCS bucket** (volume is huge — terabytes/month).
- A separate ML pipeline reads those logs, mines templates, scores anomalies, classifies failure modes, and uses an **LLM to summarise root cause + suggest a fix**.
- On-call engineers read a daily RCA digest instead of grepping GCS by hand.

This repo replicates that **shape** end-to-end on your laptop. No GCP credentials, no real data, no cloud cost.

---

## Two folders, same pipeline

```
airflow-log-rca-ml/
├── data/                   # 3 datasets, each with its own loader (see data/README.md)
│   ├── 1_synthetic_airflow/   # regenerable; lives in fake_gcs_bucket/ at runtime
│   ├── 2_loghub_hadoop/       # real Hadoop MapReduce logs (LogHub 2k sample)
│   └── 3_loghub_hdfs/         # real HDFS block-level logs (LogHub 2k sample)
├── fake_gcs_bucket/        # synthetic Airflow logs (gitignored; regenerable)
├── mini-project/           # learning-grade: one script per phase, heavy comments
│   └── datasets/           # 3 per-dataset loaders (1_synthetic.py, 2_*, 3_*)
└── production-shaped/      # engineered: package + Airflow DAGs + Docker + tests
    └── src/log_rca/datasets/   # 2 per-dataset loaders (synthetic + hadoop)
```

Use the mini-project to *understand* a phase; use the production-shaped side to see *what the same idea looks like* when split into modules with config, tests, and orchestration.

**Datasets are deliberately not unified into a common schema.** Each source has different columns (Airflow has `dag_id`/`task_id`; Hadoop has `Process`/`Component`; HDFS has `Pid`/`Component`), and we run the same RCA pipeline against one dataset at a time so we can compare results side by side. See `data/README.md` for sources, columns, licences, and citations.

---

## ML phases (delivered one at a time)

| Phase | What it does | Mini file | Prod module |
|------:|---|---|---|
| 0 | Generate synthetic Airflow logs in `fake_gcs_bucket/` | `generate_logs.py` | `src/log_rca/datagen/` |
| 1 | Mine log templates (Drain3), find templates that distinguish failed runs | `phase1_clustering.py` | `src/log_rca/ml/clustering.py` |
| 2 | Detect anomalous runs (IsolationForest on template histograms) | `phase2_anomaly.py` | `src/log_rca/ml/anomaly.py` |
| 3 | Classify failure mode (RandomForest, supervised) | `phase3_classify.py` | `src/log_rca/ml/classification.py` |
| 4 | LLM summarises root cause + suggested fix (Claude API) | `phase4_llm_rca.py` | `src/log_rca/ml/llm_rca.py` |

Current status: **Phases 0 + 1 + 2 shipped.** Phase 1 runs across all 3 datasets; Phase 2 (IsolationForest anomaly detection) runs on the synthetic dataset and hits **precision@25 = 76%** against ground-truth failure labels. Phases 3–4 land in later commits.

---

## Quickstart

```bash
# Generate synthetic logs (~2000+ files in fake_gcs_bucket/)
cd mini-project
pip install -r requirements.txt
python generate_logs.py
```

See `mini-project/README.md` and `production-shaped/README.md` for the per-folder details.

---

## What's intentionally not here

- No real GCP / Cloud Composer / GCS.
- No real bank data (everything in `fake_gcs_bucket/` is synthetic and gitignored).
- No Kubernetes / Helm / Terraform.
- Phases 2–4 not yet implemented — see `plans/` for the full design.

---

## Project-local skills

`.claude/skills/` contains curated reference skills relevant to this project (Claude API patterns, Python testing, TDD, prompt optimisation, etc.). They're pulled from the user's local skill warehouse so anyone cloning this repo can see what reference material was used.
