# Datasets

Three datasets live here. Each one is processed by a **separate** loader script (no schema-unification adapter) — different sources have different columns, and we want to compare the same RCA pipeline across them side by side rather than smear them into one synthetic shape.

| # | Dataset | Source | Records | Anomaly labels | Columns / format |
|---|---|---|---:|---|---|
| 1 | Synthetic Airflow logs | this repo (`mini-project/generate_logs.py`) | 500 runs / ~1,835 files | ✓ (`_truth.jsonl`) | Airflow task-log format; `dag_id`, `run_id`, `task_id`, `attempt`, `failure_mode` |
| 2 | LogHub Hadoop_2k | [logpai/loghub Hadoop](https://github.com/logpai/loghub/tree/master/Hadoop) | 2,000 lines / 114 templates | ✗ (full dataset has them on [Zenodo](https://doi.org/10.5281/zenodo.8196385)) | `LineId, Date, Time, Level, Process, Component, Content, EventId, EventTemplate` |
| 3 | LogHub HDFS_2k | [logpai/loghub HDFS](https://github.com/logpai/loghub/tree/master/HDFS) | 2,000 lines / 14 templates | ✗ (full HDFS_v1 has them on Zenodo) | `LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate` |

## How we use them

Each dataset has its own loader; **we do not mix records across datasets**. The same Phase-1+ ML pipeline runs against one dataset at a time so we can compare results.

| Folder | Script | Loads |
|---|---|---|
| `mini-project/datasets/` | `1_synthetic.py` | dataset 1 |
| `mini-project/datasets/` | `2_loghub_hadoop.py` | dataset 2 |
| `mini-project/datasets/` | `3_loghub_hdfs.py` | dataset 3 |
| `production-shaped/src/log_rca/datasets/` | `synthetic_airflow.py` | dataset 1 |
| `production-shaped/src/log_rca/datasets/` | `loghub_hadoop.py` | dataset 2 |

The user's plan: 3 datasets exercised in the mini-project, 2 of them re-exercised in the production-shaped side (the ones we care most about engineering rigour for). That gives 5 distinct loader scripts to compare.

## Dataset details

### 1. Synthetic Airflow logs (`data/1_synthetic_airflow/`)

**Not committed.** Regenerable in seconds. Run from the repo root:

```bash
python mini-project/generate_logs.py
```

Output lands in `fake_gcs_bucket/airflow-logs/dag_id=…/run_id=…/task_id=…/attempt=N.log`, with per-run ground-truth labels in `fake_gcs_bucket/_truth.jsonl`. See `data/1_synthetic_airflow/README.md` for the schema and `mini-project/generate_logs.py` for the failure taxonomy.

**Why we keep this even though we have real-world LogHub:** it is the **only** dataset with a clean RCA ground-truth (each failed run is labelled with one of 8 failure modes — `OOM`, `BQ_QUOTA`, `SCHEMA_MISMATCH`, etc.). LogHub's labels are at the job/block level, not the bank-flavoured-RCA level. So our Phase 3 classifier needs synthetic data to learn from.

### 2. LogHub Hadoop_2k (`data/2_loghub_hadoop/`)

Real-world Hadoop MapReduce job logs. 2,000-line sampled subset from running `WordCount` and `PageRank` under both normal and fault-injected conditions (machine-down, network-disconnect, disk-full).

Files:
- `Hadoop_2k.log` — raw log lines
- `Hadoop_2k.log_structured.csv` — pre-parsed by LogHub; columns: `LineId, Date, Time, Level, Process, Component, Content, EventId, EventTemplate`
- `Hadoop_2k.log_templates.csv` — 114 distinct templates (E1–E114) with their wildcard patterns
- `README.md` — LogHub's own readme (kept for attribution / dataset background)

Anomaly labels are **not** in the 2k subset — they live in the full Hadoop dataset on Zenodo (`abnormal_label.txt`, ~hundreds of MB). For our purposes the 2k subset is enough to demonstrate template clustering and anomaly scoring.

### 3. LogHub HDFS_2k (`data/3_loghub_hdfs/`)

Real-world Hadoop Distributed File System block-level logs. The most-studied dataset in log-mining academia (the original [Drain](https://arxiv.org/abs/2202.04301), [DeepLog](https://dl.acm.org/doi/10.1145/3133956.3134015), and many follow-ups benchmark on HDFS_v1).

Files:
- `HDFS_2k.log` — raw log lines
- `HDFS_2k.log_structured.csv` — pre-parsed; columns: `LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate`
- `HDFS_2k.log_templates.csv` — 14 distinct templates (E1–E14)
- `README.md` — LogHub's own readme

Labels for full HDFS_v1 are on Zenodo (`anomaly_label.csv`, mapping `BlockId → Normal/Anomaly`). Not in the 2k subset.

## Citations

If you use the LogHub data downstream of this project, cite:

- Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. **Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.** *ISSRE 2023.* [arXiv:2008.06448](https://arxiv.org/abs/2008.06448)
- For HDFS specifically: Wei Xu et al. **Detecting Large-Scale System Problems by Mining Console Logs.** *SOSP 2009.*
- For Hadoop specifically: Qingwei Lin et al. **Log Clustering Based Problem Identification for Online Service Systems.** *ICSE 2016.*

## License

LogHub is distributed under the [LogHub license](https://github.com/logpai/loghub/blob/master/LICENSE) (CC-BY 4.0, attribution required). Our synthetic dataset is original and MIT-licensed along with the rest of the repo.

## Why not Kaggle?

We checked Kaggle for native Apache Airflow log datasets — **none exist publicly**. Real production Airflow logs leak internal table names, hostnames, PII, etc. and are NDA'd. What Kaggle does have (`omduggineni/loghub-apache-log-data` is HTTP server logs, not Airflow; `anjolaoluwaajayi/root-cause-analysis-dataset` is small tabular incident data, useful only as a classifier-shape sanity check) was either off-target or just a mirror of LogHub. We pull straight from `logpai/loghub` on GitHub for the canonical, well-versioned source.
