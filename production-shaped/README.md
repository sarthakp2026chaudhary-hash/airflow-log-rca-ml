# production-shaped

Engineered version of the airflow-log-rca-ml pipeline. Same logical phases as `../mini-project/`, but structured the way you would actually ship this inside a bank:

- proper **package** under `src/log_rca/`
- **abstract storage interface** so the local-FS backend can be swapped for a real GCS backend with a one-class change
- **config** via `pydantic-settings` + `config/settings.yaml` (env vars override)
- **unit tests** under `tests/` with `pytest` + coverage
- **Airflow DAG** in `airflow_dags/` that orchestrates Phase 0 → Phase 4 — drop into Composer / a Composer-equivalent and it would run
- **Docker** scaffolding so the whole thing can run in a container

## Layout

```
production-shaped/
├── pyproject.toml              # package + dep groups (datagen / ml / llm / dev)
├── Dockerfile
├── docker-compose.yml          # placeholder for full airflow+postgres stack
├── config/
│   └── settings.yaml
├── src/log_rca/
│   ├── config.py               # pydantic-settings entrypoint
│   ├── storage/                # LogStorage abstract + LocalFSBackend
│   ├── datagen/                # Phase 0: synthetic log generator
│   ├── ingest/                 # Phase 1: log parser + normalizer  (later)
│   ├── ml/                     # Phases 1-4 model code             (later)
│   ├── reports/                # Markdown report writers           (later)
│   └── pipeline.py             # orchestrates phases               (later)
├── airflow_dags/
│   ├── generate_logs_dag.py    # runs Phase 0 daily
│   └── rca_pipeline_dag.py     # runs Phase 1-4 daily              (later)
├── tests/
│   ├── conftest.py
│   └── unit/
│       ├── test_log_format.py
│       ├── test_failure_modes.py
│       └── test_storage_local.py
└── scripts/
    └── run_local.sh            # convenience entrypoint
```

## Setup

```bash
cd production-shaped
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

# Install with the datagen extras for Phase 0
pip install -e ".[datagen,dev]"
```

## Run Phase 0 (generate synthetic logs)

```bash
# Console-script entrypoint installed by pyproject.toml
log-rca-gen

# Or as a module
python -m log_rca.datagen.cli
```

Output lands in `../fake_gcs_bucket/` (the same shared corpus the mini-project uses).

## Run tests

```bash
pytest -v
pytest --cov=src/log_rca --cov-report=term-missing
```

## Status

| Phase | Status |
|------:|--------|
| 0     | shipped |
| 1     | coming |
| 2     | coming |
| 3     | coming |
| 4     | coming |

## Why this layout?

| Concern | Choice | Rationale |
|---|---|---|
| Package mgmt | `pyproject.toml` (PEP 621) | Modern Python standard; supports optional dep groups per phase |
| Storage abstraction | `LogStorage` protocol + `LocalFSBackend` | A real bank would swap `LocalFSBackend` for `GCSBackend` and nothing else changes |
| Config | `pydantic-settings` + YAML | Type-safe; env vars override file values; testable |
| Orchestration | Airflow DAGs in `airflow_dags/` | Matches the real Composer setup; Phase 0 itself runs as an Airflow task in this version |
| Tests | `pytest` + `pytest-cov` | Target ≥80% on `src/log_rca/` per the rules |
| Container | `Dockerfile` + `docker-compose.yml` | Phase 0 runs in plain Python; the Airflow stack lights up in later phases |
