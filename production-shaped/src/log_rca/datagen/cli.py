"""``log-rca-gen`` console-script entrypoint.

Reads config (YAML + env overrides), constructs a ``LocalFSBackend``, and
generates ~N synthetic DAG runs into the bucket.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from faker import Faker

from log_rca.config import load_settings
from log_rca.datagen.dag_catalogue import DAGS
from log_rca.datagen.simulator import simulate_run
from log_rca.datagen.writer import write_run, write_truth
from log_rca.storage import LocalFSBackend


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic Airflow logs")
    p.add_argument(
        "--bucket-root",
        type=Path,
        help="Override storage.bucket_root from config",
    )
    p.add_argument(
        "--total-runs",
        type=int,
        help="Override datagen.total_runs from config",
    )
    p.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML config file (default: ../config/settings.yaml)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(args.config)

    bucket_root = args.bucket_root or settings.storage.bucket_root
    total_runs = args.total_runs or settings.datagen.total_runs

    storage = LocalFSBackend(bucket_root)
    rng = random.Random(settings.datagen.seed)
    fake = Faker()
    Faker.seed(settings.datagen.seed)

    # Reset truth file before this run.
    truth_key = settings.storage.truth_file
    if storage.exists(truth_key):
        storage.write_text(truth_key, "")

    print(f"Generating ~{total_runs} DAG runs into {storage.root}")
    dag_names = list(DAGS.keys())
    now = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    earliest = now - timedelta(days=settings.datagen.days_back)

    total_files = 0
    fail_counts: dict[str, int] = {}
    success_count = 0

    for i in range(total_runs):
        dag_id = rng.choice(dag_names)
        tasks = DAGS[dag_id]
        offset = rng.random() * settings.datagen.days_back * 86400
        run_start = earliest + timedelta(seconds=offset)
        run_start = run_start.replace(microsecond=rng.randint(0, 999_999))

        dr = simulate_run(
            dag_id=dag_id,
            tasks=tasks,
            run_start=run_start,
            failure_rate=settings.datagen.failure_rate,
            rng=rng,
        )
        total_files += write_run(
            dr=dr,
            storage=storage,
            logs_prefix=settings.storage.logs_prefix,
            rng=rng,
            fake=fake,
        )
        write_truth(dr=dr, storage=storage, truth_key=truth_key)

        if dr.overall_outcome == "FAILED":
            fail_counts[dr.failure_mode] = fail_counts.get(dr.failure_mode, 0) + 1
        else:
            success_count += 1

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{total_runs} runs ({total_files} log files so far)")

    print(f"\nDone. {total_files} log files written under "
          f"{storage.root / settings.storage.logs_prefix}")
    print(f"Ground truth: {storage.root / truth_key}")
    print(f"\nOutcome breakdown: SUCCESS={success_count}, "
          f"FAILED={sum(fail_counts.values())}")
    print("Failure-mode breakdown:")
    for code in sorted(fail_counts):
        print(f"  {code:<18} {fail_counts[code]:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
