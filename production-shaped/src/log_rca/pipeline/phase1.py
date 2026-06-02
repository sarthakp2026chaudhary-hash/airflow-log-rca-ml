"""Phase 1 pipeline — synthetic Airflow dataset.

Wiring:
    SyntheticAirflowLoader → TemplateMiner.fit
    → per-(dag, run) histograms
    → Discriminator.per_dag + globally + per_run_rca
    → write_phase1_report

CLI entrypoint: ``log-rca-phase1`` (declared in pyproject.toml).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from log_rca.config import load_settings
from log_rca.datasets import SyntheticAirflowLoader
from log_rca.ml import Discriminator, TemplateMiner
from log_rca.reports import write_phase1_report


def run(
    *,
    bucket_root: Path,
    report_path: Path,
    drain3_state_path: Path | None = None,
) -> dict:
    """Execute Phase 1 against the synthetic Airflow dataset.

    Returns a dict of summary stats (also printed).
    """
    loader = SyntheticAirflowLoader(bucket_root)
    truth = loader.load_truth()
    if not truth:
        raise FileNotFoundError(
            f"No ground truth at {loader.truth_path}. "
            "Run `log-rca-gen` first to generate the synthetic corpus."
        )

    miner = TemplateMiner()
    hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    n_lines = 0
    for rec in loader.load_records():
        msg = rec.message.strip()
        if not msg:
            continue
        mined = miner.add(msg)
        hist[(rec.dag_id, rec.run_id)][mined.cluster_id] += 1
        n_lines += 1

    templates = miner.templates()
    outcomes = {rid: tr.outcome for rid, tr in truth.items()}
    failure_modes = {rid: tr.failure_mode for rid, tr in truth.items()}

    discr = Discriminator()
    per_dag = discr.per_dag(hist, outcomes, templates)
    globally = discr.globally(hist, outcomes, templates)
    per_run = discr.per_run_rca(hist, outcomes, failure_modes, globally)

    write_phase1_report(
        output_path=report_path,
        dataset_label="1 — Synthetic Airflow logs",
        outcomes=outcomes,
        templates=templates,
        per_dag_discrim=per_dag,
        global_discrim=globally,
        per_run=per_run,
    )

    if drain3_state_path is not None:
        miner.save(drain3_state_path)

    return {
        "lines_processed": n_lines,
        "templates_discovered": miner.cluster_count(),
        "runs": len(truth),
        "dags_with_both_outcomes": len(per_dag),
        "failed_runs": sum(1 for o in outcomes.values() if o == "FAILED"),
        "report_path": str(report_path),
    }


# ─── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 — Drain3 template mining + RCA report (synthetic dataset)"
    )
    p.add_argument("--bucket-root", type=Path, help="override storage.bucket_root")
    p.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase1_clusters.md"),
        help="output path for the Markdown RCA report",
    )
    p.add_argument(
        "--save-state",
        type=Path,
        default=None,
        help="optional path to persist Drain3 state (.pkl)",
    )
    p.add_argument("--config", type=Path, help="path to settings.yaml")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(args.config)
    bucket = args.bucket_root or settings.storage.bucket_root

    print(f"Phase 1 -- bucket={bucket} -> report={args.report}")
    try:
        stats = run(
            bucket_root=bucket,
            report_path=args.report,
            drain3_state_path=args.save_state,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(
        f"  processed {stats['lines_processed']:,} log lines | "
        f"{stats['templates_discovered']} templates | "
        f"{stats['runs']} runs ({stats['failed_runs']} FAILED) | "
        f"{stats['dags_with_both_outcomes']} DAGs ranked"
    )
    print(f"  report: {stats['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
