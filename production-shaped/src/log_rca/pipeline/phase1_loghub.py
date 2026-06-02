"""Phase 1 pipeline — LogHub Hadoop dataset.

Reuses ``TemplateMiner`` and the LogHub-flavoured report writer to
produce a per-template breakdown of dataset 2 (Hadoop_2k). No labels,
so no Fisher discrimination.

CLI entrypoint: ``log-rca-phase1-hadoop``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from log_rca.datasets import LogHubHadoopLoader
from log_rca.ml import TemplateMiner
from log_rca.reports import write_phase1_loghub_report


def run_hadoop(
    *,
    data_dir: Path,
    report_path: Path,
    drain3_state_path: Path | None = None,
) -> dict:
    """Execute Phase 1 against the LogHub Hadoop_2k dataset."""
    loader = LogHubHadoopLoader(data_dir)
    miner = TemplateMiner()

    counts_by_cid: Counter[int] = Counter()
    counts_by_level: dict[str, Counter[int]] = defaultdict(Counter)
    n_records = 0
    for rec in loader.load_records():
        msg = rec.content.strip()
        if not msg:
            continue
        mined = miner.add(msg)
        counts_by_cid[mined.cluster_id] += 1
        counts_by_level[rec.level][mined.cluster_id] += 1
        n_records += 1

    loghub_templates = loader.load_templates()

    write_phase1_loghub_report(
        output_path=report_path,
        dataset_label="2 — LogHub Hadoop_2k",
        record_count=n_records,
        our_templates=miner.templates(),
        loghub_template_count=len(loghub_templates),
        counts_by_cid=counts_by_cid,
        counts_by_level=counts_by_level,
    )

    if drain3_state_path is not None:
        miner.save(drain3_state_path)

    return {
        "records_processed": n_records,
        "templates_drain3": miner.cluster_count(),
        "templates_loghub": len(loghub_templates),
        "levels": {lvl: sum(c.values()) for lvl, c in counts_by_level.items()},
        "report_path": str(report_path),
    }


# ─── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 — Drain3 template mining (LogHub Hadoop_2k)"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/2_loghub_hadoop"),
        help="path to data/2_loghub_hadoop (default: %(default)s)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase1_hadoop.md"),
        help="output path for the Markdown report",
    )
    p.add_argument(
        "--save-state",
        type=Path,
        default=None,
        help="optional path to persist Drain3 state (.pkl)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"Phase 1 (Hadoop) -- data={args.data_dir} -> report={args.report}")
    try:
        stats = run_hadoop(
            data_dir=args.data_dir,
            report_path=args.report,
            drain3_state_path=args.save_state,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(
        f"  records={stats['records_processed']:,} | "
        f"our_templates={stats['templates_drain3']} | "
        f"loghub_templates={stats['templates_loghub']} | "
        f"levels={stats['levels']}"
    )
    print(f"  report: {stats['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
