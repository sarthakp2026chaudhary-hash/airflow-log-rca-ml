"""Phase 2 pipeline — IsolationForest anomaly detection (synthetic dataset).

Wiring:
    SyntheticAirflowLoader  → TemplateMiner (re-runs Drain3 from scratch)
    → per-(dag, run) feature vectors (template counts + line count + ...)
    → AnomalyDetector.fit(SUCCESS-only).score(all)
    → write_phase2_report

CLI entrypoint: ``log-rca-phase2``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from log_rca.config import load_settings
from log_rca.datasets import SyntheticAirflowLoader
from log_rca.ml import AnomalyDetector, TemplateMiner
from log_rca.reports import write_phase2_report


def _build_features(loader: SyntheticAirflowLoader) -> tuple[
    list[tuple[str, str]],   # row keys
    np.ndarray,              # feature matrix
    int,                     # n_templates discovered
    dict[tuple[str, str], list[tuple[int, int]]],   # dominant templates per run
]:
    miner = TemplateMiner()
    hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    line_count: dict[tuple[str, str], int] = defaultdict(int)
    attempts: dict[tuple[str, str], set[int]] = defaultdict(set)

    for rec in loader.load_records():
        msg = rec.message.strip()
        if not msg:
            continue
        mined = miner.add(msg)
        key = (rec.dag_id, rec.run_id)
        hist[key][mined.cluster_id] += 1
        line_count[key] += 1
        attempts[key].add(rec.attempt)

    templates = miner.templates()
    all_cids = sorted(templates)
    cid_index = {cid: i for i, cid in enumerate(all_cids)}

    keys = sorted(hist.keys())
    X = np.zeros((len(keys), len(all_cids) + 3), dtype=np.float32)
    for i, key in enumerate(keys):
        for cid, cnt in hist[key].items():
            X[i, cid_index[cid]] = cnt
        X[i, -3] = line_count[key]
        X[i, -2] = len(hist[key])
        X[i, -1] = len(attempts[key])

    dominant = {key: hist[key].most_common(3) for key in keys}
    return keys, X, len(templates), dominant


def run(
    *,
    bucket_root: Path,
    report_path: Path,
    top_k: int = 25,
    model_save_path: Path | None = None,
) -> dict:
    loader = SyntheticAirflowLoader(bucket_root)
    truth = loader.load_truth()
    if not truth:
        raise FileNotFoundError(
            f"No ground truth at {loader.truth_path}. "
            "Run `log-rca-gen` first."
        )
    keys, X, n_templates, dominant = _build_features(loader)

    outcomes = {rid: tr.outcome for rid, tr in truth.items()}
    failure_modes = {rid: tr.failure_mode for rid, tr in truth.items()}

    success_mask = np.array(
        [outcomes.get(run_id) == "SUCCESS" for _, run_id in keys]
    )
    detector = AnomalyDetector().fit(X[success_mask])
    scores = detector.score(X).tolist()

    stats = write_phase2_report(
        output_path=report_path,
        keys=keys,
        scores=scores,
        outcomes=outcomes,
        failure_modes=failure_modes,
        dominant_templates=dominant,
        top_k=top_k,
        n_templates_mined=n_templates,
    )

    if model_save_path is not None:
        detector.save(model_save_path)

    return {**stats, "report_path": str(report_path), "templates": n_templates}


# ─── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 — IsolationForest anomaly detection (synthetic dataset)"
    )
    p.add_argument("--bucket-root", type=Path, help="override storage.bucket_root")
    p.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase2_anomalies.md"),
    )
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--save-model", type=Path, default=None)
    p.add_argument("--config", type=Path, help="path to settings.yaml")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(args.config)
    bucket = args.bucket_root or settings.storage.bucket_root

    print(f"Phase 2 -- bucket={bucket} -> report={args.report}")
    try:
        stats = run(
            bucket_root=bucket,
            report_path=args.report,
            top_k=args.top_k,
            model_save_path=args.save_model,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    print(
        f"  runs={stats['n_total']} | top-{stats['top_k']} "
        f"precision={stats['precision_at_k']:.0%} | "
        f"recall={stats['recall_at_k']:.0%} | "
        f"templates={stats['templates']}"
    )
    print(f"  report: {stats['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
