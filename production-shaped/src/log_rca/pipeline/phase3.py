"""Phase 3 pipeline — RandomForest failure-mode classifier (synthetic).

Wiring:
    SyntheticAirflowLoader -> TemplateMiner (FAILED runs only)
    -> per-run feature vectors
    -> FailureClassifier.cross_val_predict (stratified k-fold)
    -> write_phase3_report (classification_report + confusion matrix)

CLI entrypoint: ``log-rca-phase3``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from log_rca.config import load_settings
from log_rca.datasets import SyntheticAirflowLoader
from log_rca.ml import FailureClassifier, TemplateMiner
from log_rca.reports import write_phase3_report


def _build_features_failed_only(
    loader: SyntheticAirflowLoader,
    failed_run_ids: set[str],
) -> tuple[list[tuple[str, str]], np.ndarray, int]:
    """Featurise FAILED runs only. Returns (keys, X, n_templates)."""
    miner = TemplateMiner()
    hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for rec in loader.load_records():
        if rec.run_id not in failed_run_ids:
            continue
        msg = rec.message.strip()
        if not msg:
            continue
        mined = miner.add(msg)
        hist[(rec.dag_id, rec.run_id)][mined.cluster_id] += 1

    templates = miner.templates()
    all_cids = sorted(templates)
    cid_index = {cid: i for i, cid in enumerate(all_cids)}
    keys = sorted(hist.keys())
    X = np.zeros((len(keys), len(all_cids)), dtype=np.float32)
    for i, key in enumerate(keys):
        for cid, cnt in hist[key].items():
            X[i, cid_index[cid]] = cnt
    return keys, X, len(templates)


def run(
    *,
    bucket_root: Path,
    report_path: Path,
    n_splits: int = 5,
    model_save_path: Path | None = None,
) -> dict:
    loader = SyntheticAirflowLoader(bucket_root)
    truth = loader.load_truth()
    if not truth:
        raise FileNotFoundError(
            f"No ground truth at {loader.truth_path}. Run `log-rca-gen` first."
        )
    failed_truth = {rid: t for rid, t in truth.items() if t.outcome == "FAILED"}
    if not failed_truth:
        raise ValueError("No FAILED runs in ground truth — nothing to classify.")

    keys, X, n_templates = _build_features_failed_only(loader, set(failed_truth))
    y = np.array([failed_truth[run_id].failure_mode for _, run_id in keys])

    # Check stratification feasibility
    label_counts = Counter(y)
    min_label = min(label_counts.values())
    actual_splits = min(n_splits, min_label)
    if actual_splits < n_splits:
        print(f"WARNING: smallest class has {min_label} samples; "
              f"using {actual_splits}-fold instead of {n_splits}-fold")

    classifier = FailureClassifier()
    y_pred = classifier.cross_val_predict(X, y, n_splits=actual_splits)

    # Also fit on full data for the saved-model artifact
    classifier.fit(X, y)
    if model_save_path is not None:
        classifier.save(model_save_path)

    labels = sorted(label_counts.keys())
    stats = write_phase3_report(
        output_path=report_path,
        y_true=y, y_pred=y_pred,
        labels=labels,
        n_features=n_templates,
        n_splits=actual_splits,
    )
    stats["report_path"] = str(report_path)
    stats["label_counts"] = dict(label_counts)
    return stats


# ─── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 — RandomForest failure-mode classifier (synthetic)"
    )
    p.add_argument("--bucket-root", type=Path, help="override storage.bucket_root")
    p.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase3_classifier.md"),
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--save-model", type=Path, default=None,
                   help="persist the trained RandomForest (.pkl)")
    p.add_argument("--config", type=Path, help="path to settings.yaml")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(args.config)
    bucket = args.bucket_root or settings.storage.bucket_root

    print(f"Phase 3 -- bucket={bucket} -> report={args.report}")
    try:
        stats = run(
            bucket_root=bucket,
            report_path=args.report,
            n_splits=args.n_splits,
            model_save_path=args.save_model,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    print(
        f"  failed_runs={stats['n_samples']} | classes={stats['n_classes']} | "
        f"features={stats['n_features']} | "
        f"accuracy={stats['accuracy']:.1%}"
    )
    print(f"  report: {stats['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
