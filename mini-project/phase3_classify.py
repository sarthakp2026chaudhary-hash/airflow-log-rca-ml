"""
phase3_classify.py — Phase 3 of airflow-log-rca-ml (mini-project)
====================================================================

For each FAILED DAG run, predict its failure mode (one of 8 codes:
OOM, BQ_QUOTA, CONN_REFUSED, GCS_PERMISSION, KEY_ERROR, SCHEMA_MISMATCH,
TASK_TIMEOUT, UPSTREAM_MISSING) from its Drain3 template histogram.

Uses stratified k-fold cross-validation to get an honest per-class
precision / recall estimate. Saves a confusion matrix and a fitted
RandomForest model to ``mini-project/reports/``.

Run:
    python mini-project/phase3_classify.py

Output:
    mini-project/reports/phase3_classifier.md
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from rich.console import Console
    from rich.table import Table
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
except ImportError as e:
    print(f"ERROR: missing dependency ({e}). Run:  pip install -r mini-project/requirements.txt",
          file=sys.stderr)
    sys.exit(1)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REPORTS_DIR = HERE / "reports"
REPORT_PATH = REPORTS_DIR / "phase3_classifier.md"

N_SPLITS = 5
RANDOM_STATE = 42


def _import_loader(rel: str):
    path = HERE / "datasets" / f"{rel}.py"
    spec = importlib.util.spec_from_file_location(rel.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loader = _import_loader("1_synthetic")


def build_features_for_failed(truth: dict[str, dict]) -> tuple[
    list[tuple[str, str]],
    np.ndarray,
    list[str],
    int,
]:
    """Featurise FAILED runs only. Returns (run_keys, X, y_labels, n_templates)."""
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)

    failed_run_ids = {rid for rid, t in truth.items() if t["outcome"] == "FAILED"}
    hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)

    for rec in loader.load_records():
        if rec["run_id"] not in failed_run_ids:
            continue
        msg = rec["message"].strip()
        if not msg:
            continue
        result = miner.add_log_message(msg)
        cid = int(result["cluster_id"])
        hist[(rec["dag_id"], rec["run_id"])][cid] += 1

    templates = {int(c.cluster_id): c.get_template() for c in miner.drain.clusters}
    all_cids = sorted(templates)
    cid_index = {cid: i for i, cid in enumerate(all_cids)}

    keys = sorted(hist.keys())
    X = np.zeros((len(keys), len(all_cids)), dtype=np.float32)
    for i, key in enumerate(keys):
        for cid, cnt in hist[key].items():
            X[i, cid_index[cid]] = cnt
    y = [truth[run_id]["failure_mode"] for _, run_id in keys]
    return keys, X, y, len(templates)


def main() -> None:
    truth = loader.load_truth()
    if not truth:
        print("ERROR: no ground truth — run mini-project/generate_logs.py first.",
              file=sys.stderr)
        sys.exit(1)

    print("Phase 3 -- RandomForest failure-mode classifier")
    print("===============================================")
    keys, X, y, n_templates = build_features_for_failed(truth)
    print(f"  FAILED runs: {len(keys)}")
    print(f"  Templates as features: {n_templates}")

    y_arr = np.array(y)
    label_counts = Counter(y_arr)
    print("  Label distribution:")
    for code, n in sorted(label_counts.items()):
        print(f"    {code:<18} {n:>4}")

    # Stratified K-fold: collect out-of-fold predictions for honest metrics
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.empty_like(y_arr)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_arr), start=1):
        clf = RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X[train_idx], y_arr[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])
        print(f"  fold {fold}: trained on {len(train_idx)}, tested on {len(test_idx)}")

    # Per-class metrics
    labels = sorted(label_counts.keys())
    report = classification_report(
        y_arr, y_pred, labels=labels, digits=3, zero_division=0,
    )
    cm = confusion_matrix(y_arr, y_pred, labels=labels)
    accuracy = float((y_arr == y_pred).sum() / len(y_arr))

    # CLI summary
    console = Console()
    t = Table(title=f"Phase 3 — RandomForest, {N_SPLITS}-fold CV accuracy: "
                    f"{accuracy:.0%}")
    t.add_column("Failure mode", style="cyan")
    t.add_column("Support", justify="right")
    t.add_column("Correct", justify="right")
    t.add_column("Accuracy", justify="right")
    for label in labels:
        mask = y_arr == label
        support = int(mask.sum())
        correct = int(((y_arr == y_pred) & mask).sum())
        t.add_row(
            label, str(support), str(correct),
            f"{correct / support:.0%}" if support else "—",
        )
    console.print(t)

    # Markdown report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 3 — Failure-mode classifier (synthetic Airflow logs)")
    lines.append("")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "For each FAILED run, the template histogram (over Drain3-mined clusters) "
        "is fed to a `RandomForestClassifier` (200 trees, class-weight balanced). "
        f"We use {N_SPLITS}-fold stratified cross-validation to get an honest "
        "estimate of per-class precision/recall — no held-out leakage."
    )
    lines.append("")

    lines.append("## Overall accuracy")
    lines.append("")
    lines.append(f"**{accuracy:.1%}** ({(y_arr == y_pred).sum()}/{len(y_arr)} "
                 f"out-of-fold predictions correct)")
    lines.append("")
    lines.append("- FAILED runs: " + str(len(keys)))
    lines.append("- Features (templates): " + str(n_templates))
    lines.append("- Classes (failure modes): " + str(len(labels)))
    lines.append("")

    lines.append("## Per-class metrics (classification_report)")
    lines.append("")
    lines.append("```")
    lines.append(report)
    lines.append("```")
    lines.append("")

    lines.append("## Confusion matrix (rows = truth, cols = prediction)")
    lines.append("")
    header = "| | " + " | ".join(f"**pred {l}**" for l in labels) + " |"
    sep = "|---|" + "|".join(["---:" for _ in labels]) + "|"
    lines.append(header)
    lines.append(sep)
    for i, true_label in enumerate(labels):
        row_cells = " | ".join(str(cm[i][j]) for j in range(len(labels)))
        lines.append(f"| **truth {true_label}** | {row_cells} |")
    lines.append("")

    lines.append("## What to expect")
    lines.append("")
    lines.append(
        "Failure modes whose Drain3 templates are very distinctive (e.g. `OOM` "
        "with `MemoryError` + exit-137, or `BQ_QUOTA` with `quotaExceeded`) "
        "should score 90%+. Modes that share generic tracebacks (`KEY_ERROR` "
        "and `SCHEMA_MISMATCH` both end in Python exceptions) may confuse the "
        "model on small support. Numbers below ~80% on a class usually mean: "
        "(a) that class has only a handful of runs in the corpus, or "
        "(b) the failure-mode templates in `generate_logs.py` overlap."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
