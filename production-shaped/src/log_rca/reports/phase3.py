"""Phase 3 Markdown report — failure-mode classifier metrics."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def write_phase3_report(
    *,
    output_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    n_features: int,
    n_splits: int,
) -> dict:
    """Render per-class metrics + confusion matrix. Returns summary stats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy = float((y_true == y_pred).sum() / len(y_true))

    report_text = classification_report(
        y_true, y_pred, labels=labels, digits=3, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    lines: list[str] = []
    lines.append("# Phase 3 — Failure-mode classifier (synthetic Airflow logs)")
    lines.append("")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "RandomForest (200 trees, class-weight balanced) trained on per-run "
        "template histograms with stratified k-fold cross-validation "
        f"(n_splits={n_splits}). The held-out predictions in each fold are "
        "concatenated to compute honest per-class precision / recall — no "
        "training leakage."
    )
    lines.append("")

    lines.append("## Overall accuracy")
    lines.append("")
    lines.append(
        f"**{accuracy:.1%}** ({(y_true == y_pred).sum()}/{len(y_true)} "
        f"out-of-fold predictions correct)"
    )
    lines.append("")
    lines.append(f"- FAILED runs trained on: {len(y_true)}")
    lines.append(f"- Features (templates): {n_features}")
    lines.append(f"- Classes (failure modes): {len(labels)}")
    lines.append("")

    lines.append("## Per-class metrics")
    lines.append("")
    lines.append("```")
    lines.append(report_text)
    lines.append("```")
    lines.append("")

    lines.append("## Confusion matrix (rows = truth, cols = prediction)")
    lines.append("")
    header = "| | " + " | ".join(f"**pred {l}**" for l in labels) + " |"
    sep = "|---|" + "|".join(["---:" for _ in labels]) + "|"
    lines.append(header)
    lines.append(sep)
    for i, true_label in enumerate(labels):
        row = " | ".join(str(cm[i][j]) for j in range(len(labels)))
        lines.append(f"| **truth {true_label}** | {row} |")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "accuracy": accuracy,
        "n_samples": int(len(y_true)),
        "n_classes": len(labels),
        "n_features": n_features,
        "n_splits": n_splits,
    }
