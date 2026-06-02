"""Tests for the Phase 3 classifier report writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from log_rca.reports import write_phase3_report


def test_perfect_classifier_renders_100_percent(tmp_path: Path):
    out = tmp_path / "p3.md"
    y = np.array(["OOM"] * 5 + ["BQ_QUOTA"] * 5)
    stats = write_phase3_report(
        output_path=out,
        y_true=y, y_pred=y,
        labels=["BQ_QUOTA", "OOM"],
        n_features=10, n_splits=5,
    )
    body = out.read_text(encoding="utf-8")
    assert "**100.0%**" in body
    assert stats["accuracy"] == 1.0
    assert stats["n_samples"] == 10
    # confusion-matrix rows must appear
    assert "**truth OOM**" in body
    assert "**truth BQ_QUOTA**" in body


def test_misclassifications_appear_in_confusion_matrix(tmp_path: Path):
    out = tmp_path / "p3.md"
    y_true = np.array(["OOM"] * 4 + ["BQ_QUOTA"] * 4)
    y_pred = np.array(["OOM", "OOM", "OOM", "BQ_QUOTA",
                       "BQ_QUOTA", "BQ_QUOTA", "OOM", "BQ_QUOTA"])
    stats = write_phase3_report(
        output_path=out,
        y_true=y_true, y_pred=y_pred,
        labels=["BQ_QUOTA", "OOM"],
        n_features=5, n_splits=4,
    )
    assert stats["accuracy"] == 6 / 8
    body = out.read_text(encoding="utf-8")
    assert "Confusion matrix" in body
