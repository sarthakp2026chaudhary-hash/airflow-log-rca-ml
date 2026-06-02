"""Tests for the Phase 2 anomaly report writer."""

from __future__ import annotations

from pathlib import Path

from log_rca.reports import write_phase2_report


def test_report_writes_and_computes_precision(tmp_path: Path):
    out = tmp_path / "p2.md"
    keys = [("dag", f"r{i}") for i in range(10)]
    scores = [-1.0, -0.9, -0.8, -0.7, -0.6, 0.1, 0.2, 0.3, 0.4, 0.5]
    # rows 0-3 are FAILED, rest SUCCESS  → top-4 all failed → precision 100%
    outcomes = {f"r{i}": ("FAILED" if i < 4 else "SUCCESS") for i in range(10)}
    failure_modes = {f"r{i}": ("OOM" if i < 4 else "") for i in range(10)}
    dominant = {k: [(1, 5)] for k in keys}

    stats = write_phase2_report(
        output_path=out, keys=keys, scores=scores,
        outcomes=outcomes, failure_modes=failure_modes,
        dominant_templates=dominant, top_k=4, n_templates_mined=2,
    )
    body = out.read_text(encoding="utf-8")
    assert "# Phase 2" in body
    assert stats["n_total"] == 10
    assert stats["n_failed"] == 4
    assert stats["n_failed_in_top"] == 4
    assert stats["precision_at_k"] == 1.0
    assert stats["recall_at_k"] == 1.0


def test_report_handles_mixed_top_k(tmp_path: Path):
    """Top-K contains some failures and some successes (partial precision)."""
    keys = [("d", f"r{i}") for i in range(20)]
    scores = list(range(-10, 10))   # ascending
    # only rows 0, 2, 4 are FAILED; everything else SUCCESS
    outcomes = {
        f"r{i}": ("FAILED" if i in {0, 2, 4} else "SUCCESS") for i in range(20)
    }
    stats = write_phase2_report(
        output_path=tmp_path / "p.md",
        keys=keys, scores=scores,
        outcomes=outcomes, failure_modes={},
        dominant_templates={}, top_k=5, n_templates_mined=1,
    )
    # top-5 indexes 0,1,2,3,4 → failures = {0,2,4} = 3
    assert stats["n_failed_in_top"] == 3
    assert stats["precision_at_k"] == 3 / 5
    assert stats["recall_at_k"] == 1.0  # all 3 failures captured


def test_report_handles_no_failures(tmp_path: Path):
    keys = [("d", f"r{i}") for i in range(15)]
    scores = list(range(15))
    outcomes = {k[1]: "SUCCESS" for k in keys}
    stats = write_phase2_report(
        output_path=tmp_path / "p.md",
        keys=keys, scores=scores,
        outcomes=outcomes, failure_modes={},
        dominant_templates={}, top_k=5, n_templates_mined=1,
    )
    assert stats["n_failed"] == 0
    assert stats["n_failed_in_top"] == 0
    assert stats["precision_at_k"] == 0.0
    # recall is 0/0 → we guard with max(n_failed, 1) → 0/1 = 0
    assert stats["recall_at_k"] == 0.0
