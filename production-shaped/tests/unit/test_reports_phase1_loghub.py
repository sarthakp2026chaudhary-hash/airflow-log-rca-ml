"""Tests for the LogHub-flavoured Phase 1 report writer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from log_rca.reports import write_phase1_loghub_report


def test_report_renders_with_no_errors(tmp_path: Path):
    out = tmp_path / "r.md"
    write_phase1_loghub_report(
        output_path=out,
        dataset_label="test",
        record_count=100,
        our_templates={1: "INFO template only"},
        loghub_template_count=1,
        counts_by_cid=Counter({1: 100}),
        counts_by_level={"INFO": Counter({1: 100})},
    )
    body = out.read_text(encoding="utf-8")
    assert "## Corpus summary" in body
    assert "Log lines: **100**" in body
    assert "no ERROR/WARN/FATAL lines" in body


def test_report_surfaces_error_templates(tmp_path: Path):
    out = tmp_path / "r.md"
    write_phase1_loghub_report(
        output_path=out,
        dataset_label="test",
        record_count=4,
        our_templates={1: "Connection refused", 2: "Normal heartbeat"},
        loghub_template_count=2,
        counts_by_cid=Counter({1: 2, 2: 2}),
        counts_by_level={
            "ERROR": Counter({1: 2}),
            "INFO": Counter({2: 2}),
        },
    )
    body = out.read_text(encoding="utf-8")
    assert "Connection refused" in body
    assert "ERROR=2" in body
    # Both clusters appear in the frequent-templates table
    assert "Normal heartbeat" in body


def test_template_count_difference_is_signed(tmp_path: Path):
    out = tmp_path / "r.md"
    write_phase1_loghub_report(
        output_path=out,
        dataset_label="test",
        record_count=1,
        our_templates={1: "x", 2: "y", 3: "z"},
        loghub_template_count=10,
        counts_by_cid=Counter({1: 1}),
        counts_by_level={"INFO": Counter({1: 1})},
    )
    body = out.read_text(encoding="utf-8")
    assert "Difference: -7" in body
