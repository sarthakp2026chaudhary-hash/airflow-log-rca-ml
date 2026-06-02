"""Tests for the Phase 1 Markdown report writer."""

from __future__ import annotations

from pathlib import Path

from log_rca.ml.discrimination import DiscriminatingTemplate, PerRunRca
from log_rca.reports import write_phase1_report


def _disc(cid: int, generic: bool = False, odds: float = 5.0) -> DiscriminatingTemplate:
    return DiscriminatingTemplate(
        cluster_id=cid,
        fail_with=3, fail_total=3, succ_with=0, succ_total=10,
        odds=odds, p_value=0.01, is_generic=generic,
    )


def test_report_written_and_contains_sections(tmp_path: Path):
    out = tmp_path / "report.md"
    write_phase1_report(
        output_path=out,
        dataset_label="test",
        outcomes={"r1": "FAILED", "r2": "SUCCESS"},
        templates={1: "MemoryError", 99: "Task failed with exception"},
        per_dag_discrim={"A": [_disc(1)]},
        global_discrim={1: _disc(1), 99: _disc(99, generic=True)},
        per_run=[PerRunRca("A", "r1", "OOM", (_disc(1),))],
    )
    body = out.read_text(encoding="utf-8")
    assert body.startswith("# Phase 1")
    assert "Generic failure markers" in body
    assert "Failure-mode-specific templates per DAG" in body
    assert "Per-failed-run RCA snapshot" in body
    assert "MemoryError" in body
    assert "`OOM`" in body


def test_report_omits_generic_section_when_empty(tmp_path: Path):
    out = tmp_path / "report.md"
    write_phase1_report(
        output_path=out,
        dataset_label="test",
        outcomes={"r1": "FAILED"},
        templates={1: "MemoryError"},
        per_dag_discrim={"A": [_disc(1)]},
        global_discrim={1: _disc(1)},          # no generic entries
        per_run=[PerRunRca("A", "r1", "OOM", (_disc(1),))],
    )
    body = out.read_text(encoding="utf-8")
    assert "Generic failure markers" not in body


def test_report_handles_empty_per_run_top_templates(tmp_path: Path):
    out = tmp_path / "report.md"
    write_phase1_report(
        output_path=out,
        dataset_label="test",
        outcomes={"r1": "FAILED"},
        templates={1: "x"},
        per_dag_discrim={},
        global_discrim={},
        per_run=[PerRunRca("A", "r1", "", ())],
    )
    body = out.read_text(encoding="utf-8")
    assert "no discriminating template surfaced" in body


def test_long_template_is_truncated(tmp_path: Path):
    out = tmp_path / "report.md"
    long = "x" * 500
    write_phase1_report(
        output_path=out,
        dataset_label="test",
        outcomes={"r1": "FAILED"},
        templates={1: long},
        per_dag_discrim={"A": [_disc(1)]},
        global_discrim={1: _disc(1)},
        per_run=[PerRunRca("A", "r1", "OOM", (_disc(1),))],
    )
    body = out.read_text(encoding="utf-8")
    # ellipsis present + the full 500-x string was never rendered intact
    assert "…" in body
    assert long not in body
    # the longest run of x's in any single cell stays under ~110 chars
    import re
    longest_run = max((len(m.group(0)) for m in re.finditer(r"x+", body)), default=0)
    assert longest_run <= 110
