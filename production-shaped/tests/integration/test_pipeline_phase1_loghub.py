"""End-to-end test for the LogHub Hadoop pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from log_rca.pipeline.phase1_loghub import run_hadoop


@pytest.fixture
def hadoop_dir(tmp_path: Path) -> Path:
    """Tiny Hadoop-format fixture with one ERROR and three INFO lines."""
    d = tmp_path / "hadoop"
    d.mkdir()
    (d / "Hadoop_2k.log_structured.csv").write_text(
        "LineId,Date,Time,Level,Process,Component,Content,EventId,EventTemplate\n"
        "1,2015-10-18,18:01:47,INFO,main,c.app,Created MRAppMaster for foo,E1,\n"
        "2,2015-10-18,18:01:48,INFO,main,c.app,Created MRAppMaster for bar,E1,\n"
        "3,2015-10-18,18:01:49,ERROR,main,c.app,Connection refused to host,E2,\n"
        "4,2015-10-18,18:01:50,INFO,main,c.app,Heartbeat ok,E3,\n",
        encoding="utf-8",
    )
    (d / "Hadoop_2k.log_templates.csv").write_text(
        "EventId,EventTemplate\n"
        "E1,Created MRAppMaster for <*>\n"
        "E2,Connection refused to host\n"
        "E3,Heartbeat ok\n",
        encoding="utf-8",
    )
    return d


def test_pipeline_runs_end_to_end(hadoop_dir: Path, tmp_path: Path):
    report = tmp_path / "out.md"
    stats = run_hadoop(data_dir=hadoop_dir, report_path=report)
    assert report.exists()
    body = report.read_text(encoding="utf-8")
    # ERROR template surfaces
    assert "Connection refused" in body
    # Counts
    assert stats["records_processed"] == 4
    assert stats["templates_loghub"] == 3
    assert stats["levels"]["ERROR"] == 1
    assert stats["levels"]["INFO"] == 3


def test_pipeline_saves_state(hadoop_dir: Path, tmp_path: Path):
    state = tmp_path / "state.pkl"
    run_hadoop(
        data_dir=hadoop_dir,
        report_path=tmp_path / "r.md",
        drain3_state_path=state,
    )
    assert state.exists() and state.stat().st_size > 0


def test_pipeline_raises_when_data_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_hadoop(data_dir=tmp_path / "missing", report_path=tmp_path / "r.md")
