"""Tests for the LogHubHadoopLoader (dataset 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from log_rca.datasets import HadoopRecord, LogHubHadoopLoader


@pytest.fixture
def hadoop_dir(tmp_path: Path) -> Path:
    d = tmp_path / "hadoop"
    d.mkdir()
    (d / "Hadoop_2k.log_structured.csv").write_text(
        "LineId,Date,Time,Level,Process,Component,Content,EventId,EventTemplate\n"
        "1,2015-10-18,\"18:01:47,978\",INFO,main,"
        "org.apache.hadoop.mapreduce.v2.app.MRAppMaster,"
        "Created MRAppMaster for application appattempt_1,E29,"
        "Created MRAppMaster for application appattempt_<*>\n"
        "2,2015-10-18,\"18:01:48,963\",WARN,main,"
        "org.apache.hadoop.mapreduce.v2.app.MRAppMaster,"
        "Failed to renew lease for foo,E44,Failed to renew lease for <*>\n",
        encoding="utf-8",
    )
    (d / "Hadoop_2k.log_templates.csv").write_text(
        "EventId,EventTemplate\n"
        "E29,Created MRAppMaster for application appattempt_<*>\n"
        "E44,Failed to renew lease for <*>\n",
        encoding="utf-8",
    )
    return d


def test_load_records_typed(hadoop_dir: Path):
    loader = LogHubHadoopLoader(hadoop_dir)
    recs = list(loader.load_records())
    assert len(recs) == 2
    assert all(isinstance(r, HadoopRecord) for r in recs)

    r0 = recs[0]
    assert r0.line_id == 1
    assert r0.ts == "2015-10-18 18:01:47,978"
    assert r0.level == "INFO"
    assert r0.process == "main"
    assert r0.component == "org.apache.hadoop.mapreduce.v2.app.MRAppMaster"
    assert r0.event_id == "E29"

    r1 = recs[1]
    assert r1.level == "WARN"
    assert r1.event_id == "E44"


def test_load_templates(hadoop_dir: Path):
    loader = LogHubHadoopLoader(hadoop_dir)
    tmpls = loader.load_templates()
    assert tmpls == {
        "E29": "Created MRAppMaster for application appattempt_<*>",
        "E44": "Failed to renew lease for <*>",
    }


def test_summary(hadoop_dir: Path):
    s = LogHubHadoopLoader(hadoop_dir).summary()
    assert s["records"] == 2
    assert s["templates"] == 2
    assert s["levels"] == {"INFO": 1, "WARN": 1}
    assert s["components"] == 1
    assert ("E29", 1) in s["top_events"]
    assert ("E44", 1) in s["top_events"]


def test_missing_csv_raises(tmp_path: Path):
    loader = LogHubHadoopLoader(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="Hadoop_2k.log_structured.csv"):
        list(loader.load_records())


def test_missing_templates_yields_empty_dict(tmp_path: Path):
    d = tmp_path / "hadoop"
    d.mkdir()
    loader = LogHubHadoopLoader(d)
    assert loader.load_templates() == {}
