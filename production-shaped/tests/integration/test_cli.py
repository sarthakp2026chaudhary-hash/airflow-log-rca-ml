"""End-to-end test for the ``log-rca-gen`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

from log_rca.datagen.cli import main


def test_cli_generates_logs_and_truth(tmp_path: Path, capsys):
    bucket = tmp_path / "bucket"
    rc = main([
        "--bucket-root", str(bucket),
        "--total-runs", "5",
    ])
    assert rc == 0

    captured = capsys.readouterr()
    assert "Generating ~5 DAG runs" in captured.out
    assert "Done." in captured.out
    assert "Outcome breakdown:" in captured.out

    truth_path = bucket / "_truth.jsonl"
    assert truth_path.exists()
    lines = [json.loads(ln) for ln in truth_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 5
    for rec in lines:
        assert {"dag_id", "run_id", "outcome", "failure_mode", "task_count"} <= rec.keys()
        assert rec["outcome"] in {"SUCCESS", "FAILED"}

    log_files = list((bucket / "airflow-logs").rglob("*.log"))
    assert log_files, "expected at least one .log file"


def test_cli_resets_truth_file_between_runs(tmp_path: Path):
    bucket = tmp_path / "bucket"
    main(["--bucket-root", str(bucket), "--total-runs", "3"])
    first_truth = (bucket / "_truth.jsonl").read_text(encoding="utf-8")
    assert first_truth.count("\n") == 3

    # Second run should NOT append to first; truth file is reset.
    main(["--bucket-root", str(bucket), "--total-runs", "2"])
    second_truth = (bucket / "_truth.jsonl").read_text(encoding="utf-8")
    assert second_truth.count("\n") == 2
