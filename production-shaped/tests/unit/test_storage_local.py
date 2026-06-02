"""Tests for the LocalFSBackend storage."""

from __future__ import annotations

import pytest

from log_rca.storage import LocalFSBackend


def test_write_and_read_roundtrip(storage: LocalFSBackend):
    storage.write_text("foo/bar.txt", "hello")
    assert storage.read_text("foo/bar.txt") == "hello"


def test_write_creates_parent_dirs(storage: LocalFSBackend):
    storage.write_text("a/b/c/d.log", "x")
    assert storage.exists("a/b/c/d.log")


def test_append_line_creates_then_appends(storage: LocalFSBackend):
    storage.append_line("truth.jsonl", "one")
    storage.append_line("truth.jsonl", "two")
    body = storage.read_text("truth.jsonl")
    assert body == "one\ntwo\n"


def test_iter_keys_walks_recursively(storage: LocalFSBackend):
    storage.write_text("airflow-logs/dag_id=a/attempt=1.log", "x")
    storage.write_text("airflow-logs/dag_id=b/attempt=1.log", "y")
    storage.write_text("other/file.txt", "z")
    found = sorted(storage.iter_keys("airflow-logs"))
    assert found == [
        "airflow-logs/dag_id=a/attempt=1.log",
        "airflow-logs/dag_id=b/attempt=1.log",
    ]


def test_iter_keys_missing_prefix_yields_nothing(storage: LocalFSBackend):
    assert list(storage.iter_keys("does-not-exist")) == []


def test_rejects_path_traversal(storage: LocalFSBackend):
    with pytest.raises(ValueError, match="parent traversal"):
        storage.write_text("../escaped.txt", "nope")
