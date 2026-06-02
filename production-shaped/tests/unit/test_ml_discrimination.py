"""Tests for the statistical discriminator."""

from __future__ import annotations

from collections import Counter

import pytest

from log_rca.ml import Discriminator, is_generic_template


# ─── helpers ───────────────────────────────────────────────────────────────

def _build(runs: list[tuple[str, str, str, list[int]]]) -> tuple[
    dict[tuple[str, str], Counter[int]],
    dict[str, str],
]:
    """``runs = [(dag_id, run_id, outcome, [cluster_ids_seen]), ...]``"""
    hist: dict[tuple[str, str], Counter[int]] = {}
    outcomes: dict[str, str] = {}
    for dag, run, outcome, cids in runs:
        c: Counter[int] = Counter()
        for cid in cids:
            c[cid] += 1
        hist[(dag, run)] = c
        outcomes[run] = outcome
    return hist, outcomes


# ─── is_generic_template ───────────────────────────────────────────────────

@pytest.mark.parametrize("template", [
    "Task failed with exception",
    "TASK FAILED WITH EXCEPTION",   # case-insensitive
    "Marking task as FAILED. dag_id=x",
    "Traceback (most recent call last):",
    "  File <*> line <*> in <*>",   # traceback frame
    "[ERROR] - Task instance is in a temporary failed state and will be retried",
])
def test_is_generic_true(template: str):
    assert is_generic_template(template)


@pytest.mark.parametrize("template", [
    "MemoryError",
    "psycopg2.OperationalError: connection refused",
    "[ERROR] - Task exceeded timeout of 3600 seconds",
    "pyarrow.lib.ArrowInvalid: Schema mismatch",
    "quotaExceeded: Exceeded rate limits",
])
def test_is_generic_false(template: str):
    assert not is_generic_template(template)


# ─── per_dag ───────────────────────────────────────────────────────────────

def test_per_dag_surfaces_template_only_in_failures():
    # cluster 99 appears in every failed run of dag A, never in successes
    hist, outcomes = _build([
        ("A", "r1", "FAILED",  [99]),
        ("A", "r2", "FAILED",  [99]),
        ("A", "r3", "FAILED",  [99]),
        ("A", "r4", "SUCCESS", [1, 2]),
        ("A", "r5", "SUCCESS", [1, 2]),
        ("A", "r6", "SUCCESS", [1, 2]),
        ("A", "r7", "SUCCESS", [1, 2]),
    ])
    d = Discriminator()
    out = d.per_dag(hist, outcomes, templates={99: "MemoryError"})
    assert "A" in out
    cluster_ids = {r.cluster_id for r in out["A"]}
    assert 99 in cluster_ids


def test_per_dag_skips_dag_with_only_one_outcome():
    hist, outcomes = _build([
        ("A", "r1", "FAILED", [99]),
        ("A", "r2", "FAILED", [99]),
    ])
    out = Discriminator().per_dag(hist, outcomes, templates={99: "x"})
    assert "A" not in out


def test_per_dag_excludes_generic_by_default():
    hist, outcomes = _build([
        ("A", "r1", "FAILED",  [10]),
        ("A", "r2", "FAILED",  [10]),
        ("A", "r3", "FAILED",  [10]),
        ("A", "r4", "SUCCESS", [1]),
        ("A", "r5", "SUCCESS", [1]),
        ("A", "r6", "SUCCESS", [1]),
        ("A", "r7", "SUCCESS", [1]),
    ])
    out = Discriminator().per_dag(
        hist, outcomes,
        templates={10: "Task failed with exception"},   # generic
    )
    assert out.get("A", []) == []   # filtered out


def test_per_dag_includes_generic_when_disabled():
    hist, outcomes = _build([
        ("A", "r1", "FAILED",  [10]),
        ("A", "r2", "FAILED",  [10]),
        ("A", "r3", "FAILED",  [10]),
        ("A", "r4", "SUCCESS", [1]),
        ("A", "r5", "SUCCESS", [1]),
        ("A", "r6", "SUCCESS", [1]),
        ("A", "r7", "SUCCESS", [1]),
    ])
    out = Discriminator().per_dag(
        hist, outcomes,
        templates={10: "Task failed with exception"},
        exclude_generic=False,
    )
    assert any(r.cluster_id == 10 for r in out["A"])
    assert any(r.is_generic for r in out["A"])


# ─── globally ──────────────────────────────────────────────────────────────

def test_globally_pools_across_dags():
    # cluster 77 only appears in OOM-failing runs across many DAGs
    hist, outcomes = _build([
        ("A", "ra1", "FAILED",  [77]),
        ("A", "ra2", "FAILED",  [77]),
        ("B", "rb1", "FAILED",  [77]),
        ("C", "rc1", "FAILED",  [77]),
        ("A", "ra3", "SUCCESS", [1]),
        ("B", "rb2", "SUCCESS", [1]),
        ("C", "rc2", "SUCCESS", [1]),
        ("D", "rd1", "SUCCESS", [1]),
    ])
    out = Discriminator().globally(hist, outcomes, templates={77: "MemoryError"})
    assert 77 in out
    d = out[77]
    assert d.fail_with == 4
    assert d.succ_with == 0
    assert not d.is_generic


def test_globally_empty_when_no_failures():
    hist, outcomes = _build([
        ("A", "r1", "SUCCESS", [1]),
        ("A", "r2", "SUCCESS", [1]),
    ])
    assert Discriminator().globally(hist, outcomes, templates={1: "x"}) == {}


# ─── per_run_rca ──────────────────────────────────────────────────────────

def test_per_run_rca_returns_one_entry_per_failed_run():
    # Fisher's exact at p<0.05 needs more than 2v3 samples — go 5v5.
    hist, outcomes = _build([
        ("A", "r1",  "FAILED",  [77, 10]),
        ("A", "r2",  "FAILED",  [77, 10]),
        ("A", "r3",  "FAILED",  [77, 10]),
        ("A", "r4",  "FAILED",  [77, 10]),
        ("A", "r5",  "FAILED",  [77, 10]),
        ("A", "s1",  "SUCCESS", [1]),
        ("A", "s2",  "SUCCESS", [1]),
        ("A", "s3",  "SUCCESS", [1]),
        ("A", "s4",  "SUCCESS", [1]),
        ("A", "s5",  "SUCCESS", [1]),
    ])
    templates = {77: "MemoryError", 10: "Task failed with exception"}
    d = Discriminator()
    glob = d.globally(hist, outcomes, templates)
    failure_modes = {f"r{i}": "OOM" for i in range(1, 6)}
    per_run = d.per_run_rca(hist, outcomes, failure_modes, glob)
    assert len(per_run) == 5
    assert all(p.dag_id == "A" for p in per_run)
    # non-generic should come first
    for entry in per_run:
        assert entry.top_templates[0].cluster_id == 77   # MemoryError, not the FAILED marker
        assert not entry.top_templates[0].is_generic
        assert entry.failure_mode_truth == "OOM"
