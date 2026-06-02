"""Tests for the Drain3 TemplateMiner wrapper."""

from __future__ import annotations

from pathlib import Path

from log_rca.ml import TemplateMiner


def test_add_returns_mined_line():
    miner = TemplateMiner()
    out = miner.add("user 42 logged in")
    assert out.cluster_id >= 1
    assert out.message == "user 42 logged in"
    assert out.template  # Drain3 mines *something* from a single line


def test_similar_messages_share_cluster():
    # Drain3 only generalises once it sees varying tokens. Feed several
    # variants so the wildcard appears.
    miner = TemplateMiner()
    miner.add("user 42 logged in from host alpha")
    miner.add("user 99 logged in from host beta")
    last = miner.add("user 7 logged in from host gamma")
    # After 3 variants the template should be generalised.
    assert "<*>" in miner.templates()[last.cluster_id]


def test_different_messages_get_different_clusters():
    miner = TemplateMiner()
    a = miner.add("OOM killer triggered on host one")
    b = miner.add("Schema mismatch in column customer_id")
    assert a.cluster_id != b.cluster_id


def test_fit_returns_one_record_per_nonblank_line():
    miner = TemplateMiner()
    out = miner.fit([
        "line one with id 1",
        "",
        "  ",
        "line one with id 2",
    ])
    assert len(out) == 2
    assert out[0].cluster_id == out[1].cluster_id   # same template


def test_templates_dict_contains_known_clusters():
    miner = TemplateMiner()
    miner.add("foo bar 1")
    miner.add("foo bar 2")
    templates = miner.templates()
    assert len(templates) >= 1
    # every key is an int, every value a non-empty string
    for cid, tmpl in templates.items():
        assert isinstance(cid, int)
        assert isinstance(tmpl, str) and tmpl


def test_save_and_load_roundtrip(tmp_path: Path):
    miner = TemplateMiner()
    miner.add("user 1 logged in")
    miner.add("user 2 logged in")
    state = tmp_path / "state.pkl"
    miner.save(state)
    assert state.exists()

    loaded = TemplateMiner.load(state)
    assert loaded.cluster_count() == miner.cluster_count()
    assert loaded.templates() == miner.templates()
    # New messages on the loaded miner should reuse the existing cluster.
    new_line = loaded.add("user 99 logged in")
    assert new_line.cluster_id in loaded.templates()
