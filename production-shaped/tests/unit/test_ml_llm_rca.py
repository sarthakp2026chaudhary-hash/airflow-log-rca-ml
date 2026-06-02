"""Tests for the Phase 4 LLM RCA summariser."""

from __future__ import annotations

from pathlib import Path

import pytest

from log_rca.ml import (
    LLMRcaSummarizer,
    RcaRequest,
    StubBackend,
)
from log_rca.ml.llm_rca import (
    LLMUsage,
    build_user_prompt,
    cache_key,
    template_hash,
)


# ─── prompt / hashing helpers ─────────────────────────────────────────────

def test_template_hash_is_order_invariant():
    a = template_hash([1, 2, 3])
    b = template_hash([3, 1, 2])
    assert a == b


def test_template_hash_changes_with_different_ids():
    assert template_hash([1, 2, 3]) != template_hash([1, 2, 4])


def test_cache_key_combines_dag_mode_templates():
    k1 = cache_key("dagA", "OOM", [1, 2, 3])
    k2 = cache_key("dagB", "OOM", [1, 2, 3])
    k3 = cache_key("dagA", "OOM", [4])
    assert k1 != k2
    assert k1 != k3


def test_build_user_prompt_includes_required_fields():
    p = build_user_prompt(
        dag_id="etl_x", run_id="scheduled__1",
        failure_mode="OOM",
        top_templates=[(7, "MemoryError"), (8, "Killed")],
        log_tail=["last line"],
    )
    assert "etl_x" in p
    assert "OOM" in p
    assert "MemoryError" in p
    assert "last line" in p


# ─── Stub backend ─────────────────────────────────────────────────────────

def test_stub_backend_returns_structured_markdown():
    text, usage = StubBackend().summarise(
        "DAG: `d`\nPredicted failure mode (from upstream classifier): `OOM`\n"
    )
    assert "### Root cause" in text
    assert "### Contributing factors" in text
    assert "### Suggested fix" in text
    assert usage == LLMUsage()


# ─── Summariser + cache ───────────────────────────────────────────────────

def test_summariser_uses_cache_on_second_call(tmp_path: Path):
    cache_path = tmp_path / "cache.json"
    summariser = LLMRcaSummarizer(StubBackend(), cache_path=cache_path)
    req = RcaRequest(
        dag_id="d", run_id="r",
        failure_mode="OOM",
        top_templates=[(1, "MemoryError")],
        log_tail=["..."],
    )
    r1 = summariser.summarise(req)
    assert not r1.cache_hit
    r2 = summariser.summarise(req)
    assert r2.cache_hit
    assert r1.text == r2.text


def test_summariser_persists_cache(tmp_path: Path):
    cache_path = tmp_path / "cache.json"
    summariser = LLMRcaSummarizer(StubBackend(), cache_path=cache_path)
    req = RcaRequest(
        dag_id="d", run_id="r", failure_mode="OOM",
        top_templates=[(1, "x")], log_tail=[],
    )
    summariser.summarise(req)
    summariser.persist_cache()
    assert cache_path.exists()

    # Fresh instance picks up the cached entry
    summariser2 = LLMRcaSummarizer(StubBackend(), cache_path=cache_path)
    r = summariser2.summarise(req)
    assert r.cache_hit


def test_summariser_backend_name():
    summariser = LLMRcaSummarizer(StubBackend())
    assert summariser.backend_name == "stub"


# ─── ClaudeBackend (mocked) ───────────────────────────────────────────────

def test_claude_backend_uses_prompt_caching(monkeypatch):
    """Verify the ClaudeBackend builds a Messages payload with cache_control."""
    captured: dict = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs

            class _Usage:
                input_tokens = 12
                output_tokens = 34
                cache_read_input_tokens = 56
                cache_creation_input_tokens = 78

            class _Block:
                text = "### Root cause\nfake"

            class _Resp:
                content = [_Block()]
                usage = _Usage()

            return _Resp()

    class _FakeClient:
        messages = _FakeMessages()

    import sys
    import types

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda: _FakeClient()
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    from log_rca.ml.llm_rca import ClaudeBackend

    backend = ClaudeBackend(model="claude-sonnet-4-0")
    text, usage = backend.summarise("user prompt")

    assert "Root cause" in text
    assert usage.input_tokens == 12
    assert usage.cache_read_input_tokens == 56

    # And the request must include ephemeral cache_control on the system prompt
    sys_block = captured["kwargs"]["system"][0]
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    assert captured["kwargs"]["model"] == "claude-sonnet-4-0"
