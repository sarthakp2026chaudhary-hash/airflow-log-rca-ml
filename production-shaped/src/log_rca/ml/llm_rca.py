"""LLM-based root-cause summariser (Phase 4).

Pluggable backend: Claude (Anthropic) or stub for offline / no-key dev.
Bundles a response cache so re-runs don't re-spend on identical contexts
(keyed by dag_id + predicted failure_mode + template-set hash).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


# ─── Prompt definition ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an on-call data-platform engineer at a large bank. Apache Airflow "
    "DAGs run on Google Cloud Composer and write their logs to GCS. When a "
    "task fails, you read the trailing portion of its log and produce a brief "
    "root-cause analysis. Be precise and concrete — no padding, no caveats. "
    "Always reply in the exact Markdown structure:\n\n"
    "### Root cause\n<one sentence>\n\n### Contributing factors\n"
    "1. <factor>\n2. <factor>\n3. <factor>\n\n### Suggested fix\n"
    "<one short paragraph>"
)


def build_user_prompt(
    *,
    dag_id: str,
    run_id: str,
    failure_mode: str,
    top_templates: list[tuple[int, str]],
    log_tail: list[str],
    tail_max: int = 30,
) -> str:
    tpl_lines = "\n".join(
        f"- cluster {cid}: `{tmpl}`" for cid, tmpl in top_templates
    )
    tail_block = "\n".join(log_tail[-tail_max:])
    return (
        f"DAG: `{dag_id}`\n"
        f"Run: `{run_id}`\n"
        f"Predicted failure mode (from upstream classifier): `{failure_mode}`\n\n"
        f"Top discriminating templates in this run:\n{tpl_lines}\n\n"
        f"Last ~{tail_max} log lines of the failing task:\n```\n{tail_block}\n```\n\n"
        f"Produce the RCA in the exact Markdown structure described in the system prompt."
    )


# ─── Cache key ─────────────────────────────────────────────────────────────

def template_hash(template_ids: Iterable[int]) -> str:
    s = ",".join(str(i) for i in sorted(template_ids))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def cache_key(dag_id: str, failure_mode: str, template_ids: Iterable[int]) -> str:
    return f"{dag_id}|{failure_mode}|{template_hash(template_ids)}"


# ─── Backend protocol + implementations ────────────────────────────────────

@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class LLMBackend(Protocol):
    """Minimal surface needed to summarise an RCA from a user prompt."""

    name: str

    def summarise(self, user_prompt: str) -> tuple[str, LLMUsage]:
        ...


class StubBackend:
    """Returns a canned response. No API calls; safe for tests & CI."""

    name = "stub"

    def __init__(self, failure_mode_hint: str = "UNKNOWN", dag_hint: str = ""):
        self._mode = failure_mode_hint
        self._dag = dag_hint

    def summarise(self, user_prompt: str) -> tuple[str, LLMUsage]:
        # try to recover the failure_mode/dag_id from the prompt for nicer output
        mode = self._mode
        dag = self._dag
        for line in user_prompt.splitlines():
            if line.startswith("DAG:"):
                dag = line.split("`", 2)[1] if "`" in line else dag
            elif line.startswith("Predicted failure mode"):
                mode = line.split("`")[-2] if line.count("`") >= 2 else mode
        text = (
            f"### Root cause\n"
            f"`{mode}` failure in DAG `{dag}` — see top templates for the "
            f"specific error signature.\n\n"
            f"### Contributing factors\n"
            f"1. The task encountered the canonical {mode} condition.\n"
            f"2. Upstream retries did not recover the run.\n"
            f"3. (stub backend — no live LLM call was made; set "
            f"ANTHROPIC_API_KEY and use ClaudeBackend for real RCA.)\n\n"
            f"### Suggested fix\n"
            f"Apply the standard remediation for `{mode}` and add a corresponding alert."
        )
        return text, LLMUsage()


class ClaudeBackend:
    """Real Anthropic-API backend using the Messages API with prompt caching."""

    name = "claude"

    def __init__(self, *, model: str, max_tokens: int = 600):
        # Import lazily so stub-only environments don't need the SDK at import time.
        import anthropic
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model
        self._max_tokens = max_tokens

    def summarise(self, user_prompt: str) -> tuple[str, LLMUsage]:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(b.text for b in message.content if hasattr(b, "text"))
        usage = LLMUsage(
            input_tokens=getattr(message.usage, "input_tokens", 0),
            output_tokens=getattr(message.usage, "output_tokens", 0),
            cache_read_input_tokens=getattr(
                message.usage, "cache_read_input_tokens", 0
            ),
            cache_creation_input_tokens=getattr(
                message.usage, "cache_creation_input_tokens", 0
            ),
        )
        return text, usage


def auto_backend(*, model: str) -> LLMBackend:
    """ClaudeBackend if ANTHROPIC_API_KEY is set, else StubBackend."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeBackend(model=model)
    return StubBackend()


# ─── Summariser with cache ─────────────────────────────────────────────────

@dataclass(frozen=True)
class RcaRequest:
    dag_id: str
    run_id: str
    failure_mode: str
    top_templates: list[tuple[int, str]]
    log_tail: list[str]


@dataclass(frozen=True)
class RcaResult:
    request: RcaRequest
    text: str
    usage: LLMUsage
    cache_hit: bool


class LLMRcaSummarizer:
    """Iterates over RCA requests, deduping via the on-disk cache."""

    def __init__(self, backend: LLMBackend, cache_path: Path | None = None):
        self._backend = backend
        self._cache_path = cache_path
        self._cache: dict[str, str] = {}
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def summarise(self, req: RcaRequest) -> RcaResult:
        ck = cache_key(
            req.dag_id, req.failure_mode,
            [cid for cid, _ in req.top_templates],
        )
        if ck in self._cache:
            return RcaResult(req, self._cache[ck], LLMUsage(), cache_hit=True)
        user_prompt = build_user_prompt(
            dag_id=req.dag_id, run_id=req.run_id,
            failure_mode=req.failure_mode,
            top_templates=req.top_templates,
            log_tail=req.log_tail,
        )
        text, usage = self._backend.summarise(user_prompt)
        self._cache[ck] = text
        return RcaResult(req, text, usage, cache_hit=False)

    def persist_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, indent=2), encoding="utf-8"
            )
