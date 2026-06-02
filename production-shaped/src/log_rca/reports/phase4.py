"""Phase 4 Markdown report — LLM RCA summaries."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from log_rca.ml.llm_rca import LLMUsage, RcaResult


def write_phase4_report(
    *,
    output_path: Path,
    backend_name: str,
    model: str,
    total_failed: int,
    results: list[RcaResult],
    total_usage: LLMUsage,
    cache_hits: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 4 — LLM RCA summaries (synthetic Airflow logs)")
    lines.append("")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(
        f"**Backend:** `{backend_name}`  |  "
        f"**Model:** `{model if backend_name == 'claude' else '(stub)'}`"
    )
    lines.append(
        f"**Failed runs sampled:** {len(results)} of {total_failed} total"
    )
    if backend_name == "claude":
        lines.append("")
        lines.append(
            f"**Token usage:** input={total_usage.input_tokens:,}, "
            f"output={total_usage.output_tokens:,}, "
            f"cache_read={total_usage.cache_read_input_tokens:,}, "
            f"cache_create={total_usage.cache_creation_input_tokens:,}, "
            f"cache_hits={cache_hits}/{len(results)}"
        )
    elif backend_name == "stub":
        lines.append("")
        lines.append(
            "> **Stub mode** — no LLM call was made. Set `ANTHROPIC_API_KEY` "
            "and re-run for real RCA summaries."
        )
    lines.append("")

    for r in results:
        lines.append(
            f"## `{r.request.dag_id}` — `{r.request.run_id[-20:]}` "
            f"({r.request.failure_mode})"
        )
        if r.cache_hit:
            lines.append("")
            lines.append("_(cache hit — no LLM call this run)_")
        lines.append("")
        lines.append(r.text)
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
