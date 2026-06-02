"""Phase 1 Markdown report writer.

Pure formatting; takes pre-computed structures from
``log_rca.ml.discrimination`` plus the run-truth dict and produces a
self-contained ``.md`` file.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from log_rca.ml.discrimination import (
    DiscriminatingTemplate,
    PerRunRca,
    is_generic_template,
)


def _trunc(s: str, n: int = 100) -> str:
    s = s.replace("|", "\\|").replace("\n", " ")
    return s[: n - 1] + "…" if len(s) > n else s


def write_phase1_report(
    *,
    output_path: Path,
    dataset_label: str,
    outcomes: dict[str, str],
    templates: dict[int, str],
    per_dag_discrim: dict[str, list[DiscriminatingTemplate]],
    global_discrim: dict[int, DiscriminatingTemplate],
    per_run: list[PerRunRca],
    per_run_show: int = 40,
) -> Path:
    """Write the Phase 1 RCA report to ``output_path``. Returns the path."""
    n_total = len(outcomes)
    n_failed = sum(1 for o in outcomes.values() if o == "FAILED")
    n_succ = n_total - n_failed

    lines: list[str] = []
    lines.append("# Phase 1 — Template clustering + RCA report")
    lines.append("")
    lines.append(f"**Dataset:** {dataset_label}")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Corpus summary")
    lines.append("")
    lines.append(f"- DAG runs: **{n_total}** ({n_succ} SUCCESS, {n_failed} FAILED)")
    lines.append(f"- Distinct templates discovered by Drain3: **{len(templates)}**")
    lines.append(f"- DAGs with both successes and failures: **{len(per_dag_discrim)}**")
    lines.append("")

    # generic markers up top
    generic = sorted(
        (d for d in global_discrim.values() if d.is_generic),
        key=lambda d: -d.fail_with,
    )
    if generic:
        lines.append("## Generic failure markers (informational)")
        lines.append("")
        lines.append(
            "These templates appear in essentially every failed run regardless of "
            "the underlying cause — they confirm a task failed but don't tell you "
            "*why*. They're excluded from per-DAG discriminator tables and "
            "de-prioritised in per-run RCA."
        )
        lines.append("")
        lines.append("| Cluster | Failed-runs coverage | Template |")
        lines.append("|---:|---:|---|")
        for d in generic:
            tmpl = _trunc(templates.get(d.cluster_id, "<unknown>"), 90)
            lines.append(
                f"| {d.cluster_id} | {d.fail_with}/{d.fail_total} | `{tmpl}` |"
            )
        lines.append("")

    # per-DAG sections
    lines.append("## Failure-mode-specific templates per DAG")
    lines.append("")
    lines.append(
        "For each DAG with both successes and failures, the templates whose "
        "presence in a run is a statistically significant predictor of failure "
        "(one-sided Fisher's exact test, p ≤ 0.05, odds ratio ≥ 2). "
        "**Generic markers are filtered out** so the rows point at the actual "
        "root cause."
    )
    lines.append("")
    for dag_id in sorted(per_dag_discrim):
        rows = [r for r in per_dag_discrim[dag_id] if not r.is_generic]
        if not rows:
            continue
        lines.append(f"### {dag_id}")
        lines.append("")
        lines.append(
            "| Cluster | Failed runs with | Successful runs with "
            "| Odds ratio | p-value | Template |"
        )
        lines.append("|---:|---:|---:|---:|---:|---|")
        for r in rows:
            tmpl = _trunc(templates.get(r.cluster_id, "<unknown>"), 100)
            odds_s = "inf" if r.odds == float("inf") else f"{r.odds:.1f}"
            lines.append(
                f"| {r.cluster_id} | {r.fail_with}/{r.fail_total} | "
                f"{r.succ_with}/{r.succ_total} | {odds_s} | "
                f"{r.p_value:.2e} | `{tmpl}` |"
            )
        lines.append("")

    # per-run RCA
    lines.append("## Per-failed-run RCA snapshot")
    lines.append("")
    lines.append(
        f"For each failed run, the strongest **non-generic** discriminating template "
        f"that fired in it, ranked by global odds ratio. "
        f"(showing first {min(per_run_show, len(per_run))} of {len(per_run)})"
    )
    lines.append("")
    lines.append("| DAG | Run | Truth failure_mode | Strongest template (snippet) |")
    lines.append("|---|---|---|---|")
    for entry in per_run[:per_run_show]:
        if entry.top_templates:
            top = entry.top_templates[0]
            tmpl = _trunc(templates.get(top.cluster_id, "<unknown>"), 90)
            cell = f"cluster {top.cluster_id}: `{tmpl}`"
        else:
            cell = "_(no discriminating template surfaced)_"
        lines.append(
            f"| {entry.dag_id} | `{entry.run_id[-20:]}` "
            f"| `{entry.failure_mode_truth}` | {cell} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
