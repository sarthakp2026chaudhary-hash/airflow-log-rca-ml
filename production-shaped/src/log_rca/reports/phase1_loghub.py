"""Phase 1 Markdown report — LogHub-shaped datasets (no labels).

Differs from ``phase1.py`` because LogHub's 2k subsets have no anomaly
labels, so we cannot run a Fisher's-exact discriminator. Instead this
report focuses on log-level breakdown and surfaces ERROR / WARN / FATAL
templates as RCA candidates.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path


def _trunc(s: str, n: int = 110) -> str:
    s = s.replace("|", "\\|").replace("\n", " ")
    return s[: n - 3] + "..." if len(s) > n else s


def write_phase1_loghub_report(
    *,
    output_path: Path,
    dataset_label: str,
    record_count: int,
    our_templates: dict[int, str],
    loghub_template_count: int,
    counts_by_cid: Counter[int],
    counts_by_level: dict[str, Counter[int]],
    top_n_frequent: int = 15,
    top_n_error_level: int = 20,
    error_levels: tuple[str, ...] = ("ERROR", "FATAL", "WARN"),
) -> Path:
    """Render a LogHub-flavoured Phase 1 report. Returns the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_cids: Counter[int] = Counter()
    for lvl in error_levels:
        for cid, n in counts_by_level.get(lvl, {}).items():
            error_cids[cid] += n

    lines: list[str] = []
    lines.append(f"# Phase 1 — Template clustering ({dataset_label})")
    lines.append("")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Corpus summary")
    lines.append("")
    lines.append(f"- Log lines: **{record_count:,}**")
    level_mix = ", ".join(
        f"{lvl}={sum(counts_by_level[lvl].values()):,}"
        for lvl in sorted(counts_by_level) if counts_by_level[lvl]
    )
    lines.append(f"- Levels seen: {level_mix}")
    lines.append(f"- Distinct templates mined by our Drain3: **{len(our_templates)}**")
    lines.append(f"- Distinct templates pre-parsed by LogHub: **{loghub_template_count}**")
    lines.append("")
    lines.append("> No anomaly labels exist in the LogHub 2k subset, so the "
                 "Fisher's-exact discriminator step from the synthetic Phase 1 is "
                 "omitted. Instead we focus on **ERROR / WARN / FATAL templates** "
                 "as RCA candidates.")
    lines.append("")

    lines.append(f"## RCA candidates — top {top_n_error_level} "
                 f"{'/'.join(error_levels)} templates")
    lines.append("")
    if error_cids:
        lines.append("| Cluster | Count | Level mix | Template |")
        lines.append("|---:|---:|---|---|")
        for cid, _ in error_cids.most_common(top_n_error_level):
            lvl_mix = ", ".join(
                f"{lvl}={counts_by_level.get(lvl, {}).get(cid, 0)}"
                for lvl in error_levels
                if counts_by_level.get(lvl, {}).get(cid, 0) > 0
            )
            tmpl = _trunc(our_templates.get(cid, "<unknown>"))
            lines.append(f"| {cid} | {error_cids[cid]} | {lvl_mix} | `{tmpl}` |")
    else:
        lines.append("_(no ERROR/WARN/FATAL lines in this subset)_")
    lines.append("")

    lines.append(f"## Top-{top_n_frequent} frequent templates (background noise)")
    lines.append("")
    lines.append("| Cluster | Count | Template |")
    lines.append("|---:|---:|---|")
    for cid, n in counts_by_cid.most_common(top_n_frequent):
        tmpl = _trunc(our_templates.get(cid, "<unknown>"))
        lines.append(f"| {cid} | {n} | `{tmpl}` |")
    lines.append("")

    lines.append("## Drain3 vs LogHub — template-count comparison")
    lines.append("")
    diff = len(our_templates) - loghub_template_count
    lines.append(
        f"- LogHub's pre-parsed mining: **{loghub_template_count}** templates\n"
        f"- Our Drain3 mining: **{len(our_templates)}** templates\n"
        f"- Difference: {diff:+d}"
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
