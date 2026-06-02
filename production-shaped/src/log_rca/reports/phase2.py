"""Phase 2 Markdown report — anomaly detection (synthetic dataset)."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path


def write_phase2_report(
    *,
    output_path: Path,
    keys: list[tuple[str, str]],          # (dag_id, run_id) for each row
    scores: list[float],                   # decision_function output (same length)
    outcomes: dict[str, str],              # run_id -> SUCCESS/FAILED
    failure_modes: dict[str, str],         # run_id -> failure mode (or "")
    dominant_templates: dict[tuple[str, str], list[tuple[int, int]]],  # key -> [(cid, count), ...]
    top_k: int = 25,
    n_templates_mined: int = 0,
) -> dict:
    """Write the Phase 2 report and return summary stats."""
    n_total = len(keys)
    n_success = sum(1 for o in outcomes.values() if o == "SUCCESS")
    n_failed = sum(1 for o in outcomes.values() if o == "FAILED")

    # rank ascending (most anomalous first)
    order = sorted(range(n_total), key=lambda i: scores[i])
    top = order[:top_k]
    n_failed_in_top = sum(
        1 for i in top if outcomes.get(keys[i][1]) == "FAILED"
    )
    precision_at_k = n_failed_in_top / max(top_k, 1)
    recall_at_k = n_failed_in_top / max(n_failed, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 2 — Anomaly detection (synthetic Airflow logs)")
    lines.append("")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "IsolationForest fit on SUCCESS rows only (one-class outlier detection), "
        f"scored against all {n_total} runs. Feature vector per run = "
        f"template histogram (over {n_templates_mined} Drain3 clusters) + line count + "
        "unique-template count + attempt count. Lower score = more anomalous."
    )
    lines.append("")

    lines.append(f"## Verification @ k={top_k}")
    lines.append("")
    lines.append(
        f"- Total runs: **{n_total}** ({n_success} SUCCESS, {n_failed} FAILED)\n"
        f"- Top-{top_k} most-anomalous runs by score:\n"
        f"  - **{n_failed_in_top}** are FAILED in ground truth → "
        f"precision@{top_k} = **{precision_at_k:.0%}**\n"
        f"  - covers **{n_failed_in_top}/{n_failed}** of all failures → "
        f"recall@{top_k} = **{recall_at_k:.0%}**"
    )
    lines.append("")

    lines.append("## Top anomalous runs")
    lines.append("")
    lines.append("| Rank | DAG | Run | Score | Truth | Failure mode | Dominant templates |")
    lines.append("|---:|---|---|---:|---|---|---|")
    for rank, i in enumerate(top, start=1):
        dag_id, run_id = keys[i]
        outcome = outcomes.get(run_id, "?")
        mode = failure_modes.get(run_id, "") or "—"
        dominant = dominant_templates.get(keys[i], [])[:3]
        dom_str = ", ".join(f"{cid}({n})" for cid, n in dominant)
        lines.append(
            f"| {rank} | {dag_id} | `{run_id[-20:]}` | {scores[i]:.4f} | "
            f"{outcome} | `{mode}` | {dom_str} |"
        )
    lines.append("")

    fm_in_top: Counter[str] = Counter()
    for i in top:
        m = failure_modes.get(keys[i][1]) or "(SUCCESS)"
        fm_in_top[m] += 1
    lines.append("## Failure-mode distribution in top-K anomalies")
    lines.append("")
    lines.append("| Failure mode | Count in top-K |")
    lines.append("|---|---:|")
    for mode, n in fm_in_top.most_common():
        lines.append(f"| `{mode}` | {n} |")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "n_total": n_total, "n_success": n_success, "n_failed": n_failed,
        "top_k": top_k, "n_failed_in_top": n_failed_in_top,
        "precision_at_k": precision_at_k, "recall_at_k": recall_at_k,
    }
