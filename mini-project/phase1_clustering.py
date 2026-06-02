"""
phase1_clustering.py — Phase 1 of airflow-log-rca-ml (mini-project)
====================================================================

Mines log templates with Drain3, builds per-(dag,run) template histograms,
and uses Fisher's exact test to surface the templates that distinguish
failed runs from successful runs of the same DAG. Writes a Markdown RCA
report under ``mini-project/reports/``.

Dataset: synthetic Airflow logs (loaded via ``datasets/1_synthetic.py``).

Read top-to-bottom. No package imports — uses importlib so the numbered
loader file (``1_synthetic.py``) stays usable as-is.

Run:
    python mini-project/phase1_clustering.py

Output:
    mini-project/reports/phase1_clusters.md
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from rich.console import Console
    from rich.table import Table
    from scipy.stats import fisher_exact
except ImportError as e:
    print(f"ERROR: missing dependency ({e}). Run:  pip install -r mini-project/requirements.txt",
          file=sys.stderr)
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REPORTS_DIR = HERE / "reports"
REPORT_PATH = REPORTS_DIR / "phase1_clusters.md"

# Significance thresholds for "this template distinguishes failed runs".
MIN_FAIL_RUNS_WITH = 2      # template must appear in >= this many failed runs
P_VALUE_THRESHOLD = 0.05
MIN_ODDS_RATIO = 2.0        # over-represented in failures by at least 2x
TOP_TEMPLATES_PER_DAG = 10  # cap report length
PER_RUN_TOP_K = 5           # discriminators surfaced per failed run

# Templates that ALWAYS appear when a task fails — useful as a sanity check
# but uninformative for root-cause analysis. We list them once at the top of
# the report and de-prioritise them in per-run RCA.
GENERIC_FAILURE_PATTERNS = (
    "task failed with exception",
    "marking task as failed",
    "traceback (most recent call last)",
    "task instance is in a temporary failed state",
    "transient error, will retry",
)
# Templates that match "File X line Y in Z" — the per-frame traceback header.
# Drain3 mines them as "File <*> line <*> in <*>" (or close variants). These
# are universal across every Python traceback and add no RCA value.
_TRACEBACK_FRAME_RE = re.compile(r"^\s*file\s+<\*>\s+line\s+<\*>", re.IGNORECASE)


def _is_generic(template: str) -> bool:
    t = template.lower()
    if any(p in t for p in GENERIC_FAILURE_PATTERNS):
        return True
    if _TRACEBACK_FRAME_RE.search(t):
        return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# Loader import (numbered filename is not importable via `import`)
# ════════════════════════════════════════════════════════════════════════════

def _import_loader(rel: str):
    """Load a numbered loader module via importlib."""
    path = HERE / "datasets" / f"{rel}.py"
    spec = importlib.util.spec_from_file_location(rel.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loader = _import_loader("1_synthetic")


# ════════════════════════════════════════════════════════════════════════════
# Drain3 setup
# ════════════════════════════════════════════════════════════════════════════

def build_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    # Don't persist state; mini is one-shot.
    config.profiling_enabled = False
    # Drain default similarity threshold (0.4) is fine; depth 4 is the default.
    return TemplateMiner(config=config)


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — feed every log line through Drain3, build histograms per (dag, run)
# ════════════════════════════════════════════════════════════════════════════

def build_histograms() -> tuple[
    dict[tuple[str, str], Counter[int]],   # (dag_id, run_id) -> {cluster_id: count}
    dict[int, str],                        # cluster_id -> template string
    dict[int, list[str]],                  # cluster_id -> 3 example raw messages
    TemplateMiner,
]:
    miner = build_miner()
    hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    examples: dict[int, list[str]] = defaultdict(list)
    templates: dict[int, str] = {}

    n_lines = 0
    for rec in loader.load_records():
        msg = rec["message"].strip()
        if not msg:
            continue
        result = miner.add_log_message(msg)
        cid = result["cluster_id"]
        hist[(rec["dag_id"], rec["run_id"])][cid] += 1
        templates[cid] = result["template_mined"]
        if len(examples[cid]) < 3:
            examples[cid].append(msg)
        n_lines += 1

    print(f"  fed {n_lines:,} log lines through Drain3")
    print(f"  discovered {len(templates):,} distinct templates")
    print(f"  built histograms for {len(hist):,} DAG runs")
    return hist, templates, examples, miner


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — for each DAG, find templates over-represented in failed runs
# ════════════════════════════════════════════════════════════════════════════

def discriminating_templates(
    hist: dict[tuple[str, str], Counter[int]],
    truth: dict[str, dict],
) -> dict[str, list[dict]]:
    """Per DAG: ranked list of templates that distinguish FAILED from SUCCESS runs."""
    # Bucket runs by (dag, outcome)
    runs_by_dag: dict[str, dict[str, list[Counter[int]]]] = defaultdict(
        lambda: {"SUCCESS": [], "FAILED": []}
    )
    for (dag_id, run_id), h in hist.items():
        tr = truth.get(run_id)
        if tr is None:
            continue
        outcome = tr["outcome"]
        if outcome not in ("SUCCESS", "FAILED"):
            continue
        runs_by_dag[dag_id][outcome].append(h)

    out: dict[str, list[dict]] = {}
    for dag_id, buckets in runs_by_dag.items():
        succ = buckets["SUCCESS"]
        fail = buckets["FAILED"]
        if not succ or not fail:
            continue
        all_cids: set[int] = set()
        for h in succ + fail:
            all_cids.update(h.keys())

        rows: list[dict] = []
        for cid in all_cids:
            n_fail_with = sum(1 for h in fail if cid in h)
            n_succ_with = sum(1 for h in succ if cid in h)
            if n_fail_with < MIN_FAIL_RUNS_WITH:
                continue
            n_fail_wo = len(fail) - n_fail_with
            n_succ_wo = len(succ) - n_succ_with
            odds, p = fisher_exact(
                [[n_fail_with, n_fail_wo], [n_succ_with, n_succ_wo]],
                alternative="greater",
            )
            if odds < MIN_ODDS_RATIO or p > P_VALUE_THRESHOLD:
                continue
            rows.append({
                "cluster_id": cid,
                "fail_with": n_fail_with,
                "fail_total": len(fail),
                "succ_with": n_succ_with,
                "succ_total": len(succ),
                "odds": odds,
                "p_value": p,
            })
        rows.sort(key=lambda r: (-r["odds"], r["p_value"]))
        out[dag_id] = rows[:TOP_TEMPLATES_PER_DAG]
    return out


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — for each failed run, the top discriminating templates that appeared in it
# ════════════════════════════════════════════════════════════════════════════

def global_discriminators(
    hist: dict[tuple[str, str], Counter[int]],
    truth: dict[str, dict],
    templates: dict[int, str],
) -> dict[int, dict]:
    """Across ALL DAGs: which clusters are over-represented in failed runs?

    We use this for per-run RCA so failure-mode-specific templates surface
    even in DAGs where their failure type is rare. Generic failure markers
    are deprioritised — kept but flagged.
    """
    succ: list[Counter[int]] = []
    fail: list[Counter[int]] = []
    for (_, run_id), h in hist.items():
        tr = truth.get(run_id)
        if tr is None:
            continue
        (fail if tr["outcome"] == "FAILED" else succ).append(h)
    if not succ or not fail:
        return {}

    all_cids: set[int] = set()
    for h in succ + fail:
        all_cids.update(h.keys())

    out: dict[int, dict] = {}
    for cid in all_cids:
        n_f = sum(1 for h in fail if cid in h)
        n_s = sum(1 for h in succ if cid in h)
        if n_f < MIN_FAIL_RUNS_WITH:
            continue
        odds, p = fisher_exact(
            [[n_f, len(fail) - n_f], [n_s, len(succ) - n_s]],
            alternative="greater",
        )
        if odds < MIN_ODDS_RATIO or p > P_VALUE_THRESHOLD:
            continue
        out[cid] = {
            "cluster_id": cid,
            "fail_with": n_f, "fail_total": len(fail),
            "succ_with": n_s, "succ_total": len(succ),
            "odds": odds, "p_value": p,
            "is_generic": _is_generic(templates.get(cid, "")),
        }
    return out


def per_run_rca(
    hist: dict[tuple[str, str], Counter[int]],
    truth: dict[str, dict],
    global_discrim: dict[int, dict],
) -> list[dict]:
    """For each FAILED run, rank the discriminator clusters present in it
    by odds, preferring non-generic templates."""
    out: list[dict] = []
    for (dag_id, run_id), h in hist.items():
        tr = truth.get(run_id)
        if tr is None or tr["outcome"] != "FAILED":
            continue
        candidates = [
            {"cluster_id": cid, "count_in_run": cnt, **global_discrim[cid]}
            for cid, cnt in h.items() if cid in global_discrim
        ]
        # rank: non-generic first, then by odds desc
        candidates.sort(key=lambda c: (c["is_generic"], -c["odds"]))
        out.append({
            "dag_id": dag_id,
            "run_id": run_id,
            "failure_mode_truth": tr.get("failure_mode", ""),
            "top_templates": candidates[:PER_RUN_TOP_K],
        })
    return out


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — write Markdown report
# ════════════════════════════════════════════════════════════════════════════

def write_report(
    truth: dict[str, dict],
    templates: dict[int, str],
    examples: dict[int, list[str]],
    discrim: dict[str, list[dict]],
    global_discrim: dict[int, dict],
    per_run: list[dict],
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    n_total = len(truth)
    n_failed = sum(1 for t in truth.values() if t["outcome"] == "FAILED")
    n_succ = n_total - n_failed

    lines: list[str] = []
    lines.append("# Phase 1 — Template clustering + RCA report")
    lines.append("")
    lines.append(f"**Dataset:** 1 — Synthetic Airflow logs")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Corpus summary")
    lines.append("")
    lines.append(f"- DAG runs: **{n_total}** ({n_succ} SUCCESS, {n_failed} FAILED)")
    lines.append(f"- Distinct templates discovered by Drain3: **{len(templates)}**")
    lines.append(f"- DAGs with both successes and failures: **{len(discrim)}**")
    lines.append("")

    # --- Generic failure markers, called out once at the top ---
    generic_cids = sorted(
        (cid for cid, d in global_discrim.items() if d["is_generic"]),
        key=lambda c: -global_discrim[c]["fail_with"],
    )
    if generic_cids:
        lines.append("## Generic failure markers (informational)")
        lines.append("")
        lines.append("These templates appear in essentially every failed run regardless "
                     "of the underlying cause — they confirm that the task failed but "
                     "don't tell you *why*. They're excluded from per-DAG discriminator "
                     "tables below and de-prioritised in per-run RCA.")
        lines.append("")
        lines.append("| Cluster | Failed-runs coverage | Template |")
        lines.append("|---:|---:|---|")
        for cid in generic_cids:
            d = global_discrim[cid]
            tmpl = templates.get(cid, "<unknown>").replace("|", "\\|")
            if len(tmpl) > 90:
                tmpl = tmpl[:87] + "…"
            lines.append(f"| {cid} | {d['fail_with']}/{d['fail_total']} | `{tmpl}` |")
        lines.append("")

    lines.append("## Failure-mode-specific templates per DAG")
    lines.append("")
    lines.append("For each DAG with both successes and failures, we list the templates "
                 "whose presence in a run is a statistically significant predictor of "
                 "failure (one-sided Fisher's exact test, p < 0.05, odds ratio ≥ 2). "
                 "**Generic markers from the section above are filtered out so you see "
                 "the templates that point at the actual root cause.**")
    lines.append("")

    for dag_id in sorted(discrim):
        # Filter out generic markers from per-DAG tables.
        rows = [r for r in discrim[dag_id]
                if not _is_generic(templates.get(r["cluster_id"], ""))]
        if not rows:
            continue
        lines.append(f"### {dag_id}")
        lines.append("")
        lines.append("| Cluster | Failed runs with | Successful runs with | Odds ratio | p-value | Template |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for r in rows:
            cid = r["cluster_id"]
            tmpl = templates.get(cid, "<unknown>")
            tmpl_md = tmpl.replace("|", "\\|")
            if len(tmpl_md) > 100:
                tmpl_md = tmpl_md[:97] + "…"
            lines.append(
                f"| {cid} | {r['fail_with']}/{r['fail_total']} | "
                f"{r['succ_with']}/{r['succ_total']} | "
                f"{r['odds']:.1f} | {r['p_value']:.2e} | `{tmpl_md}` |"
            )
        lines.append("")
        # add one example raw line per top template
        lines.append("<details><summary>Example raw lines</summary>")
        lines.append("")
        for r in rows:
            cid = r["cluster_id"]
            for ex in examples.get(cid, [])[:1]:
                lines.append(f"- Cluster {cid}: `{ex[:160]}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("## Per-failed-run RCA snapshot")
    lines.append("")
    lines.append(f"For each failed run, the top {PER_RUN_TOP_K} **non-generic** "
                 f"discriminating templates that appeared in it, ranked by global "
                 f"odds ratio. The first template snippet should usually point at the "
                 f"actual failure mode. (showing first 40 of {len(per_run)})")
    lines.append("")
    lines.append("| DAG | Run | Truth failure_mode | Strongest template (snippet) |")
    lines.append("|---|---|---|---|")
    for entry in per_run[:40]:
        if entry["top_templates"]:
            top = entry["top_templates"][0]
            tmpl = templates.get(top["cluster_id"], "<unknown>")
            snippet = tmpl[:90] + ("…" if len(tmpl) > 90 else "")
            snippet = snippet.replace("|", "\\|").replace("\n", " ")
            cell = f"cluster {top['cluster_id']}: `{snippet}`"
        else:
            cell = "_(no discriminating template surfaced)_"
        lines.append(
            f"| {entry['dag_id']} | `{entry['run_id'][-20:]}` "
            f"| `{entry['failure_mode_truth']}` | {cell} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH


# ════════════════════════════════════════════════════════════════════════════
# CLI summary via rich
# ════════════════════════════════════════════════════════════════════════════

def print_rich_summary(discrim: dict[str, list[dict]], templates: dict[int, str]) -> None:
    console = Console()
    table = Table(title="Phase 1 — top non-generic discriminator per DAG", show_lines=False)
    table.add_column("DAG", style="cyan", no_wrap=True)
    table.add_column("#non-generic", justify="right")
    table.add_column("strongest non-generic template (truncated)")
    table.add_column("odds", justify="right")
    for dag_id in sorted(discrim):
        rows = [r for r in discrim[dag_id]
                if not _is_generic(templates.get(r["cluster_id"], ""))]
        if not rows:
            table.add_row(dag_id, "0", "[dim]_(only generic markers)_[/dim]", "—")
            continue
        top = rows[0]
        tmpl = templates.get(top["cluster_id"], "<unknown>")
        if len(tmpl) > 70:
            tmpl = tmpl[:67] + "…"
        table.add_row(
            dag_id,
            str(len(rows)),
            tmpl,
            f"{top['odds']:.1f}",
        )
    console.print(table)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    truth = loader.load_truth()
    if not truth:
        print(f"ERROR: no ground truth at {loader.TRUTH_PATH}.", file=sys.stderr)
        print("Run:  python mini-project/generate_logs.py", file=sys.stderr)
        sys.exit(1)

    print("Phase 1 — template clustering + RCA")
    print("===================================")
    hist, templates, examples, _miner = build_histograms()
    discrim = discriminating_templates(hist, truth)
    glob = global_discriminators(hist, truth, templates)
    per_run = per_run_rca(hist, truth, glob)

    print()
    print_rich_summary(discrim, templates)

    out = write_report(truth, templates, examples, discrim, glob, per_run)
    print(f"\nReport written: {out}")


if __name__ == "__main__":
    main()
