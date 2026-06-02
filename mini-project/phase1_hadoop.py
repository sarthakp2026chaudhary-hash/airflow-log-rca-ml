"""
phase1_hadoop.py — Phase 1 on dataset 2 (LogHub Hadoop_2k)
============================================================

Runs the same Drain3 template-mining step as phase1_clustering.py but
against real Hadoop MapReduce job logs. Because the LogHub 2k sample
has no anomaly labels we can't do the Fisher's-exact discriminator
step, so the report focuses on:

  1. How many templates our Drain3 mined vs LogHub's pre-parsed 114
  2. Top templates by frequency (the "background noise" of a working cluster)
  3. ERROR / WARN / FATAL templates — the actual RCA candidates

Run:
    python mini-project/phase1_hadoop.py

Output:
    mini-project/reports/phase1_hadoop.md
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from rich.console import Console
    from rich.table import Table
except ImportError as e:
    print(f"ERROR: missing dependency ({e}). Run:  pip install -r mini-project/requirements.txt",
          file=sys.stderr)
    sys.exit(1)


HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"
REPORT_PATH = REPORTS_DIR / "phase1_hadoop.md"

TOP_N_FREQUENT = 15
TOP_N_ERROR_LEVEL = 20


# Loader import (numbered filename is not importable via `import`)
def _import_loader(rel: str):
    path = HERE / "datasets" / f"{rel}.py"
    spec = importlib.util.spec_from_file_location(rel.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loader = _import_loader("2_loghub_hadoop")


def main() -> None:
    print("Phase 1 (Hadoop) -- Drain3 template mining")
    print("==========================================")
    records = list(loader.load_records())
    loghub_templates = loader.load_templates()
    if not records:
        print("ERROR: no records loaded. Check data/2_loghub_hadoop/.", file=sys.stderr)
        sys.exit(1)

    # Run our own Drain3 over the raw `content` column.
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)

    our_counts: Counter[int] = Counter()
    our_templates: dict[int, str] = {}
    examples: dict[int, list[str]] = defaultdict(list)
    by_level: dict[str, Counter[int]] = defaultdict(Counter)

    for rec in records:
        msg = rec["content"].strip()
        if not msg:
            continue
        result = miner.add_log_message(msg)
        cid = int(result["cluster_id"])
        our_counts[cid] += 1
        our_templates[cid] = result["template_mined"]
        by_level[rec["level"]][cid] += 1
        if len(examples[cid]) < 3:
            examples[cid].append(msg)

    print(f"  records ingested:           {len(records):,}")
    print(f"  LogHub pre-parsed templates: {len(loghub_templates):,}")
    print(f"  Templates our Drain3 mined:  {len(our_templates):,}")

    # Top-frequent templates
    top_freq = our_counts.most_common(TOP_N_FREQUENT)

    # ERROR / WARN / FATAL templates ranked by frequency
    error_level_cids: Counter[int] = Counter()
    for level in ("ERROR", "FATAL", "WARN"):
        for cid, n in by_level[level].items():
            error_level_cids[cid] += n
    top_errors = error_level_cids.most_common(TOP_N_ERROR_LEVEL)

    # Rich CLI summary
    console = Console()
    t = Table(title="Hadoop — top ERROR/WARN/FATAL templates (RCA candidates)")
    t.add_column("Cluster", justify="right")
    t.add_column("Count", justify="right")
    t.add_column("Template (truncated)")
    for cid, n in top_errors[:10]:
        tmpl = our_templates.get(cid, "<unknown>")
        if len(tmpl) > 90:
            tmpl = tmpl[:87] + "..."
        t.add_row(str(cid), str(n), tmpl)
    console.print(t)

    # Markdown report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 1 — Template clustering (dataset 2 — LogHub Hadoop_2k)")
    lines.append("")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Corpus summary")
    lines.append("")
    lines.append(f"- Log lines: **{len(records):,}**")
    lines.append(f"- Levels seen: " + ", ".join(
        f"{lvl}={sum(by_level[lvl].values()):,}"
        for lvl in sorted(by_level) if by_level[lvl]
    ))
    lines.append(f"- Distinct templates mined by our Drain3: **{len(our_templates)}**")
    lines.append(f"- Distinct templates pre-parsed by LogHub: **{len(loghub_templates)}**")
    lines.append("")
    lines.append("> No anomaly labels exist in the LogHub 2k subset, so the Fisher's-exact "
                 "discriminator step from the synthetic Phase 1 is omitted. Instead we "
                 "focus on **ERROR / WARN / FATAL templates** as RCA candidates.")
    lines.append("")

    lines.append("## RCA candidates — ERROR / WARN / FATAL templates")
    lines.append("")
    lines.append("Sorted by frequency. These are the templates a Hadoop on-call engineer "
                 "would investigate first.")
    lines.append("")
    lines.append("| Cluster | Count | Level mix | Template |")
    lines.append("|---:|---:|---|---|")
    for cid, _ in top_errors:
        lvl_mix = ", ".join(
            f"{lvl}={by_level[lvl][cid]}"
            for lvl in ("ERROR", "FATAL", "WARN")
            if by_level[lvl][cid] > 0
        )
        tmpl = our_templates.get(cid, "<unknown>").replace("|", "\\|")
        if len(tmpl) > 110:
            tmpl = tmpl[:107] + "..."
        lines.append(f"| {cid} | {error_level_cids[cid]} | {lvl_mix} | `{tmpl}` |")
    lines.append("")

    lines.append("## Top-frequent templates (background noise of a healthy cluster)")
    lines.append("")
    lines.append("| Cluster | Count | Template |")
    lines.append("|---:|---:|---|")
    for cid, n in top_freq:
        tmpl = our_templates.get(cid, "<unknown>").replace("|", "\\|")
        if len(tmpl) > 110:
            tmpl = tmpl[:107] + "..."
        lines.append(f"| {cid} | {n} | `{tmpl}` |")
    lines.append("")

    lines.append("## Drain3 vs LogHub — template-count comparison")
    lines.append("")
    lines.append(
        f"- LogHub's pre-parsed Hadoop_2k has **{len(loghub_templates)}** templates "
        f"(E1–E{len(loghub_templates)}).\n"
        f"- Our Drain3 mined **{len(our_templates)}** templates from the same lines."
    )
    if len(our_templates) < len(loghub_templates):
        lines.append("- We're more aggressive at merging (lower template count) than LogHub's tuned config.")
    elif len(our_templates) > len(loghub_templates):
        lines.append("- We split more clusters than LogHub. Tightening Drain3's similarity threshold would converge.")
    else:
        lines.append("- Count matches exactly. Both configs converged.")
    lines.append("")
    lines.append("This is a useful cross-check: if both methods discover similar template "
                 "distributions, downstream phases (anomaly detection, classification) can trust "
                 "either source.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
