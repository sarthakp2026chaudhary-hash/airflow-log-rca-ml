"""
phase1_hdfs.py — Phase 1 on dataset 3 (LogHub HDFS_2k)
========================================================

Mirrors phase1_hadoop.py but on HDFS_2k. HDFS has a tiny template
vocabulary (14 in LogHub's mining) so output is compact — useful to
see how Drain3 behaves on a "narrow" log source.

Run:
    python mini-project/phase1_hdfs.py

Output:
    mini-project/reports/phase1_hdfs.md
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
REPORT_PATH = REPORTS_DIR / "phase1_hdfs.md"

TOP_N_FREQUENT = 14   # HDFS has only ~14 templates, show them all
TOP_N_ERROR_LEVEL = 10


def _import_loader(rel: str):
    path = HERE / "datasets" / f"{rel}.py"
    spec = importlib.util.spec_from_file_location(rel.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loader = _import_loader("3_loghub_hdfs")


def main() -> None:
    print("Phase 1 (HDFS) -- Drain3 template mining")
    print("========================================")
    records = list(loader.load_records())
    loghub_templates = loader.load_templates()
    if not records:
        print("ERROR: no records loaded. Check data/3_loghub_hdfs/.", file=sys.stderr)
        sys.exit(1)

    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)

    our_counts: Counter[int] = Counter()
    our_templates: dict[int, str] = {}
    by_level: dict[str, Counter[int]] = defaultdict(Counter)
    by_component: Counter[str] = Counter()

    for rec in records:
        msg = rec["content"].strip()
        if not msg:
            continue
        result = miner.add_log_message(msg)
        cid = int(result["cluster_id"])
        our_counts[cid] += 1
        our_templates[cid] = result["template_mined"]
        by_level[rec["level"]][cid] += 1
        by_component[rec["component"]] += 1

    print(f"  records ingested:           {len(records):,}")
    print(f"  LogHub pre-parsed templates: {len(loghub_templates):,}")
    print(f"  Templates our Drain3 mined:  {len(our_templates):,}")

    # HDFS components are concrete subsystems (DataNode, NameSystem, etc.)
    console = Console()
    t = Table(title="HDFS — template breakdown by Drain3")
    t.add_column("Cluster", justify="right")
    t.add_column("Count", justify="right")
    t.add_column("Template (truncated)")
    for cid, n in our_counts.most_common(10):
        tmpl = our_templates.get(cid, "<unknown>")
        if len(tmpl) > 90:
            tmpl = tmpl[:87] + "..."
        t.add_row(str(cid), str(n), tmpl)
    console.print(t)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 1 — Template clustering (dataset 3 — LogHub HDFS_2k)")
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
    lines.append(f"- Components seen: **{len(by_component)}** "
                 f"({', '.join(c for c, _ in by_component.most_common(3))}, …)")
    lines.append(f"- Distinct templates mined by our Drain3: **{len(our_templates)}**")
    lines.append(f"- Distinct templates pre-parsed by LogHub: **{len(loghub_templates)}**")
    lines.append("")
    lines.append("> The HDFS 2k sample is a narrow log source — only ~14 templates exist. "
                 "All HDFS_v1 academic anomaly-detection benchmarks use these block-level "
                 "operations as their primitive event vocabulary.")
    lines.append("")

    lines.append("## All Drain3-mined templates (HDFS is narrow so we show every one)")
    lines.append("")
    lines.append("| Cluster | Count | Level mix | Template |")
    lines.append("|---:|---:|---|---|")
    for cid, n in our_counts.most_common(TOP_N_FREQUENT):
        lvl_mix = ", ".join(
            f"{lvl}={by_level[lvl][cid]}"
            for lvl in sorted(by_level)
            if by_level[lvl][cid] > 0
        )
        tmpl = our_templates.get(cid, "<unknown>").replace("|", "\\|")
        if len(tmpl) > 110:
            tmpl = tmpl[:107] + "..."
        lines.append(f"| {cid} | {n} | {lvl_mix} | `{tmpl}` |")
    lines.append("")

    lines.append("## WARN-level templates (HDFS has no ERROR-level lines in the 2k sample)")
    lines.append("")
    warn_counts: Counter[int] = Counter()
    for cid, n in by_level.get("WARN", {}).items():
        warn_counts[cid] = n
    if not warn_counts:
        lines.append("_(none)_")
    else:
        lines.append("| Cluster | Count | Template |")
        lines.append("|---:|---:|---|")
        for cid, n in warn_counts.most_common(TOP_N_ERROR_LEVEL):
            tmpl = our_templates.get(cid, "<unknown>").replace("|", "\\|")
            if len(tmpl) > 110:
                tmpl = tmpl[:107] + "..."
            lines.append(f"| {cid} | {n} | `{tmpl}` |")
    lines.append("")

    lines.append("## Drain3 vs LogHub")
    lines.append("")
    lines.append(
        f"- LogHub's pre-parsed HDFS_2k has **{len(loghub_templates)}** templates "
        f"(E1–E{len(loghub_templates)}).\n"
        f"- Our Drain3 mined **{len(our_templates)}** templates from the same lines."
    )
    lines.append("")
    lines.append("HDFS is the canonical benchmark in the log-mining literature — full HDFS_v1 "
                 "(11M lines, on Zenodo) is what Drain, DeepLog, LogBERT etc. evaluate against. "
                 "Our 2k subset is enough to demonstrate the pipeline; for serious anomaly "
                 "detection (Phase 2+) we'd switch to the full corpus with `anomaly_label.csv`.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
