"""
Dataset 3 loader — LogHub HDFS_2k (mini-project)
=================================================

Real Hadoop Distributed File System block-level logs from logpai/loghub,
2,000-line sampled subset. The single most-studied dataset in log-mining
academia — the Drain and DeepLog papers benchmark on full HDFS_v1.

Run standalone:
    python mini-project/datasets/3_loghub_hdfs.py

Yields one record per log LINE in the HDFS schema (note: it has Pid where
Hadoop has Process — different sources, different columns, no normalisation).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "3_loghub_hdfs"
RAW_LOG = DATA_DIR / "HDFS_2k.log"
STRUCTURED_CSV = DATA_DIR / "HDFS_2k.log_structured.csv"
TEMPLATES_CSV = DATA_DIR / "HDFS_2k.log_templates.csv"

# Columns in HDFS_2k.log_structured.csv:
#   LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate


def load_records() -> Iterator[dict]:
    """Yield one record per LINE from the LogHub-parsed CSV."""
    if not STRUCTURED_CSV.exists():
        raise FileNotFoundError(
            f"{STRUCTURED_CSV} not found. See data/README.md to download."
        )
    with STRUCTURED_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "line_id":        int(row["LineId"]),
                "ts":             f"{row['Date']} {row['Time']}",
                "pid":            int(row["Pid"]),
                "level":          row["Level"],
                "component":      row["Component"],
                "content":        row["Content"],
                "event_id":       row["EventId"],
                "event_template": row["EventTemplate"],
            }


def load_templates() -> dict[str, str]:
    out: dict[str, str] = {}
    if not TEMPLATES_CSV.exists():
        return out
    with TEMPLATES_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["EventId"]] = row["EventTemplate"]
    return out


def summarise() -> None:
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} not found.", file=sys.stderr)
        sys.exit(1)

    records = list(load_records())
    templates = load_templates()
    by_level: dict[str, int] = {}
    by_event: dict[str, int] = {}
    by_component: dict[str, int] = {}
    for rec in records:
        by_level[rec["level"]] = by_level.get(rec["level"], 0) + 1
        by_event[rec["event_id"]] = by_event.get(rec["event_id"], 0) + 1
        by_component[rec["component"]] = by_component.get(rec["component"], 0) + 1

    top_events = sorted(by_event.items(), key=lambda kv: -kv[1])[:10]

    print("Dataset 3: LogHub HDFS_2k")
    print(f"  data dir:        {DATA_DIR}")
    print(f"  raw log file:    {RAW_LOG.name} ({RAW_LOG.stat().st_size // 1024} KB)")
    print(f"  parsed records:  {len(records):,}")
    print(f"  templates:       {len(templates)}")
    print(f"  levels seen:     {dict(sorted(by_level.items()))}")
    print(f"  unique components: {len(by_component)}")
    print(f"  top-10 templates by frequency:")
    for eid, count in top_events:
        tmpl = templates.get(eid, "<unknown>")
        snippet = (tmpl[:75] + "…") if len(tmpl) > 75 else tmpl
        print(f"    {eid:<5} {count:>5}  {snippet}")


if __name__ == "__main__":
    summarise()
