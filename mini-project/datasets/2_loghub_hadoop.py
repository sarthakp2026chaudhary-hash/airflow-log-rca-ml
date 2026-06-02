"""
Dataset 2 loader — LogHub Hadoop_2k (mini-project)
===================================================

Real Hadoop MapReduce job logs from logpai/loghub, 2,000-line sampled
subset. LogHub pre-parsed every line into a structured CSV with EventId
and EventTemplate, so we get cluster assignments for free.

Run standalone:
    python mini-project/datasets/2_loghub_hadoop.py

Yields one record per log LINE in the Hadoop schema (different from the
Airflow schema — by design, we do not normalise across datasets).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "2_loghub_hadoop"
RAW_LOG = DATA_DIR / "Hadoop_2k.log"
STRUCTURED_CSV = DATA_DIR / "Hadoop_2k.log_structured.csv"
TEMPLATES_CSV = DATA_DIR / "Hadoop_2k.log_templates.csv"

# Columns in Hadoop_2k.log_structured.csv:
#   LineId, Date, Time, Level, Process, Component, Content, EventId, EventTemplate


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
                "level":          row["Level"],
                "process":        row["Process"],
                "component":      row["Component"],
                "content":        row["Content"],
                "event_id":       row["EventId"],
                "event_template": row["EventTemplate"],
            }


def load_templates() -> dict[str, str]:
    """Return ``{EventId: EventTemplate}`` from Hadoop_2k.log_templates.csv."""
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
        print("Datasets are committed in the repo; this folder should exist.", file=sys.stderr)
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

    print("Dataset 2: LogHub Hadoop_2k")
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
