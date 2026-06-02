"""Loader for dataset 2 — LogHub Hadoop_2k pre-parsed CSV.

LogHub already parses every line through Drain and labels it with an
``EventId``/``EventTemplate``, so this loader is small: it just reads
the structured CSV and yields typed records in the Hadoop schema.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class HadoopRecord:
    """One line from Hadoop_2k.log_structured.csv (LogHub-parsed)."""

    line_id: int
    ts: str            # "<Date> <Time>"
    level: str         # INFO / WARN / ERROR / FATAL
    process: str
    component: str
    content: str
    event_id: str      # E1..E114
    event_template: str


class LogHubHadoopLoader:
    """Reads ``data/2_loghub_hadoop/Hadoop_2k.log_structured.csv``."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()

    # ----- paths -----

    @property
    def raw_log_path(self) -> Path:
        return self.data_dir / "Hadoop_2k.log"

    @property
    def structured_csv_path(self) -> Path:
        return self.data_dir / "Hadoop_2k.log_structured.csv"

    @property
    def templates_csv_path(self) -> Path:
        return self.data_dir / "Hadoop_2k.log_templates.csv"

    # ----- public API -----

    def load_records(self) -> Iterator[HadoopRecord]:
        if not self.structured_csv_path.exists():
            raise FileNotFoundError(
                f"{self.structured_csv_path} not found. "
                "See data/README.md for the LogHub download instructions."
            )
        with self.structured_csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield HadoopRecord(
                    line_id=int(row["LineId"]),
                    ts=f"{row['Date']} {row['Time']}",
                    level=row["Level"],
                    process=row["Process"],
                    component=row["Component"],
                    content=row["Content"],
                    event_id=row["EventId"],
                    event_template=row["EventTemplate"],
                )

    def load_templates(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if not self.templates_csv_path.exists():
            return out
        with self.templates_csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                out[row["EventId"]] = row["EventTemplate"]
        return out

    def summary(self) -> dict[str, object]:
        by_level: dict[str, int] = {}
        by_event: dict[str, int] = {}
        by_component: dict[str, int] = {}
        n = 0
        for rec in self.load_records():
            n += 1
            by_level[rec.level] = by_level.get(rec.level, 0) + 1
            by_event[rec.event_id] = by_event.get(rec.event_id, 0) + 1
            by_component[rec.component] = by_component.get(rec.component, 0) + 1
        templates = self.load_templates()
        return {
            "records": n,
            "templates": len(templates),
            "levels": by_level,
            "components": len(by_component),
            "top_events": sorted(by_event.items(), key=lambda kv: -kv[1])[:10],
        }
