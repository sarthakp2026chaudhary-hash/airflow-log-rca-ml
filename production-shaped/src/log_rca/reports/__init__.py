"""Markdown report writers (one per phase / dataset shape)."""

from log_rca.reports.phase1 import write_phase1_report
from log_rca.reports.phase1_loghub import write_phase1_loghub_report

__all__ = ["write_phase1_report", "write_phase1_loghub_report"]
