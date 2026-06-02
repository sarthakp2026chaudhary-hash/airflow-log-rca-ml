"""Statistical discovery of failure-correlated templates.

Pure functions over ``(dag_id, run_id) -> Counter[cluster_id]`` histograms
and ``{run_id: outcome}`` truth labels. No I/O.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from scipy.stats import fisher_exact


# ─── Configuration ─────────────────────────────────────────────────────────

DEFAULT_MIN_FAIL_RUNS_WITH = 2
DEFAULT_P_VALUE_THRESHOLD = 0.05
DEFAULT_MIN_ODDS_RATIO = 2.0

# Drain3 templates we treat as "universal failure markers": present in
# essentially every failed task regardless of cause. Excluded from per-DAG
# discriminator tables and de-prioritised in per-run RCA.
GENERIC_FAILURE_PATTERNS: tuple[str, ...] = (
    "task failed with exception",
    "marking task as failed",
    "traceback (most recent call last)",
    "task instance is in a temporary failed state",
    "transient error, will retry",
)
_TRACEBACK_FRAME_RE = re.compile(r"^\s*file\s+<\*>\s+line\s+<\*>", re.IGNORECASE)


def is_generic_template(template: str) -> bool:
    """True if ``template`` is a universal failure marker (no RCA value)."""
    t = template.lower()
    if any(p in t for p in GENERIC_FAILURE_PATTERNS):
        return True
    return bool(_TRACEBACK_FRAME_RE.search(t))


# ─── Result types ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DiscriminatingTemplate:
    cluster_id: int
    fail_with: int
    fail_total: int
    succ_with: int
    succ_total: int
    odds: float
    p_value: float
    is_generic: bool


@dataclass(frozen=True)
class PerRunRca:
    dag_id: str
    run_id: str
    failure_mode_truth: str
    top_templates: tuple[DiscriminatingTemplate, ...]


# ─── Public API ────────────────────────────────────────────────────────────

class Discriminator:
    """Builds per-DAG and global discriminator tables from run histograms."""

    def __init__(
        self,
        *,
        min_fail_runs_with: int = DEFAULT_MIN_FAIL_RUNS_WITH,
        p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
        min_odds_ratio: float = DEFAULT_MIN_ODDS_RATIO,
    ):
        self.min_fail_runs_with = min_fail_runs_with
        self.p_value_threshold = p_value_threshold
        self.min_odds_ratio = min_odds_ratio

    # ----- core math -----

    def _fisher(
        self, n_fail_with: int, n_fail_total: int,
        n_succ_with: int, n_succ_total: int,
    ) -> tuple[float, float]:
        return fisher_exact(
            [[n_fail_with, n_fail_total - n_fail_with],
             [n_succ_with, n_succ_total - n_succ_with]],
            alternative="greater",
        )

    def _keep(self, *, n_fail_with: int, odds: float, p: float) -> bool:
        return (
            n_fail_with >= self.min_fail_runs_with
            and odds >= self.min_odds_ratio
            and p <= self.p_value_threshold
        )

    # ----- per-DAG -----

    def per_dag(
        self,
        hist: dict[tuple[str, str], Counter[int]],
        outcome_by_run: dict[str, str],
        templates: dict[int, str],
        top_k: int = 10,
        exclude_generic: bool = True,
    ) -> dict[str, list[DiscriminatingTemplate]]:
        """Per DAG: ranked list of templates that distinguish FAILED from SUCCESS."""
        runs_by_dag: dict[str, dict[str, list[Counter[int]]]] = defaultdict(
            lambda: {"SUCCESS": [], "FAILED": []}
        )
        for (dag_id, run_id), h in hist.items():
            o = outcome_by_run.get(run_id)
            if o in ("SUCCESS", "FAILED"):
                runs_by_dag[dag_id][o].append(h)

        out: dict[str, list[DiscriminatingTemplate]] = {}
        for dag_id, buckets in runs_by_dag.items():
            succ = buckets["SUCCESS"]
            fail = buckets["FAILED"]
            if not succ or not fail:
                continue
            all_cids: set[int] = set()
            for h in succ + fail:
                all_cids.update(h.keys())

            rows: list[DiscriminatingTemplate] = []
            for cid in all_cids:
                n_f = sum(1 for h in fail if cid in h)
                n_s = sum(1 for h in succ if cid in h)
                odds, p = self._fisher(n_f, len(fail), n_s, len(succ))
                if not self._keep(n_fail_with=n_f, odds=odds, p=p):
                    continue
                generic = is_generic_template(templates.get(cid, ""))
                if exclude_generic and generic:
                    continue
                rows.append(DiscriminatingTemplate(
                    cluster_id=cid,
                    fail_with=n_f, fail_total=len(fail),
                    succ_with=n_s, succ_total=len(succ),
                    odds=float(odds), p_value=float(p),
                    is_generic=generic,
                ))
            rows.sort(key=lambda r: (-r.odds, r.p_value))
            out[dag_id] = rows[:top_k]
        return out

    # ----- global -----

    def globally(
        self,
        hist: dict[tuple[str, str], Counter[int]],
        outcome_by_run: dict[str, str],
        templates: dict[int, str],
    ) -> dict[int, DiscriminatingTemplate]:
        """Across all DAGs: which clusters are over-represented in failures?"""
        succ: list[Counter[int]] = []
        fail: list[Counter[int]] = []
        for (_, run_id), h in hist.items():
            o = outcome_by_run.get(run_id)
            if o == "SUCCESS":
                succ.append(h)
            elif o == "FAILED":
                fail.append(h)
        if not succ or not fail:
            return {}

        all_cids: set[int] = set()
        for h in succ + fail:
            all_cids.update(h.keys())

        out: dict[int, DiscriminatingTemplate] = {}
        for cid in all_cids:
            n_f = sum(1 for h in fail if cid in h)
            n_s = sum(1 for h in succ if cid in h)
            odds, p = self._fisher(n_f, len(fail), n_s, len(succ))
            if not self._keep(n_fail_with=n_f, odds=odds, p=p):
                continue
            out[cid] = DiscriminatingTemplate(
                cluster_id=cid,
                fail_with=n_f, fail_total=len(fail),
                succ_with=n_s, succ_total=len(succ),
                odds=float(odds), p_value=float(p),
                is_generic=is_generic_template(templates.get(cid, "")),
            )
        return out

    # ----- per-run RCA -----

    def per_run_rca(
        self,
        hist: dict[tuple[str, str], Counter[int]],
        outcome_by_run: dict[str, str],
        failure_mode_by_run: dict[str, str],
        global_discrim: dict[int, DiscriminatingTemplate],
        top_k: int = 5,
    ) -> list[PerRunRca]:
        """For each FAILED run, the top non-generic discriminators that fired."""
        out: list[PerRunRca] = []
        for (dag_id, run_id), h in hist.items():
            if outcome_by_run.get(run_id) != "FAILED":
                continue
            cands = [global_discrim[cid] for cid in h if cid in global_discrim]
            cands.sort(key=lambda c: (c.is_generic, -c.odds))
            out.append(PerRunRca(
                dag_id=dag_id,
                run_id=run_id,
                failure_mode_truth=failure_mode_by_run.get(run_id, ""),
                top_templates=tuple(cands[:top_k]),
            ))
        return out
