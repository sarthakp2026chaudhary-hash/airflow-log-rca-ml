"""
phase2_anomaly.py — Phase 2 of airflow-log-rca-ml (mini-project)
==================================================================

Featurises each DAG run as (template histogram + run duration + line
count + unique-template count + attempt count), fits IsolationForest on
SUCCESS runs ONLY (one-class outlier detection), then scores every run.

Anomalous runs that turn out to be FAILED in the ground truth = true
positives. Anomalous runs that are SUCCESS = false alarms (worth
inspecting because they may be "weird-but-survived").

Run:
    python mini-project/phase2_anomaly.py

Output:
    mini-project/reports/phase2_anomalies.md
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from rich.console import Console
    from rich.table import Table
    from sklearn.ensemble import IsolationForest
except ImportError as e:
    print(f"ERROR: missing dependency ({e}). Run:  pip install -r mini-project/requirements.txt",
          file=sys.stderr)
    sys.exit(1)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REPORTS_DIR = HERE / "reports"
REPORT_PATH = REPORTS_DIR / "phase2_anomalies.md"

TOP_N_ANOMALIES = 25
CONTAMINATION = "auto"   # let sklearn pick; tune later if needed
RANDOM_STATE = 42


# Numbered-filename loader import
def _import_loader(rel: str):
    path = HERE / "datasets" / f"{rel}.py"
    spec = importlib.util.spec_from_file_location(rel.replace(".", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loader = _import_loader("1_synthetic")


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Drain3 mining + per-run features
# ════════════════════════════════════════════════════════════════════════════

def build_features() -> tuple[
    list[tuple[str, str]],   # run keys in row order
    np.ndarray,              # feature matrix (n_runs, n_features)
    list[str],               # column names
    dict[int, str],          # cluster_id -> template
    dict[tuple[str, str], Counter[int]],   # raw histograms for reporting
]:
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)

    hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    line_count: dict[tuple[str, str], int] = defaultdict(int)
    attempts_seen: dict[tuple[str, str], set[int]] = defaultdict(set)

    for rec in loader.load_records():
        msg = rec["message"].strip()
        if not msg:
            continue
        result = miner.add_log_message(msg)
        cid = int(result["cluster_id"])
        key = (rec["dag_id"], rec["run_id"])
        hist[key][cid] += 1
        line_count[key] += 1
        attempts_seen[key].add(rec["attempt"])

    templates = {
        int(c.cluster_id): c.get_template()
        for c in miner.drain.clusters
    }
    all_cids = sorted(templates.keys())
    cid_index = {cid: i for i, cid in enumerate(all_cids)}

    # Build dense matrix: one row per run, columns = [hist counts | line_count | unique_templates | attempts]
    keys = sorted(hist.keys())
    n_extra = 3
    X = np.zeros((len(keys), len(all_cids) + n_extra), dtype=np.float32)
    for i, key in enumerate(keys):
        for cid, cnt in hist[key].items():
            X[i, cid_index[cid]] = cnt
        X[i, -3] = line_count[key]
        X[i, -2] = len(hist[key])               # unique-template count
        X[i, -1] = len(attempts_seen[key])      # attempt count

    columns = [f"cluster_{c}" for c in all_cids] + [
        "line_count", "unique_templates", "attempts",
    ]
    return keys, X, columns, templates, hist


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Fit IsolationForest on SUCCESS-only rows; score everything
# ════════════════════════════════════════════════════════════════════════════

def detect_anomalies(
    keys: list[tuple[str, str]],
    X: np.ndarray,
    truth: dict[str, dict],
) -> tuple[IsolationForest, np.ndarray]:
    success_mask = np.array([
        truth.get(run_id, {}).get("outcome") == "SUCCESS"
        for _, run_id in keys
    ])
    if success_mask.sum() < 10:
        raise RuntimeError("Not enough SUCCESS runs to fit IsolationForest")

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        max_samples="auto",
    )
    model.fit(X[success_mask])
    # decision_function: HIGHER = more normal, lower = more anomalous
    scores = model.decision_function(X)
    return model, scores


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Markdown report
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    truth = loader.load_truth()
    if not truth:
        print(f"ERROR: no ground truth at {loader.TRUTH_PATH}.", file=sys.stderr)
        print("Run:  python mini-project/generate_logs.py", file=sys.stderr)
        sys.exit(1)

    print("Phase 2 -- IsolationForest anomaly detection")
    print("============================================")
    keys, X, cols, templates, hist = build_features()
    print(f"  fitted feature matrix shape: {X.shape}")

    _model, scores = detect_anomalies(keys, X, truth)

    # Rank runs by anomaly: most-anomalous first
    order = np.argsort(scores)   # ascending → most anomalous first
    top = order[:TOP_N_ANOMALIES]

    # Verification: how many of top-N anomalies were actually FAILED?
    n_failed_in_top = sum(
        1 for i in top
        if truth.get(keys[i][1], {}).get("outcome") == "FAILED"
    )
    total_failed = sum(1 for t in truth.values() if t["outcome"] == "FAILED")
    precision_at_k = n_failed_in_top / max(TOP_N_ANOMALIES, 1)
    recall_at_k = n_failed_in_top / max(total_failed, 1)

    print(
        f"  top-{TOP_N_ANOMALIES} anomalies: {n_failed_in_top}/{TOP_N_ANOMALIES} "
        f"are FAILED in ground truth (precision@k={precision_at_k:.0%}, "
        f"recall@k={recall_at_k:.0%} of {total_failed} total failures)"
    )

    # Rich CLI summary
    console = Console()
    t = Table(title=f"Phase 2 — top {min(15, TOP_N_ANOMALIES)} anomalous DAG runs")
    t.add_column("Rank", justify="right")
    t.add_column("DAG", style="cyan")
    t.add_column("Run (tail)", overflow="fold")
    t.add_column("Score", justify="right")
    t.add_column("Truth outcome", style="bold")
    t.add_column("Failure mode")
    for rank, i in enumerate(top[:15], start=1):
        dag_id, run_id = keys[i]
        tr = truth.get(run_id, {})
        outcome = tr.get("outcome", "?")
        style = "red" if outcome == "FAILED" else "green"
        t.add_row(
            str(rank), dag_id, run_id[-20:],
            f"{scores[i]:.4f}",
            f"[{style}]{outcome}[/]",
            tr.get("failure_mode", "") or "—",
        )
    console.print(t)

    # Markdown report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 2 — Anomaly detection (synthetic Airflow logs)")
    lines.append("")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Each DAG run is featurised as a dense vector: per-template counts "
        f"(over the {len(templates)} clusters Drain3 mined) plus three "
        "extra features — total line count, number of unique templates, and "
        "attempt count. IsolationForest is fit on SUCCESS runs only (one-class "
        "outlier detection) and then scores every run. Lower decision score = "
        "more anomalous."
    )
    lines.append("")
    lines.append("## Verification @ k="
                 f"{TOP_N_ANOMALIES}")
    lines.append("")
    lines.append(
        f"- Total runs: **{len(keys)}** "
        f"({sum(1 for t in truth.values() if t['outcome']=='SUCCESS')} SUCCESS, "
        f"{total_failed} FAILED)\n"
        f"- Top-{TOP_N_ANOMALIES} most-anomalous runs by score:\n"
        f"  - **{n_failed_in_top}** are FAILED in ground truth → "
        f"precision@{TOP_N_ANOMALIES} = **{precision_at_k:.0%}**\n"
        f"  - covers **{n_failed_in_top}/{total_failed}** of all failures → "
        f"recall@{TOP_N_ANOMALIES} = **{recall_at_k:.0%}**"
    )
    lines.append("")

    lines.append("## Top anomalous runs")
    lines.append("")
    lines.append("| Rank | DAG | Run | Score | Truth | Failure mode | Dominant templates |")
    lines.append("|---:|---|---|---:|---|---|---|")
    for rank, i in enumerate(top, start=1):
        dag_id, run_id = keys[i]
        tr = truth.get(run_id, {})
        outcome = tr.get("outcome", "?")
        mode = tr.get("failure_mode", "") or "—"
        # 3 dominant templates by raw count in this run
        h = hist[keys[i]]
        dominant = h.most_common(3)
        dom_str = ", ".join(
            f"{cid}({n})" for cid, n in dominant
        )
        lines.append(
            f"| {rank} | {dag_id} | `{run_id[-20:]}` | {scores[i]:.4f} | "
            f"{outcome} | `{mode}` | {dom_str} |"
        )
    lines.append("")

    lines.append("## Failure-mode distribution in top-K anomalies")
    lines.append("")
    fm_in_top: Counter[str] = Counter()
    for i in top:
        m = truth.get(keys[i][1], {}).get("failure_mode") or "(SUCCESS)"
        fm_in_top[m] += 1
    lines.append("| Failure mode | Count in top-K |")
    lines.append("|---|---:|")
    for mode, n in fm_in_top.most_common():
        lines.append(f"| `{mode}` | {n} |")
    lines.append("")

    lines.append("## What the false alarms tell you")
    lines.append("")
    lines.append(
        "SUCCESS runs in the top-K are 'weird-but-survived' — runs whose log "
        "shape is unusual compared to the rest of their DAG. These are worth "
        "looking at because they may be: retries that succeeded after a "
        "near-miss, runs that processed unusual data volumes, or genuine "
        "noise. They are NOT classifier-grade signal for Phase 3 — Phase 2's "
        "job is just to flag them for inspection."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
