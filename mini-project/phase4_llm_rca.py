"""
phase4_llm_rca.py — Phase 4 of airflow-log-rca-ml (mini-project)
==================================================================

For a sample of failed runs, sends Claude a compact RCA prompt and
captures the response. Output Markdown has one RCA section per run
(root cause, contributing factors, suggested fix).

Two backends:
  - 'claude'  : real Anthropic API (needs ANTHROPIC_API_KEY env var)
  - 'stub'    : canned templated response — for testing without spending

Default: 'claude' if ANTHROPIC_API_KEY is set, else 'stub'.

Cost guard: defaults to 5 failed runs sampled deterministically (--n).
Use --all if you want every failure scored.

Run:
    python mini-project/phase4_llm_rca.py             # uses stub if no key
    python mini-project/phase4_llm_rca.py --n 10
    ANTHROPIC_API_KEY=... python mini-project/phase4_llm_rca.py --backend claude

Output:
    mini-project/reports/phase4_llm_rca.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from rich.console import Console
except ImportError as e:
    print(f"ERROR: missing dependency ({e}). Run:  pip install -r mini-project/requirements.txt",
          file=sys.stderr)
    sys.exit(1)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
REPORTS_DIR = HERE / "reports"
REPORT_PATH = REPORTS_DIR / "phase4_llm_rca.md"
CACHE_PATH = REPORTS_DIR / "phase4_cache.json"

DEFAULT_N = 5
DEFAULT_MODEL = "claude-sonnet-4-0"
MAX_TOKENS = 600


# Numbered-filename loader
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
# Step 1 — sample failed runs and extract per-run context
# ════════════════════════════════════════════════════════════════════════════

def collect_failed_runs(truth: dict[str, dict], n: int | None) -> list[tuple[str, str, str]]:
    """Return list of (dag_id, run_id, failure_mode) for failed runs.
    If n is None, return all; else sample n deterministically.
    """
    pool = [
        (t["dag_id"], rid, t["failure_mode"])
        for rid, t in truth.items()
        if t["outcome"] == "FAILED"
    ]
    pool.sort()
    if n is None or n >= len(pool):
        return pool
    rng = random.Random(42)
    return sorted(rng.sample(pool, n))


def mine_templates_and_collect_lines(
    target_runs: set[tuple[str, str]],
) -> tuple[
    dict[int, str],                             # cluster_id -> template
    dict[tuple[str, str], Counter[int]],        # run -> histogram
    dict[tuple[str, str], list[str]],           # run -> last ~50 lines
]:
    """One pass through the corpus. For each target run, keep:
    - per-run template histogram
    - last 50 raw lines (the traceback tail is the most useful chunk)
    """
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)
    hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    tail: dict[tuple[str, str], list[str]] = defaultdict(list)

    for rec in loader.load_records():
        key = (rec["dag_id"], rec["run_id"])
        msg = rec["message"]
        if not msg.strip():
            continue
        result = miner.add_log_message(msg.strip())
        cid = int(result["cluster_id"])
        if key in target_runs:
            hist[key][cid] += 1
            tail[key].append(msg)
            if len(tail[key]) > 50:
                tail[key].pop(0)
    templates = {int(c.cluster_id): c.get_template() for c in miner.drain.clusters}
    return templates, hist, tail


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Prompt building
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an on-call data-platform engineer at a large bank. Apache Airflow "
    "DAGs run on Google Cloud Composer and write their logs to GCS. When a "
    "task fails, you read the trailing portion of its log and produce a brief "
    "root-cause analysis. Be precise and concrete — no padding, no caveats. "
    "Always reply in the exact Markdown structure:\n\n"
    "### Root cause\n<one sentence>\n\n### Contributing factors\n"
    "1. <factor>\n2. <factor>\n3. <factor>\n\n### Suggested fix\n<one short paragraph>"
)


def build_user_prompt(
    *,
    dag_id: str,
    run_id: str,
    failure_mode_predicted: str,
    top_templates: list[tuple[int, str]],
    log_tail: list[str],
) -> str:
    tpl_lines = "\n".join(
        f"- cluster {cid} (×N): `{tmpl}`" for cid, tmpl in top_templates
    )
    tail_block = "\n".join(log_tail[-30:])     # cap tail length
    return (
        f"DAG: `{dag_id}`\n"
        f"Run: `{run_id}`\n"
        f"Predicted failure mode (from upstream classifier): `{failure_mode_predicted}`\n\n"
        f"Top discriminating templates in this run:\n{tpl_lines}\n\n"
        f"Last ~30 log lines of the failing task:\n```\n{tail_block}\n```\n\n"
        f"Produce the RCA in the exact Markdown structure described in the system prompt."
    )


def _template_hash(template_ids: Iterable[int]) -> str:
    s = ",".join(str(i) for i in sorted(template_ids))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def cache_key(dag_id: str, failure_mode: str, template_ids: Iterable[int]) -> str:
    return f"{dag_id}|{failure_mode}|{_template_hash(template_ids)}"


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Backends
# ════════════════════════════════════════════════════════════════════════════

def stub_response(failure_mode: str, dag_id: str) -> str:
    return (
        f"### Root cause\n"
        f"`{failure_mode}` failure in DAG `{dag_id}` — see top templates for the "
        f"specific error signature.\n\n"
        f"### Contributing factors\n"
        f"1. The task encountered the canonical {failure_mode} condition during execution.\n"
        f"2. Upstream retries did not recover the run.\n"
        f"3. (stub mode — no live LLM call was made; set ANTHROPIC_API_KEY for real RCA.)\n\n"
        f"### Suggested fix\n"
        f"Apply the standard remediation for `{failure_mode}` and add a corresponding "
        f"alert. Re-run this script with `--backend claude` for a model-generated fix."
    )


def call_claude(user_prompt: str, model: str) -> tuple[str, dict]:
    """Returns (response_text, usage_dict)."""
    import anthropic   # imported lazily so stub mode doesn't require it
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},   # static system prompt → cache
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in message.content if hasattr(b, "text"))
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "cache_read_input_tokens": getattr(
            message.usage, "cache_read_input_tokens", 0
        ),
        "cache_creation_input_tokens": getattr(
            message.usage, "cache_creation_input_tokens", 0
        ),
    }
    return text, usage


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"sample N failed runs (default {DEFAULT_N})")
    p.add_argument("--all", action="store_true", help="score every failed run")
    p.add_argument("--backend", choices=("auto", "claude", "stub"), default="auto")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    truth = loader.load_truth()
    if not truth:
        print("ERROR: no ground truth — run mini-project/generate_logs.py first.",
              file=sys.stderr)
        sys.exit(1)

    backend = args.backend
    if backend == "auto":
        backend = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "stub"
    print(f"Phase 4 — LLM RCA   backend={backend}   model={args.model if backend == 'claude' else '(stub)'}")

    n = None if args.all else args.n
    targets = collect_failed_runs(truth, n)
    if not targets:
        print("No FAILED runs in ground truth.", file=sys.stderr)
        sys.exit(1)
    target_keys = {(dag, run_id) for dag, run_id, _ in targets}

    print(f"  selected {len(targets)} failed runs for RCA")
    templates, hist, tail = mine_templates_and_collect_lines(target_keys)

    # Load response cache
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    console = Console()
    results: list[dict] = []
    total_usage = {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0}
    n_cache_hits = 0
    for dag_id, run_id, mode in targets:
        key = (dag_id, run_id)
        h = hist[key]
        top_templates = [(cid, templates.get(cid, "")) for cid, _ in h.most_common(5)]
        log_tail = tail.get(key, [])

        ck = cache_key(dag_id, mode, [cid for cid, _ in top_templates])
        if ck in cache:
            text = cache[ck]
            n_cache_hits += 1
        else:
            user_prompt = build_user_prompt(
                dag_id=dag_id, run_id=run_id,
                failure_mode_predicted=mode,
                top_templates=top_templates,
                log_tail=log_tail,
            )
            if backend == "claude":
                text, usage = call_claude(user_prompt, args.model)
                total_usage["input"] += usage["input_tokens"]
                total_usage["output"] += usage["output_tokens"]
                total_usage["cache_read"] += usage["cache_read_input_tokens"]
                total_usage["cache_create"] += usage["cache_creation_input_tokens"]
            else:
                text = stub_response(mode, dag_id)
            cache[ck] = text

        results.append({
            "dag_id": dag_id, "run_id": run_id, "failure_mode": mode,
            "rca": text, "top_templates": top_templates,
        })
        console.print(f"[cyan]{dag_id}[/cyan] / [dim]{run_id[-20:]}[/dim] "
                      f"-> `{mode}` ({len(text)} chars)")

    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    # Markdown report
    lines: list[str] = []
    lines.append("# Phase 4 — LLM RCA summaries (synthetic Airflow logs)")
    lines.append("")
    lines.append(f"**Generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(f"**Backend:** `{backend}`  |  **Model:** `{args.model if backend == 'claude' else '(stub)'}`")
    lines.append(f"**Failed runs sampled:** {len(targets)} "
                 f"(of {sum(1 for t in truth.values() if t['outcome']=='FAILED')} total)")
    lines.append("")
    if backend == "claude":
        lines.append(
            f"**Token usage:** input={total_usage['input']:,}, "
            f"output={total_usage['output']:,}, "
            f"cache_read={total_usage['cache_read']:,}, "
            f"cache_create={total_usage['cache_create']:,}, "
            f"cache_hits={n_cache_hits}/{len(targets)}"
        )
        lines.append("")
    elif backend == "stub":
        lines.append("> **Stub mode** — no LLM call was made. Set `ANTHROPIC_API_KEY` "
                     "and re-run with `--backend claude` for real RCA summaries.")
        lines.append("")

    for r in results:
        lines.append(f"## `{r['dag_id']}` — `{r['run_id'][-20:]}` ({r['failure_mode']})")
        lines.append("")
        lines.append(r["rca"])
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")
    print(f"Response cache: {CACHE_PATH} ({len(cache)} entries)")


if __name__ == "__main__":
    main()
