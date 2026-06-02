"""Phase 4 pipeline — LLM RCA summaries (synthetic dataset).

For a sample of FAILED runs, build the per-run RCA context, call the
chosen LLM backend, persist a response cache, write a Markdown report.

CLI entrypoint: ``log-rca-phase4``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from log_rca.config import load_settings
from log_rca.datasets import SyntheticAirflowLoader
from log_rca.ml import (
    ClaudeBackend,
    LLMRcaSummarizer,
    RcaRequest,
    StubBackend,
    TemplateMiner,
)
from log_rca.ml.llm_rca import LLMUsage
from log_rca.reports import write_phase4_report


def _pick_backend(name: str, model: str):
    if name == "stub":
        return StubBackend()
    if name == "claude":
        return ClaudeBackend(model=model)
    # auto
    return ClaudeBackend(model=model) if os.environ.get("ANTHROPIC_API_KEY") else StubBackend()


def _collect_failed_runs(
    truth: dict, n: int | None,
) -> list[tuple[str, str, str]]:
    pool = sorted(
        (t.dag_id, rid, t.failure_mode)
        for rid, t in truth.items() if t.outcome == "FAILED"
    )
    if n is None or n >= len(pool):
        return pool
    rng = random.Random(42)
    return sorted(rng.sample(pool, n))


def _build_rca_requests(
    loader: SyntheticAirflowLoader,
    targets: list[tuple[str, str, str]],
    max_tail: int = 50,
) -> list[RcaRequest]:
    target_keys = {(d, r) for d, r, _ in targets}
    mode_by_run = {r: m for _, r, m in targets}

    miner = TemplateMiner()
    hist: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    tail: dict[tuple[str, str], list[str]] = defaultdict(list)

    for rec in loader.load_records():
        msg = rec.message.strip()
        if not msg:
            continue
        mined = miner.add(msg)
        key = (rec.dag_id, rec.run_id)
        if key in target_keys:
            hist[key][mined.cluster_id] += 1
            tail[key].append(rec.message)
            if len(tail[key]) > max_tail:
                tail[key].pop(0)

    templates = miner.templates()
    out: list[RcaRequest] = []
    for dag_id, run_id, mode in targets:
        h = hist[(dag_id, run_id)]
        top_templates = [(cid, templates.get(cid, "")) for cid, _ in h.most_common(5)]
        out.append(RcaRequest(
            dag_id=dag_id, run_id=run_id,
            failure_mode=mode_by_run.get(run_id, mode),
            top_templates=top_templates,
            log_tail=tail.get((dag_id, run_id), []),
        ))
    return out


def run(
    *,
    bucket_root: Path,
    report_path: Path,
    backend_name: str = "auto",
    model: str = "claude-sonnet-4-0",
    n: int | None = 5,
    cache_path: Path | None = None,
) -> dict:
    loader = SyntheticAirflowLoader(bucket_root)
    truth = loader.load_truth()
    if not truth:
        raise FileNotFoundError(
            f"No ground truth at {loader.truth_path}. Run `log-rca-gen` first."
        )

    targets = _collect_failed_runs(truth, n)
    if not targets:
        raise ValueError("No FAILED runs to summarise.")

    requests = _build_rca_requests(loader, targets)
    backend = _pick_backend(backend_name, model)
    summarizer = LLMRcaSummarizer(backend, cache_path=cache_path)

    results = []
    total = LLMUsage()
    cache_hits = 0
    for req in requests:
        r = summarizer.summarise(req)
        results.append(r)
        if r.cache_hit:
            cache_hits += 1
        total = LLMUsage(
            input_tokens=total.input_tokens + r.usage.input_tokens,
            output_tokens=total.output_tokens + r.usage.output_tokens,
            cache_read_input_tokens=total.cache_read_input_tokens
                + r.usage.cache_read_input_tokens,
            cache_creation_input_tokens=total.cache_creation_input_tokens
                + r.usage.cache_creation_input_tokens,
        )
    summarizer.persist_cache()

    write_phase4_report(
        output_path=report_path,
        backend_name=summarizer.backend_name,
        model=model,
        total_failed=sum(1 for t in truth.values() if t.outcome == "FAILED"),
        results=results,
        total_usage=total,
        cache_hits=cache_hits,
    )

    return {
        "n_results": len(results),
        "cache_hits": cache_hits,
        "backend": summarizer.backend_name,
        "input_tokens": total.input_tokens,
        "output_tokens": total.output_tokens,
        "report_path": str(report_path),
    }


# ─── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4 — LLM RCA summaries"
    )
    p.add_argument("--bucket-root", type=Path)
    p.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase4_llm_rca.md"),
    )
    p.add_argument("--cache",
                   type=Path,
                   default=Path("reports/phase4_cache.json"),
                   help="response cache path")
    p.add_argument("--n", type=int, default=5,
                   help="sample N failed runs (default 5)")
    p.add_argument("--all", action="store_true", help="score every failed run")
    p.add_argument(
        "--backend", choices=("auto", "claude", "stub"), default="auto",
    )
    p.add_argument("--model", default=None,
                   help="override LLM model (default from settings.yaml)")
    p.add_argument("--config", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings(args.config)
    bucket = args.bucket_root or settings.storage.bucket_root
    model = args.model or settings.llm.model

    print(f"Phase 4 -- bucket={bucket} -> report={args.report}")
    try:
        stats = run(
            bucket_root=bucket,
            report_path=args.report,
            backend_name=args.backend,
            model=model,
            n=None if args.all else args.n,
            cache_path=args.cache,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(
        f"  backend={stats['backend']} | results={stats['n_results']} | "
        f"cache_hits={stats['cache_hits']} | "
        f"input_tokens={stats['input_tokens']} | "
        f"output_tokens={stats['output_tokens']}"
    )
    print(f"  report: {stats['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
