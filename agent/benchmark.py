#!/usr/bin/env python3
"""
Hybrid Token-Efficient Routing Agent — Benchmark Runner
========================================================

Manually-run script that loads a task set, runs the full four-tier pipeline
(reusing run_pipeline() from task_runner.py), and produces:

  - A clean summary table printed to stdout
  - A machine-readable JSON summary written alongside the results file
    (same directory as RESULTS_OUTPUT_PATH), named benchmark_summary.json

Usage:
    export BENCHMARK_TASKS_PATH=/input/benchmark-tasks.json   (default: ./input/tasks.json)
    export CONFIDENCE_THRESHOLD=0.75
    export FIREWORKS_API_KEY=fw_...
    python agent/benchmark.py

The script does NOT write individual task results to a separate file — only
the aggregated summary.  Use the standard TASKS_INPUT_PATH / RESULTS_OUTPUT_PATH
env vars in task_runner.py for full result inspection.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Reuse the same pipeline and data models as the main batch runner.
from task_runner import (
    Config,
    Task,
    TaskResult,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Logging — stderr only, stdout is reserved for the table
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MATH_SOLVED_TAG = "local (math-solved)"
LOCAL_MODEL_TAG = "local (model)"


def classify_path(r: TaskResult, task_map: dict[str, Task]) -> str:
    """
    Return a more descriptive path label.

    - "cache" stays "cache"
    - If path == "local" and tokens_used == 0 and confidence == 1.0
      and the task's type is "math", it was the math solver -> math-solved
    - Everything else "local" (model-generated and accepted) -> local (model)
    - "remote" stays "remote"
    """
    if r.path == "cache":
        return "cache"
    if r.path == "remote":
        return "remote"
    # path == "local"
    task = task_map.get(r.task_id)
    if (
        task
        and task.task_type == "math"
        and r.tokens_used == 0
        and r.confidence == 1.0
    ):
        return MATH_SOLVED_TAG
    return LOCAL_MODEL_TAG


def pct(part: int, total: int) -> str:
    """Format as percentage string, or '—' if total is 0."""
    if total == 0:
        return "   —"
    return f"{100.0 * part / total:6.1f}%"


def fmt_tokens(n: int) -> str:
    """Format a token count with thousand separators."""
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_overall_table(
    total: int,
    path_counts: Counter[str],
    tokens_by_path: dict[str, int],
    avg_tokens: float,
    estimated_savings: int,
) -> None:
    """Print the overall summary table to stdout."""
    sep = "─" * 78

    print()
    print("╭" + "─" * 76 + "╮")
    print(f"│  {'Hybrid Routing Agent — Benchmark Results':^72s}  │")
    print("╰" + "─" * 76 + "╯")
    print()

    rows = [
        ("Total tasks", str(total), ""),
        ("Cache hits", str(path_counts.get("cache", 0)), pct(path_counts.get("cache", 0), total)),
        ("Local (math-solved)", str(path_counts.get(MATH_SOLVED_TAG, 0)), pct(path_counts.get(MATH_SOLVED_TAG, 0), total)),
        ("Local (model)", str(path_counts.get(LOCAL_MODEL_TAG, 0)), pct(path_counts.get(LOCAL_MODEL_TAG, 0), total)),
        ("Remote", str(path_counts.get("remote", 0)), pct(path_counts.get("remote", 0), total)),
    ]

    print(f"  {'Route':<22s} {'Count':>8s} {'%':>9s}")
    print(f"  {sep}")
    for label, count_str, pct_str in rows:
        print(f"  {label:<22s} {count_str:>8s} {pct_str:>9s}")

    print()
    print(f"  {'Total tokens used:':<30s} {fmt_tokens(sum(tokens_by_path.values())):>10s}")
    print(f"  {'Avg tokens per task:':<30s} {fmt_tokens(int(round(avg_tokens))):>10s}")
    print(f"  {'Estimated tokens saved vs. always-remote:':<30s} {fmt_tokens(estimated_savings):>10s}")
    print()


def print_type_table(
    breakdown: dict[str, dict[str, Any]],
    total_tasks: int,
) -> None:
    """Print per-task-type breakdown table to stdout."""
    sep = "─" * 78

    print("  Per-task-type breakdown:")
    print(f"  {'Task type':<22s} {'Count':>8s} {'%':>9s} {'Cache':>8s} {'Math':>8s} {'Model':>8s} {'Remote':>8s} {'Avg tok':>8s}")
    print(f"  {sep}")

    for ttype in sorted(breakdown.keys()):
        b = breakdown[ttype]
        print(
            f"  {ttype:<22s}"
            f" {b['count']:>8d}"
            f" {pct(b['count'], total_tasks):>9s}"
            f" {b['cache']:>8d}"
            f" {b['math_solved']:>8d}"
            f" {b['local_model']:>8d}"
            f" {b['remote']:>8d}"
            f" {fmt_tokens(b['avg_tokens']):>8s}"
        )
    print()


def print_aggregate_summary(summary: dict[str, Any]) -> None:
    """Print the aggregate summary to stdout."""
    overall = summary["overall"]
    sep = "─" * 68

    print()
    print("╭" + "─" * 66 + "╮")
    print(f"│  {'Aggregate Metrics Summary':^62s}  │")
    print("╰" + "─" * 66 + "╯")
    print()

    print(f"  {'Metric':<32s} {'Value':>16s}")
    print(f"  {sep}")

    lines = [
        ("Total tasks", f"{overall['total_tasks']}"),
        ("Cache hits", f"{overall['cache_hits']}"),
        ("Local answers", f"{overall['local_answers']}"),
        ("Remote escalations", f"{overall['remote_escalations']}"),
        ("Deadline fallbacks", f"{overall['deadline_fallbacks']}"),
        ("Total tokens used", f"{overall['total_tokens_used']:,}"),
        ("Avg tokens per task", f"{overall['avg_tokens_per_task']:,.1f}"),
        ("Avg latency (ms)", f"{overall['avg_latency_ms']:,.1f}"),
        ("Remote escalation rate", f"{overall['remote_escalation_rate']:.1f}%"),
    ]
    for label, value in lines:
        print(f"  {label:<32s} {value:>16s}")

    print()
    print(f"  Path distribution:")
    for path, count in overall["by_path"].items():
        pct_str = pct(count, overall["total_tasks"])
        print(f"    {path:<24s} {count:>6d} {pct_str:>9s}")
    print()


# ---------------------------------------------------------------------------
# Aggregate summary builder
# ---------------------------------------------------------------------------

def build_aggregate_summary(
    results: list[TaskResult],
    path_counts: Counter[str],
    total_tokens: int,
    avg_tokens: float,
) -> dict[str, Any]:
    """Build the aggregate summary dict from pipeline results.

    Metrics computed:
      - total_tasks
      - cache_hits, local_answers, remote_escalations (by raw ``r.path``)
      - deadline_fallbacks (count of results with ``deadline_fallback=True``)
      - total_tokens_used
      - avg_tokens_per_task
      - avg_latency_ms (mean of all non-None ``latency_ms`` values)
      - remote_escalation_rate (as a percentage)
      - by_path (raw path distribution: cache/local/remote)
    """
    total = len(results)

    cache_hits = sum(1 for r in results if r.path == "cache")
    local_answers = sum(1 for r in results if r.path == "local")
    remote_escalations = sum(1 for r in results if r.path == "remote")
    deadline_fallbacks = sum(1 for r in results if r.deadline_fallback)

    # Average latency: mean of all non-None latency_ms values
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    remote_escalation_rate = (remote_escalations / total * 100.0) if total > 0 else 0.0

    overall: dict[str, Any] = {
        "total_tasks": total,
        "cache_hits": cache_hits,
        "local_answers": local_answers,
        "remote_escalations": remote_escalations,
        "deadline_fallbacks": deadline_fallbacks,
        "by_path": {
            "cache": path_counts.get("cache", 0),
            "local_math_solved": path_counts.get(MATH_SOLVED_TAG, 0),
            "local_model": path_counts.get(LOCAL_MODEL_TAG, 0),
            "remote": path_counts.get("remote", 0),
        },
        "total_tokens_used": total_tokens,
        "avg_tokens_per_task": round(avg_tokens, 1),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "remote_escalation_rate": round(remote_escalation_rate, 1),
    }

    return {"overall": overall}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # ---- Configuration ----
    tasks_path = os.environ.get("BENCHMARK_TASKS_PATH", "/input/tasks.json")

    # Use a throwaway output path — we don't want benchmark runs to overwrite
    # the main results file.
    config = Config()
    config.tasks_input_path = tasks_path
    config.results_output_path = "/tmp/benchmark_results.json"

    log.info("Benchmark configuration:")
    log.info("  BENCHMARK_TASKS_PATH = %s", tasks_path)
    log.info("  CONFIDENCE_THRESHOLD = %s", config.confidence_threshold)
    log.info("  FIREWORKS_BASE_URL   = %s", config.fireworks_base_url)
    log.info("  ALLOWED_MODELS       = %s", config.allowed_models)
    log.info("  FIREWORKS_API_KEY    = %s", "***" if config.fireworks_api_key else "(not set)")

    # ---- Run the pipeline ----
    try:
        results, tasks = run_pipeline(config)
    except (FileNotFoundError, ValueError, TypeError) as e:
        log.error("Pipeline failed — %s", e)
        return 1

    if not results:
        log.error("No results produced — check input file and model config.")
        return 1

    # Build a task_id -> Task map for type lookups
    task_map: dict[str, Task] = {t.task_id: t for t in tasks}

    # ---- Compute overall statistics ----

    total = len(results)
    path_labels = [classify_path(r, task_map) for r in results]
    path_counts: Counter[str] = Counter(path_labels)

    tokens_by_path: dict[str, int] = defaultdict(int)
    for r, label in zip(results, path_labels):
        tokens_by_path[label] += r.tokens_used or 0

    total_tokens = sum(tokens_by_path.values())
    avg_tokens = total_tokens / total if total > 0 else 0.0

    # ---- Estimated token savings vs. "always remote" baseline ----
    remote_token_list = [
        r.tokens_used
        for r in results
        if r.path == "remote" and (r.tokens_used or 0) > 0
    ]
    avg_remote_tokens = (
        sum(remote_token_list) / len(remote_token_list)
        if remote_token_list
        else 256
    )
    hypothetical_remote_total = total * avg_remote_tokens
    estimated_savings = int(hypothetical_remote_total - total_tokens)

    # ---- Per-task-type breakdown ----
    type_breakdown: dict[str, dict[str, Any]] = {}

    for ttype in sorted({t.task_type for t in tasks}):
        matching_tasks = [t for t in tasks if t.task_type == ttype]
        matching_results = [
            (r, label)
            for r, label in zip(results, path_labels)
            if task_map[r.task_id].task_type == ttype
        ]
        count = len(matching_tasks)

        cache_c = sum(1 for _, lbl in matching_results if lbl == "cache")
        math_c = sum(1 for _, lbl in matching_results if lbl == MATH_SOLVED_TAG)
        model_c = sum(1 for _, lbl in matching_results if lbl == LOCAL_MODEL_TAG)
        remote_c = sum(1 for _, lbl in matching_results if lbl == "remote")

        type_tokens = sum(r.tokens_used or 0 for r, _ in matching_results)
        avg_type_tokens = int(round(type_tokens / count)) if count > 0 else 0

        type_breakdown[ttype] = {
            "count": count,
            "cache": cache_c,
            "math_solved": math_c,
            "local_model": model_c,
            "remote": remote_c,
            "total_tokens": type_tokens,
            "avg_tokens": avg_type_tokens,
        }

    # ---- Build the new aggregate summary ----
    agg_summary = build_aggregate_summary(results, path_counts, total_tokens, avg_tokens)

    # ---- Print tables to stdout ----
    print_overall_table(total, path_counts, tokens_by_path, avg_tokens, estimated_savings)
    print_type_table(type_breakdown, total)
    print_aggregate_summary(agg_summary)

    # ---- Write JSON summary to the same directory as results_output_path ----
    summary: dict[str, Any] = {
        "config": {
            "tasks_path": config.tasks_input_path,
            "confidence_threshold": config.confidence_threshold,
            "allowed_models": config.allowed_models,
        },
        "overall": agg_summary["overall"],
        "by_task_type": type_breakdown,
    }

    output_dir = Path(os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Benchmark summary written to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
