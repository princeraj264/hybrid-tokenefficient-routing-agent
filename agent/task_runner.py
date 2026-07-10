#!/usr/bin/env python3
"""
Hybrid Token-Efficient Routing Agent — Task Runner
===================================================

Standalone batch processor for the AMD Developer Hackathon Act II (2026).

Reads a JSON array of tasks from a file, processes each through a three-tier
routing pipeline (cache → local Gemma inference → remote Fireworks escalation),
and writes results to an output file.

Usage:
    export TASKS_INPUT_PATH=/input/tasks.json
    export RESULTS_OUTPUT_PATH=/output/results.json
    export CONFIDENCE_THRESHOLD=0.75
    export FIREWORKS_API_KEY=fw_...
    export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
    export ALLOWED_MODELS=accounts/fireworks/models/gemma-2b,accounts/fireworks/models/qwen3.5-plus
    python task_runner.py

No server, no ports, no interactive input — runs once top-to-bottom and exits.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("task_runner")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    task_type: str  # factual_qa, math, sentiment, summarization, ner, code_debugging, logic, code_generation
    prompt: str

@dataclass
class TaskResult:
    task_id: str
    answer: str
    path: str          # "cache", "local", or "remote"
    model_used: str | None = None
    tokens_used: int | None = None
    confidence: float | None = None
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    tasks_input_path: str = field(
        default_factory=lambda: os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
    )
    results_output_path: str = field(
        default_factory=lambda: os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
    )
    fireworks_api_key: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_API_KEY", "")
    )
    fireworks_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    allowed_models: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        raw = os.environ.get("ALLOWED_MODELS", "")
        if raw:
            self.allowed_models = [m.strip() for m in raw.split(",") if m.strip()]

        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            log.warning(
                "CONFIDENCE_THRESHOLD %.2f is outside [0, 1]; clamping to 0.75",
                self.confidence_threshold,
            )
            self.confidence_threshold = 0.75


# ---------------------------------------------------------------------------
# In-memory semantic cache (exact-match for this standalone runner)
# ---------------------------------------------------------------------------

class TaskCache:
    """Simple exact-match cache mapping prompt -> (answer, tokens_used)."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, int]] = {}

    def get(self, prompt: str) -> tuple[str, int] | None:
        return self._store.get(prompt)

    def put(self, prompt: str, answer: str, tokens_used: int) -> None:
        self._store[prompt] = (answer, tokens_used)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Local model (Gemma 2B via llama-cpp-python with logprobs)
# ---------------------------------------------------------------------------

class LocalModel:
    """
    Wraps llama-cpp-python for Gemma 2B inference with log-probability scoring.

    Designed to run via ROCm on an AMD Instinct™ GPU when deployed to the
    AMD Developer Cloud. During local development it runs on CPU.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model = None
        self._model_path = (
            model_path
            or os.environ.get("GEMMA_MODEL_PATH")
            or "/models/gemma-2b-it-Q4_K_M.gguf"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from llama_cpp import Llama

            log.info("Loading local model from %s ...", self._model_path)
            t0 = time.perf_counter()
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                logits_all=True,      # enable per-token logprobs
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            log.info("Local model loaded in %.2fs", elapsed)
        except ImportError:
            log.error(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )
            raise
        except FileNotFoundError:
            log.warning(
                "Model file %s not found. Local inference will be unavailable.",
                self._model_path,
            )
            self._model = None

    def generate(self, prompt: str, max_tokens: int = 256) -> tuple[str, float, int]:
        """
        Generate an answer using the local model.

        Returns:
            (answer_text, mean_log_probability, tokens_used)
        """
        self._load()
        if self._model is None:
            log.error("No local model loaded; falling back semantics.")
            return (
                "[Local model unavailable — check GEMMA_MODEL_PATH]",
                0.0,
                0,
            )

        t0 = time.perf_counter()
        output = self._model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            logprobs=True,          # request per-token log probabilities
            echo=False,
        )
        elapsed = time.perf_counter() - t0

        # --- Extract the generated text ---
        choices = output.get("choices", [])
        if not choices:
            return ("", 0.0, 0)

        choice = choices[0]
        text: str = choice.get("text", "") or choice.get("content", "")

        # --- Extract log probabilities ---
        logprobs_data = choice.get("logprobs")
        mean_confidence = 0.0
        token_count = 0

        if logprobs_data and "token_logprobs" in logprobs_data:
            token_logprobs: list[float | None] = logprobs_data["token_logprobs"]
            valid_probs = [lp for lp in token_logprobs if lp is not None]
            if valid_probs:
                # Convert cumulative log-probability to a mean probability per token.
                # logprobs are the *log* of the probability of the chosen token.
                # We exponentiate and average for an interpretable [0, 1] confidence.
                mean_confidence = sum(
                    lp for lp in valid_probs
                ) / len(valid_probs)
                # Convert from log-space so e.g. -0.05 -> 0.951
                mean_confidence = max(0.0, min(1.0, 2.71828 ** mean_confidence))

            token_count = len(token_logprobs)

        tokens_used = output.get("usage", {}).get("completion_tokens", token_count)

        log.info(
            "Local model: %d tokens, mean confidence=%.4f, latency=%.1fms",
            tokens_used,
            mean_confidence,
            elapsed * 1000,
        )

        return text.strip(), mean_confidence, tokens_used


# ---------------------------------------------------------------------------
# Remote model (Fireworks AI API)
# ---------------------------------------------------------------------------

class RemoteModel:
    """
    Calls the Fireworks AI inference API (OpenAI-compatible).

    Picks the first model from ALLOWED_MODELS (treated as cheapest-first).
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def generate(
        self, prompt: str, max_tokens: int = 512
    ) -> tuple[str, str, int]:
        """
        Generate an answer via the Fireworks API.

        Returns:
            (answer_text, model_name, tokens_used)
        """
        if not self._config.fireworks_api_key:
            log.error("FIREWORKS_API_KEY is not set — remote escalation unavailable.")
            return (
                "[Remote model unavailable — no API key]",
                "none",
                0,
            )

        if not self._config.allowed_models:
            log.error("ALLOWED_MODELS is empty — remote escalation unavailable.")
            return (
                "[Remote model unavailable — no allowed models configured]",
                "none",
                0,
            )

        import requests

        model = self._config.allowed_models[0]  # cheapest-first ordering
        url = f"{self._config.fireworks_base_url.rstrip('/')}/chat/completions"

        log.info("Escalating to remote model: %s", model)

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._config.fireworks_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=120,
            )
            resp.raise_for_status()
            elapsed = time.perf_counter() - t0
            data = resp.json()

            choice = data["choices"][0]
            text = choice["message"]["content"].strip()
            usage = data.get("usage", {})
            tokens_used = usage.get("completion_tokens", 0)

            log.info(
                "Remote model %s: %d tokens, latency=%.1fms",
                model,
                tokens_used,
                elapsed * 1000,
            )

            return text, model, tokens_used

        except Exception as exc:
            log.error("Fireworks API call failed: %s", exc)
            return (
                f"[Remote call failed: {exc}]",
                model,
                0,
            )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_tasks(path: str) -> list[Task]:
    """Load and validate the task list from a JSON file."""
    p = Path(path)
    if not p.is_file():
        log.error("Tasks file not found: %s", path)
        return []

    raw = json.loads(p.read_text("utf-8"))
    if not isinstance(raw, list):
        log.error("Expected a JSON array at %s, got %s", path, type(raw).__name__)
        return []

    valid_types = {
        "factual_qa", "math", "sentiment", "summarization",
        "ner", "code_debugging", "logic", "code_generation",
    }

    tasks: list[Task] = []
    for i, item in enumerate(raw):
        tid = item.get("task_id", f"task_{i}")
        ttype = item.get("task_type", "unknown")
        prompt = item.get("prompt", "")

        if ttype not in valid_types:
            log.warning("Task %s: unknown task_type '%s'; proceeding anyway", tid, ttype)

        if not prompt.strip():
            log.warning("Task %s: empty prompt; skipping", tid)
            continue

        tasks.append(Task(task_id=tid, task_type=ttype, prompt=prompt.strip()))

    log.info("Loaded %d tasks from %s", len(tasks), path)
    return tasks


def write_results(results: list[TaskResult], path: str) -> None:
    """Write the results array to a JSON file, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for r in results:
        d = {
            "task_id": r.task_id,
            "answer": r.answer,
            "path": r.path,
        }
        if r.model_used is not None:
            d["model_used"] = r.model_used
        if r.tokens_used is not None:
            d["tokens_used"] = r.tokens_used
        serializable.append(d)

    p.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), "utf-8")
    log.info("Wrote %d results to %s", len(results), path)


def process_tasks(config: Config) -> int:
    """
    Run the full routing pipeline over all input tasks.

    Returns exit code (0 = success, 1 = error).
    """
    tasks = load_tasks(config.tasks_input_path)
    if not tasks:
        log.info("No tasks to process.")
        return 0

    cache = TaskCache()
    local_model = LocalModel()
    remote_model = RemoteModel(config)

    results: list[TaskResult] = []

    for task in tasks:
        log.info("Processing task %s (%s) ...", task.task_id, task.task_type)

        # ---- Tier 1: Cache ----
        cached = cache.get(task.prompt)
        if cached is not None:
            answer, tokens_saved = cached
            log.info("  → Cache HIT (saved %d tokens)", tokens_saved)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    answer=answer,
                    path="cache",
                    tokens_used=0,
                )
            )
            continue

        # ---- Tier 2: Local inference ----
        t_start = time.perf_counter()
        answer, confidence, tokens_used = local_model.generate(task.prompt)
        latency = (time.perf_counter() - t_start) * 1000

        if confidence >= config.confidence_threshold:
            log.info(
                "  → Local (confidence=%.4f ≥ %.2f) — accepted",
                confidence,
                config.confidence_threshold,
            )
            cache.put(task.prompt, answer, tokens_used)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    answer=answer,
                    path="local",
                    tokens_used=tokens_used,
                    confidence=round(confidence, 4),
                    latency_ms=round(latency, 1),
                )
            )
            continue

        # ---- Tier 3: Remote escalation ----
        log.info(
            "  → Local confidence=%.4f < %.2f — escalating to remote",
            confidence,
            config.confidence_threshold,
        )
        remote_answer, model_name, remote_tokens = remote_model.generate(task.prompt)

        # Cache the remote answer too so repeat queries hit cache
        cache.put(task.prompt, remote_answer, remote_tokens)

        results.append(
            TaskResult(
                task_id=task.task_id,
                answer=remote_answer,
                path="remote",
                model_used=model_name,
                tokens_used=remote_tokens,
                confidence=round(confidence, 4),
                latency_ms=round(latency, 1),
            )
        )

    write_results(results, config.results_output_path)

    # Print a brief summary to stderr
    paths = [r.path for r in results]
    cache_hits = paths.count("cache")
    local_hits = paths.count("local")
    remote_hits = paths.count("remote")
    log.info(
        "Summary: %d total — %d cache, %d local, %d remote",
        len(results),
        cache_hits,
        local_hits,
        remote_hits,
    )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        config = Config()
        log.info("Configuration:")
        log.info("  TASKS_INPUT_PATH    = %s", config.tasks_input_path)
        log.info("  RESULTS_OUTPUT_PATH = %s", config.results_output_path)
        log.info("  CONFIDENCE_THRESHOLD= %s", config.confidence_threshold)
        log.info("  FIREWORKS_BASE_URL  = %s", config.fireworks_base_url)
        log.info("  ALLOWED_MODELS      = %s", config.allowed_models)
        log.info("  FIREWORKS_API_KEY   = %s", "***" if config.fireworks_api_key else "(not set)")

        return process_tasks(config)

    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())