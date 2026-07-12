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
    export ALLOWED_MODELS=accounts/fireworks/models/gemma-2-9b-it,accounts/fireworks/models/qwen3.7-plus
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

from solvers import (
    try_solve_math,
    extract_code,
    extract_test_cases,
    verify_code,
    verify_ner_answer,
    classify_task,
    VALID_TASK_TYPES,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("task_runner")

class LocalModelUnavailableError(RuntimeError):
    """Raised when LocalModel.generate() is called without an available model file."""

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
    deadline_fallback: bool = False


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
    remote_escalation_threshold: float = field(
        default_factory=lambda: float(os.environ.get("REMOTE_ESCALATION_THRESHOLD", "0.5"))
    )
    strict_output_schema: bool = field(
        default_factory=lambda: os.environ.get("STRICT_OUTPUT_SCHEMA", "false").lower()
        in ("1", "true", "yes")
    )
    max_runtime_seconds: float = field(
        default_factory=lambda: float(os.environ.get("MAX_RUNTIME_SECONDS", "420"))
    )

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

        if self.remote_escalation_threshold < 0.0 or self.remote_escalation_threshold > 1.0:
            log.warning(
                "REMOTE_ESCALATION_THRESHOLD %.2f is outside [0, 1]; clamping to 0.5",
                self.remote_escalation_threshold,
            )
            self.remote_escalation_threshold = 0.5


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

    @property
    def is_available(self) -> bool:
        """Check whether the model file exists on disk without loading the model."""
        if self._model is not None:
            return True
        p = Path(self._model_path)
        return p.is_file() and p.stat().st_size > 0

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
            raise LocalModelUnavailableError(
                f"Local model file '{self._model_path}' is not available. "
                "Check is_available before calling generate()."
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
        self, prompt: str, max_tokens: int = 512, model_index: int = 0
    ) -> tuple[str, str, int, float | None]:
        """
        Generate an answer via the Fireworks API using the model at the given index.

        Requests logprobs so callers can compute per-tier confidence scores.
        This allows the escalation chain (economy → premium) to decide
        whether a cheap model's output is trustworthy.

        Returns:
            (answer_text, model_name, tokens_used, confidence_or_none)
            confidence is None if logprobs were unavailable from the API.
        """
        if not self._config.fireworks_api_key:
            log.error("FIREWORKS_API_KEY is not set — remote escalation unavailable.")
            return ("[Remote model unavailable — no API key]", "none", 0, None)

        if not self._config.allowed_models:
            log.error("ALLOWED_MODELS is empty — remote escalation unavailable.")
            return ("[Remote model unavailable — no allowed models configured]", "none", 0, None)

        if model_index >= len(self._config.allowed_models):
            log.error(
                "model_index %d out of range (allowed_models has %d entries)",
                model_index,
                len(self._config.allowed_models),
            )
            return ("[Remote model unavailable — index out of range]", "none", 0, None)

        import requests

        model = self._config.allowed_models[model_index]
        url = f"{self._config.fireworks_base_url.rstrip('/')}/chat/completions"

        log.info("Remote tier %d — model: %s", model_index, model)

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
                    "logprobs": True,  # request per-token logprobs for confidence scoring
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

            # --- Extract logprobs-based confidence ---
            confidence: float | None = None
            logprobs_data = choice.get("logprobs")
            if logprobs_data and "content" in logprobs_data:
                token_logprobs = [
                    entry["logprob"]
                    for entry in logprobs_data["content"]
                    if entry.get("logprob") is not None
                ]
                if token_logprobs:
                    mean_lp = sum(token_logprobs) / len(token_logprobs)
                    confidence = max(0.0, min(1.0, 2.71828 ** mean_lp))

            log.info(
                "Remote model %s: %d tokens, confidence=%s, latency=%.1fms",
                model,
                tokens_used,
                f"{confidence:.4f}" if confidence is not None else "N/A",
                elapsed * 1000,
            )

            return text, model, tokens_used, confidence

        except Exception as exc:
            log.error("Fireworks API call failed: %s", exc)
            return (f"[Remote call failed: {exc}]", model, 0, None)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_tasks(path: str) -> list[Task]:
    """Load and validate the task list from a JSON file.

    Returns an empty list for a legitimate empty JSON array (``[]``).
    Raises an exception if the file doesn't exist, isn't valid JSON, or
    isn't a JSON array — the caller should propagate this to ``main()``
    so it logs the error and exits with code 1.
    """
    p = Path(path)
    if not p.is_file():
        log.error("Tasks file not found: %s", path)
        raise FileNotFoundError(f"Tasks file not found: {path}")

    try:
        raw = json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.error("Invalid content in tasks file %s: %s", path, e)
        raise ValueError(f"Invalid content in tasks file {path}: {e}") from e

    if not isinstance(raw, list):
        log.error("Expected a JSON array at %s, got %s", path, type(raw).__name__)
        raise TypeError(f"Expected a JSON array at {path}, got {type(raw).__name__}")

    tasks: list[Task] = []
    for i, item in enumerate(raw):
        tid = item.get("task_id", f"task_{i}")
        prompt = item.get("prompt", "")

        if not prompt.strip():
            log.warning("Task %s: empty prompt; skipping", tid)
            continue

        # Use explicit task_type if valid, otherwise infer from prompt
        ttype = item.get("task_type", None)
        if ttype not in VALID_TASK_TYPES:
            inferred = classify_task(prompt)
            if ttype is not None and ttype not in VALID_TASK_TYPES:
                log.info(
                    "Task %s: unknown task_type %r from JSON — inferred %r from prompt",
                    tid, ttype, inferred,
                )
            else:
                log.info(
                    "Task %s: no task_type in JSON — inferred %r from prompt",
                    tid, inferred,
                )
            ttype = inferred

        tasks.append(Task(task_id=tid, task_type=ttype, prompt=prompt.strip()))

    log.info("Loaded %d tasks from %s", len(tasks), path)
    return tasks


def write_results(results: list[TaskResult], path: str, strict: bool = False) -> None:
    """Write the results array to a JSON file, creating parent dirs if needed.

    When *strict* is True, each entry emits only task_id and answer (the
    minimal AMD-hackathon-required schema). The default (strict=False) includes
    all available fields (path, model_used, tokens_used, confidence, latency_ms).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for r in results:
        if strict:
            d: dict[str, object] = {"task_id": r.task_id, "answer": r.answer}
        else:
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


def run_pipeline(config: Config) -> tuple[list[TaskResult], list[Task]]:
    """
    Run the full routing pipeline over all input tasks.

    Returns (results, tasks) so callers can inspect path distribution,
    token usage, and per-type breakdown without modifying the pipeline.

    This is the core that both process_tasks() and benchmark.py reuse.
    """
    tasks = load_tasks(config.tasks_input_path)
    if not tasks:
        log.info("No tasks to process.")
        return [], []

    cache = TaskCache()
    local_model = LocalModel()
    local_available = local_model.is_available
    if not local_available:
        log.warning("=" * 70)
        log.warning(
            "LOCAL MODEL UNAVAILABLE: '%s' not found or empty on disk. "
            "All tasks will be routed directly to Fireworks (remote escalation).",
            local_model._model_path,
        )
        log.warning("=" * 70)
    remote_model = RemoteModel(config)

    results: list[TaskResult] = []

    pipeline_start = time.perf_counter()
    max_runtime = config.max_runtime_seconds
    deadline_hit = False

    for task in tasks:
        # Check wall-clock deadline — once hit, skip all remote escalation
        if not deadline_hit:
            elapsed = time.perf_counter() - pipeline_start
            if elapsed > max_runtime:
                deadline_hit = True
                remaining = len(tasks) - len(results)
                log.warning(
                    "RUN DEADLINE REACHED: %.1fs elapsed (limit %.0fs). "
                    "Skipping remote escalation for %d remaining task(s) — "
                    "falling back to best-effort local answers.",
                    elapsed, max_runtime, remaining,
                )

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

        # ---- Deadline check: skip ALL further processing (including local
        # model inference) once the wall-clock budget is exhausted. ----
        if deadline_hit:
            log.warning(
                "  → Deadline exceeded → skipping ALL inference for task %s, "
                "returning fallback answer",
                task.task_id,
            )
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    answer="[Deadline exceeded before this task could be processed - no inference attempted]",
                    path="deadline_skipped",
                    tokens_used=0,
                    confidence=0.0,
                    latency_ms=0.0,
                )
            )
            continue

        # ---- Tier 1b: Math solver (deterministic, zero-cost) ----
        if task.task_type == "math":
            math_answer = try_solve_math(task.prompt)
            if math_answer is not None:
                log.info("  → Math solved deterministically (0 tokens)")
                cache.put(task.prompt, math_answer, 0)
                results.append(
                    TaskResult(
                        task_id=task.task_id,
                        answer=math_answer,
                        path="local",
                        tokens_used=0,
                        confidence=1.0,
                    )
                )
                continue
            # If try_solve_math returned None, fall through to local inference

        # ---- Tier 2: Code verification pipeline ----
        if task.task_type in ("code_debugging", "code_generation"):
            if not local_available:
                log.info("  → Local model unavailable — escalating directly to remote")
                answer = ""
                confidence = 0.0
                tokens_used = 0
                latency = 0.0
            else:
                t_start = time.perf_counter()
                answer, confidence, tokens_used = local_model.generate(task.prompt)
                latency = (time.perf_counter() - t_start) * 1000

            code = extract_code(answer)
            test_cases = extract_test_cases(task.prompt)

            err = ""
            if code is not None:
                verified, err = verify_code(code, test_cases=test_cases)
                if verified:
                    log.info(
                        "  → Code verified locally (test cases: %d)",
                        len(test_cases),
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
            else:
                err = "No code block found in local output"

            # Local code missing or failed verification
            if deadline_hit:
                log.info(
                    "  → Deadline reached — accepting local answer "
                    "without remote escalation for task %s",
                    task.task_id,
                )
                fallback_answer = (
                    answer.strip()
                    if answer.strip()
                    else "[Deadline reached — best-effort answer, no local output available]"
                )
                cache.put(task.prompt, fallback_answer, tokens_used)
                results.append(
                    TaskResult(
                        task_id=task.task_id,
                        answer=fallback_answer,
                        path="local",
                        tokens_used=tokens_used,
                        confidence=round(confidence, 4),
                        latency_ms=round(latency, 1),
                        deadline_fallback=True,
                    )
                )
                continue

            log.info(
                "  → Local code %s — escalating to remote",
                "missing" if code is None else "failed verification",
            )
            repair_prompt = (
                task.prompt
                + "\n\n[Note: the previous attempt failed.\nError: "
                + err
                + "\nPlease fix the code and respond with a corrected ```python block.]"
            )

            # Try economy model first, then premium if verification fails
            max_remote_calls = min(2, len(config.allowed_models))
            remote_answer: str = ""
            model_name: str = "none"
            remote_tokens: int = 0
            remote_confidence: float | None = None

            for tier_index in range(max_remote_calls):
                (
                    remote_answer,
                    model_name,
                    remote_tokens,
                    remote_confidence,
                ) = remote_model.generate(repair_prompt, model_index=tier_index)
                log.info(
                    "  → Remote tier %d (%s): %d tokens",
                    tier_index,
                    model_name,
                    remote_tokens,
                )

                # Verify the remote answer
                remote_code = extract_code(remote_answer)
                if remote_code is not None:
                    remote_verified, remote_err = verify_code(
                        remote_code, test_cases=test_cases
                    )
                    if remote_verified:
                        log.info(
                            "  → Code verified at remote tier %d (test cases: %d)",
                            tier_index,
                            len(test_cases),
                        )
                        break  # Accept this tier's answer
                    else:
                        log.warning(
                            "  → Remote tier %d code failed verification: %s",
                            tier_index,
                            remote_err,
                        )
                else:
                    log.warning(
                        "  → Remote tier %d answer contains no code block",
                        tier_index,
                    )

                # If this was the last permitted call, accept anyway (one escalation max)
                if tier_index == max_remote_calls - 1:
                    log.warning(
                        "  → Task %s could not be verified even after %d remote call(s) — "
                        "accepting last answer",
                        task.task_id,
                        max_remote_calls,
                    )

            cache.put(task.prompt, remote_answer, remote_tokens)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    answer=remote_answer,
                    path="remote",
                    model_used=model_name,
                    tokens_used=remote_tokens,
                    confidence=round(remote_confidence, 4) if remote_confidence is not None else round(confidence, 4),
                    latency_ms=round(latency, 1),
                )
            )
            continue

        # ---- Tier 3: Local inference (confidence-based) ----
        if not local_available:
            log.info("  → Local model unavailable — escalating directly to remote")
            answer = ""
            confidence = 0.0
            tokens_used = 0
            latency = 0.0
        else:
            t_start = time.perf_counter()
            answer, confidence, tokens_used = local_model.generate(task.prompt)
            latency = (time.perf_counter() - t_start) * 1000

        if confidence >= config.confidence_threshold:
            # For NER tasks, additionally verify no blatant hallucinations
            if task.task_type == "ner" and not verify_ner_answer(task.prompt, answer):
                log.info(
                    "  → Local (confidence=%.4f ≥ %.2f) — NER hallucination "
                    "detected, escalating to remote",
                    confidence,
                    config.confidence_threshold,
                )
                # Fall through to Tier 4 (remote escalation) below
            else:
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

        # ---- Tier 4: Remote escalation (multi-tier, economy → premium) ----
        if deadline_hit:
            log.info(
                "  → Deadline reached — accepting local answer "
                "(confidence=%.4f) without remote escalation for task %s",
                confidence,
                task.task_id,
            )
            fallback_answer = (
                answer.strip()
                if answer.strip()
                else "[Deadline reached — best-effort answer, no local output available]"
            )
            cache.put(task.prompt, fallback_answer, tokens_used)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    answer=fallback_answer,
                    path="local",
                    tokens_used=tokens_used,
                    confidence=round(confidence, 4),
                    latency_ms=round(latency, 1),
                    deadline_fallback=True,
                )
            )
            continue

        t_remote_start = time.perf_counter()
        if confidence < config.confidence_threshold:
            log.info(
                "  → Local confidence=%.4f < %.2f — escalating to remote",
                confidence,
                config.confidence_threshold,
            )
        else:
            log.info(
                "  → NER hallucination fallthrough — escalating to remote",
            )

        max_remote_calls = min(2, len(config.allowed_models))
        remote_answer: str = ""
        model_name: str = "none"
        remote_tokens: int = 0
        remote_confidence: float | None = None

        for tier_index in range(max_remote_calls):
            (
                remote_answer,
                model_name,
                remote_tokens,
                remote_confidence,
            ) = remote_model.generate(task.prompt, model_index=tier_index)
            log.info(
                "  → Remote tier %d (%s): %d tokens, confidence=%s",
                tier_index,
                model_name,
                remote_tokens,
                f"{remote_confidence:.4f}" if remote_confidence is not None else "N/A",
            )

            if remote_confidence is not None and remote_confidence >= config.remote_escalation_threshold:
                # For NER tasks, verify entity grounding before accepting
                if task.task_type == "ner" and not verify_ner_answer(task.prompt, remote_answer):
                    if tier_index < max_remote_calls - 1:
                        log.warning(
                            "  → Remote tier %d (confidence=%.4f ≥ %.2f) — "
                            "NER grounding FAILED, continuing to next tier",
                            tier_index,
                            remote_confidence,
                            config.remote_escalation_threshold,
                        )
                        continue
                    else:
                        log.warning(
                            "  → Task %s: final answer (tier %d, confidence=%.4f) "
                            "failed NER grounding verification — accepting anyway",
                            task.task_id,
                            tier_index,
                            remote_confidence,
                        )
                else:
                    log.info(
                        "  → Remote tier %d accepted (confidence=%.4f ≥ %.2f)",
                        tier_index,
                        remote_confidence,
                        config.remote_escalation_threshold,
                    )
                break  # Accept this model's output

            # For code/math tasks, also run solver verification at the remote tier
            if task.task_type in ("code_debugging", "code_generation"):
                code = extract_code(remote_answer)
                if code is not None:
                    test_cases = extract_test_cases(task.prompt)
                    verified, err = verify_code(code, test_cases=test_cases)
                    if verified:
                        log.info(
                            "  → Code verified at remote tier %d (test cases: %d)",
                            tier_index,
                            len(test_cases),
                        )
                        break
                    else:
                        log.warning(
                            "  → Remote tier %d code failed verification: %s",
                            tier_index,
                            err,
                        )
                else:
                    log.warning(
                        "  → Remote tier %d answer contains no code block",
                        tier_index,
                    )
            elif task.task_type == "math":
                math_answer = try_solve_math(remote_answer)
                if math_answer is not None:
                    log.info("  → Math solved at remote tier %d", tier_index)
                    remote_answer = math_answer
                    break
                else:
                    log.warning(
                        "  → Remote tier %d answer not deterministically verifiable as math",
                        tier_index,
                    )

            # If this is the last permitted call, accept anyway
            if tier_index == max_remote_calls - 1:
                if task.task_type == "ner":
                    log.warning(
                        "  → Task %s: NER grounding failed after %d remote call(s) — "
                        "accepting last answer despite NER hallucination risk",
                        task.task_id,
                        max_remote_calls,
                    )
                else:
                    log.warning(
                        "  → Task %s: remote confidence=%.4f < %.2f after %d call(s) — "
                        "accepting anyway (no more tiers)",
                        task.task_id,
                        remote_confidence if remote_confidence is not None else 0.0,
                        config.remote_escalation_threshold,
                        max_remote_calls,
                    )

        remote_latency = (time.perf_counter() - t_remote_start) * 1000

        # Cache the remote answer too so repeat queries hit cache
        cache.put(task.prompt, remote_answer, remote_tokens)

        results.append(
            TaskResult(
                task_id=task.task_id,
                answer=remote_answer,
                path="remote",
                model_used=model_name,
                tokens_used=remote_tokens,
                confidence=round(remote_confidence, 4) if remote_confidence is not None else round(confidence, 4),
                latency_ms=round(remote_latency, 1),
            )
        )

    return results, tasks


def process_tasks(config: Config) -> int:
    """
    Run the full routing pipeline over all input tasks.

    Thin wrapper around run_pipeline() that persists results and prints
    a summary to stderr.

    Returns exit code (0 = success, 1 = error).
    """
    results, tasks = run_pipeline(config)

    # Always write results so the output file exists on any exit-code-0 path
    # (including a legitimate empty task list).  Never let load_tasks failure
    # result in a silent "no file written" on exit code 0.
    write_results(results, config.results_output_path, strict=config.strict_output_schema)

    if not results:
        return 0

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