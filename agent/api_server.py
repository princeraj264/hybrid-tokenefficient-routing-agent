#!/usr/bin/env python3
"""
Hybrid Token-Efficient Routing Agent — API Server
==================================================

FastAPI server that wraps the three-tier routing pipeline (cache → local Gemma
inference → remote Fireworks escalation) behind HTTP endpoints. Designed to run
alongside the React frontend as the "backend" service in docker-compose.yml.

Endpoints:
    POST /query   — Run a single query through the routing pipeline
    GET  /health  — Liveness check

Usage:
    export FIREWORKS_API_KEY=fw_...
    export ALLOWED_MODELS=accounts/fireworks/models/qwen3.7-plus
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Reuse the exact same classes from the batch task runner
from task_runner import Config, TaskCache, LocalModel, RemoteModel

# Reuse the same solver functions for deterministic math, code verification, and NER sanity checks
from solvers import (
    try_solve_math,
    extract_code,
    extract_test_cases,
    verify_code,
    verify_ner_answer,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("api_server")

# ---------------------------------------------------------------------------
# App & singletons
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hybrid Routing Agent",
    version="1.0.0",
    description="Three-tier query router: cache → local Gemma 2B → remote Fireworks AI",
)

# CORS — allow the Vite dev server (localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise shared state once at import time.
_config = Config()
_cache = TaskCache()
_local_model = LocalModel()
_remote_model = RemoteModel(_config)

log.info("API server configuration:")
log.info("  CONFIDENCE_THRESHOLD = %s", _config.confidence_threshold)
log.info("  FIREWORKS_BASE_URL   = %s", _config.fireworks_base_url)
log.info("  ALLOWED_MODELS       = %s", _config.allowed_models)
log.info("  FIREWORKS_API_KEY    = %s", "***" if _config.fireworks_api_key else "(not set)")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query string")
    task_type: str = Field(
        "general",
        description=(
            "Task type for routing: math, code_debugging, code_generation, ner, "
            "factual_qa, sentiment, summarization, logic, or general (default)"
        ),
    )

class QueryResponse(BaseModel):
    answer: str
    path: str  # "cache" | "local" | "remote"
    confidence: float
    tokens_used: int
    latency_ms: float

class HealthResponse(BaseModel):
    status: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — returns {'status': 'ok'} when the server is running."""
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Run a single query through the four-tier routing pipeline.

    Tiers (in order):
        1. Cache (exact-match HIT → return instantly)
        1b. Math solver (task_type="math" → try_solve_math)
        2. Code verification (code_debugging / code_generation → local infer +
           extract_code + extract_test_cases + verify_code; escalate on failure)
        3. Local inference (confidence-based; NER tasks additionally run
           verify_ner_answer to catch hallucinations)
        4. Remote escalation (Fireworks API)

    Returns the answer along with metadata about which tier handled the request,
    the confidence score, token usage, and latency.
    """
    prompt = req.query.strip()
    task_type = req.task_type.lower()
    start = time.perf_counter()

    # ---- Tier 1: Cache ----
    cached = _cache.get(prompt)
    if cached is not None:
        answer, _tokens_saved = cached
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("  → Cache HIT (%.1fms)", elapsed_ms)
        return QueryResponse(
            answer=answer,
            path="cache",
            confidence=1.0,
            tokens_used=0,
            latency_ms=round(elapsed_ms, 1),
        )

    # ---- Tier 1b: Math solver (deterministic, zero-cost) ----
    if task_type == "math":
        math_answer = try_solve_math(prompt)
        if math_answer is not None:
            log.info("  → Math solved deterministically (0 tokens)")
            _cache.put(prompt, math_answer, 0)
            total_ms = (time.perf_counter() - start) * 1000
            return QueryResponse(
                answer=math_answer,
                path="local",
                confidence=1.0,
                tokens_used=0,
                latency_ms=round(total_ms, 1),
            )
        # Fall through to local inference if math solver returns None

    # ---- Tier 2: Code verification pipeline ----
    if task_type in ("code_debugging", "code_generation"):
        t0 = time.perf_counter()
        answer, confidence, tokens_used = _local_model.generate(prompt)
        latency = (time.perf_counter() - t0) * 1000

        code = extract_code(answer)
        test_cases = extract_test_cases(prompt)

        err = ""
        if code is not None:
            verified, err = verify_code(code, test_cases=test_cases)
            if verified:
                log.info(
                    "  → Code verified locally (test cases: %d)",
                    len(test_cases),
                )
                _cache.put(prompt, answer, tokens_used)
                total_ms = (time.perf_counter() - start) * 1000
                return QueryResponse(
                    answer=answer,
                    path="local",
                    confidence=round(confidence, 4),
                    tokens_used=tokens_used,
                    latency_ms=round(total_ms, 1),
                )
        else:
            err = "No code block found in local output"

        # Local code missing or failed verification — escalate to remote
        log.info(
            "  → Local code %s — escalating to remote",
            "missing" if code is None else "failed verification",
        )
        repair_prompt = (
            prompt
            + "\n\n[Note: the previous attempt failed.\nError: "
            + err
            + "\nPlease fix the code and respond with a corrected ```python block.]"
        )
        remote_answer, model_name, remote_tokens = _remote_model.generate(
            repair_prompt
        )

        # Verify the remote answer too (one escalation max)
        remote_code = extract_code(remote_answer)
        if remote_code is not None:
            remote_verified, remote_err = verify_code(
                remote_code, test_cases=test_cases
            )
            if not remote_verified:
                log.warning(
                    "  → Remote code also failed verification: %s",
                    remote_err,
                )
        else:
            log.warning(
                "  → Remote answer contains no code block",
            )

        _cache.put(prompt, remote_answer, remote_tokens)
        total_ms = (time.perf_counter() - start) * 1000
        return QueryResponse(
            answer=remote_answer,
            path="remote",
            confidence=round(confidence, 4),
            tokens_used=remote_tokens,
            latency_ms=round(total_ms, 1),
        )

    # ---- Tier 3: Local inference (confidence-based) ----
    t0 = time.perf_counter()
    answer, confidence, tokens_used = _local_model.generate(prompt)
    latency = (time.perf_counter() - t0) * 1000

    if confidence >= _config.confidence_threshold:
        # For NER tasks, additionally verify no blatant hallucinations
        if task_type == "ner" and not verify_ner_answer(prompt, answer):
            log.info(
                "  → Local (confidence=%.4f ≥ %.2f) — NER hallucination "
                "detected, escalating to remote",
                confidence,
                _config.confidence_threshold,
            )
            # Fall through to Tier 4 (remote escalation) below
        else:
            log.info(
                "  → Local (confidence=%.4f ≥ %.2f) — accepted",
                confidence,
                _config.confidence_threshold,
            )
            _cache.put(prompt, answer, tokens_used)
            total_ms = (time.perf_counter() - start) * 1000
            return QueryResponse(
                answer=answer,
                path="local",
                confidence=round(confidence, 4),
                tokens_used=tokens_used,
                latency_ms=round(total_ms, 1),
            )

    # ---- Tier 4: Remote escalation ----
    if confidence < _config.confidence_threshold:
        log.info(
            "  → Local confidence=%.4f < %.2f — escalating to remote",
            confidence,
            _config.confidence_threshold,
        )
    else:
        log.info(
            "  → NER hallucination fallthrough — escalating to remote",
        )
    remote_answer, model_name, remote_tokens = _remote_model.generate(prompt)
    _cache.put(prompt, remote_answer, remote_tokens)
    total_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=remote_answer,
        path="remote",
        confidence=round(confidence, 4),
        tokens_used=remote_tokens,
        latency_ms=round(total_ms, 1),
    )


# ---------------------------------------------------------------------------
# Entry point (for `python api_server.py` without uvicorn CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    log.info("Starting API server on %s:%d", host, port)
    uvicorn.run("api_server:app", host=host, port=port, log_level="info")