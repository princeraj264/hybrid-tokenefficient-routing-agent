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
from pydantic import BaseModel, Field

# Reuse the exact same classes from the batch task runner
from task_runner import Config, TaskCache, LocalModel, RemoteModel

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
    Run a single query through the three-tier routing pipeline.

    Returns the answer along with metadata about which tier handled the request,
    the confidence score, token usage, and latency.
    """
    prompt = req.query.strip()
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

    # ---- Tier 2: Local inference ----
    t0 = time.perf_counter()
    answer, confidence, tokens_used = _local_model.generate(prompt)
    local_latency = (time.perf_counter() - t0) * 1000

    if confidence >= _config.confidence_threshold:
        _cache.put(prompt, answer, tokens_used)
        total_ms = (time.perf_counter() - start) * 1000
        log.info(
            "  → Local (confidence=%.4f ≥ %.2f, latency=%.1fms)",
            confidence,
            _config.confidence_threshold,
            local_latency,
        )
        return QueryResponse(
            answer=answer,
            path="local",
            confidence=round(confidence, 4),
            tokens_used=tokens_used,
            latency_ms=round(total_ms, 1),
        )

    # ---- Tier 3: Remote escalation ----
    log.info(
        "  → Local confidence=%.4f < %.2f — escalating to remote",
        confidence,
        _config.confidence_threshold,
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