# =============================================================================
# Hybrid Token-Efficient Routing Agent — Task Runner Dockerfile
# Target: linux/amd64
# =============================================================================
# Build stage: compile llama-cpp-python, install dependencies, copy Gemma model
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    coreutils \
    && rm -rf /var/lib/apt/lists/*
# Install Python dependencies (llama-cpp-python compiles against CPU-only BLAS
# for maximum portability on linux/amd64; ROCm BLAS can be enabled at build
# time by setting CMAKE_ARGS="-DLLAMA_HIPBLAS=ON" when deploying to AMD Instinct)
RUN pip install --no-cache-dir \
    llama-cpp-python \
    requests \
    numpy
# Gemma 2B GGUF model (~1.7 GB) is downloaded on the host beforehand and
# copied in here, rather than fetched via curl during the build — large
# single-shot HTTPS downloads were unreliable inside the build environment
# (WSL2/Docker Desktop networking dropped the connection repeatedly).
# Expected location on host: agent/models/gemma-2b-it-Q4_K_M.gguf
# If you need to re-download it, search Hugging Face for "gemma-2-2b-it-GGUF".
ARG EXPECTED_MODEL_SHA256=""
COPY models/gemma-2b-it-Q4_K_M.gguf /models/gemma-2b-it-Q4_K_M.gguf
RUN SHA256=$(sha256sum /models/gemma-2b-it-Q4_K_M.gguf | cut -d' ' -f1) && \
    SIZE=$(stat -c%s /models/gemma-2b-it-Q4_K_M.gguf 2>/dev/null || stat -f%z /models/gemma-2b-it-Q4_K_M.gguf 2>/dev/null) && \
    echo "Model file: size=$SIZE bytes, sha256=$SHA256" && \
    if [ -n "$EXPECTED_MODEL_SHA256" ] && [ "$SHA256" != "$EXPECTED_MODEL_SHA256" ]; then \
        echo "FATAL: SHA256 mismatch — expected $EXPECTED_MODEL_SHA256, got $SHA256"; \
        exit 1; \
    fi && \
    if [ "$SIZE" -lt 1400000000 ]; then \
        echo "FATAL: Model file ($SIZE bytes) is smaller than 1.4 GB — corrupt or incomplete."; \
        exit 1; \
    fi
# =============================================================================
# Runtime stage: minimal image with only the installed packages and model file
FROM python:3.11-slim
# Install libgomp1 for llama-cpp-python's OpenMP shared library,
# plus coreutils (for stat(1)) and curl (for runtime model fallback fetch).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    coreutils \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /models /models
# Create non-root user
RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --ingroup app app
COPY --chown=app:app task_runner.py /app/task_runner.py
COPY --chown=app:app solvers.py /app/solvers.py
COPY --chown=app:app benchmark.py /app/benchmark.py
WORKDIR /app
USER app
ENTRYPOINT ["python", "task_runner.py"]
