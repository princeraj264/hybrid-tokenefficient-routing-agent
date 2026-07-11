# =============================================================================
# Hybrid Token-Efficient Routing Agent — Task Runner Dockerfile
# Target: linux/amd64
# =============================================================================
# Build stage: compile llama-cpp-python, install dependencies, download Gemma model
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (llama-cpp-python compiles against CPU-only BLAS
# for maximum portability on linux/amd64; ROCm BLAS can be enabled at build
# time by setting CMAKE_ARGS="-DLLAMA_HIPBLAS=ON" when deploying to AMD Instinct)
RUN pip install --no-cache-dir \
    llama-cpp-python \
    requests \
    numpy

# Download Gemma 2B GGUF model from Hugging Face.
# This file is ~1.7 GB; the image will be self-contained once built.
# If this URL becomes unavailable, search Hugging Face for "gemma-2-2b-it-GGUF"
# and update the URL accordingly.
RUN mkdir -p /models && \
    curl -L -o /models/gemma-2b-it-Q4_K_M.gguf \
    https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf

# =============================================================================
# Runtime stage: minimal image with only the installed packages and model file
FROM python:3.11-slim

# Install libgomp1 for llama-cpp-python's OpenMP shared library
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /models /models

# Create non-root user
RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --ingroup app app

COPY --chown=app:app task_runner.py /app/task_runner.py
COPY --chown=app:app solvers.py /app/solvers.py

WORKDIR /app
USER app

ENTRYPOINT ["python", "task_runner.py"]