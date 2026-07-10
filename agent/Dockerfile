# =============================================================================
# Hybrid Token-Efficient Routing Agent — Task Runner Dockerfile
# Target: linux/amd64
# =============================================================================
# Build stage: compile llama-cpp-python and install dependencies
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (llama-cpp-python compiles against CPU-only BLAS
# for maximum portability on linux/amd64; ROCm BLAS can be enabled at build
# time by setting CMAKE_ARGS="-DLLAMA_HIPBLAS=ON" when deploying to AMD Instinct)
RUN pip install --no-cache-dir \
    llama-cpp-python \
    requests \
    numpy

# =============================================================================
# Runtime stage: minimal image with only the installed packages
FROM python:3.11-slim

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --ingroup app app

COPY --chown=app:app task_runner.py /app/task_runner.py

WORKDIR /app
USER app

ENTRYPOINT ["python", "app/task_runner.py"]