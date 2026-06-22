FROM python:3.12-slim

# libgomp1: OpenMP threading used by ONNX Runtime on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv — fast Python package manager (statically-linked binary, no extra deps)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy manifests first so dependency layer is cached independently of app code changes
COPY pyproject.toml uv.lock ./

# Install production deps into .venv
RUN uv sync --frozen --no-dev

# Add venv to PATH for both uv run and direct invocations
ENV PATH="/app/.venv/bin:$PATH"

# App code (no model folders — served from volume at runtime)
COPY app/ ./app/
COPY download_models.py ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Models directory — will be mounted as a named Docker volume
# Create it here with correct ownership before switching to non-root user
RUN mkdir -p /app/models

# Non-root user for production security
RUN useradd --uid 1000 --no-create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ONNX Runtime thread tuning — defaults tuned for 4-vCPU Azure VM
# intra_op: parallelism within a single op (matrix multiply) — set to vCPU count
# inter_op: parallelism between graph nodes — keep at 1 for transformer architecture
ENV ONNX_INTRA_OP_NUM_THREADS=4
ENV ONNX_INTER_OP_NUM_THREADS=1

# Models live in a named volume mounted here — never baked into the image
ENV MODELS_DIR=/app/models

EXPOSE 8001

ENTRYPOINT ["./entrypoint.sh"]
