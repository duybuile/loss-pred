# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv from the official image (avoids pip overhead)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests — these rarely change, so this layer is cached
# unless pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./

# Install production dependencies into /app/.venv
# --frozen: no lock-file updates, guarantees reproducibility
# --no-dev: skip pytest and other dev-only packages
# --no-install-project: don't install the project package itself (we COPY source below)
RUN uv sync --frozen --no-dev --no-install-project

# Pre-download the sentence-transformer embedding model so the first request
# is not delayed by a network fetch inside the container.
RUN uv run python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the pre-built venv and cached model weights from the builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
# /root/.cache/torch is not copied — sentence_transformers stores model weights
# in the huggingface cache; the torch cache is only written for GPU/CUDA builds.

# Copy application source — order from least to most frequently changed
COPY conf/ conf/
COPY evals/ evals/
COPY app/ app/
COPY utils/ utils/
COPY prompt/ prompt/

# data/ and app/artifacts/ are intentionally excluded from the image.
# Mount them at runtime via docker-compose volumes (see docker-compose.yml).

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
