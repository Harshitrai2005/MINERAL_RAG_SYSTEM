# ── Mineral Exploration Intelligence System v2 ───────────────────────────────
# Multi-stage build:
#   stage 1 (builder) — install all Python deps into /install
#   stage 2 (runtime) — copy only /install + app code, no build tools
# This cuts the final image size by ~40% vs a single-stage build.

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to compile/install some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .

# Install CPU-only PyTorch FIRST from the official CPU wheel index.
# This is the single most important Dockerfile trick for free-tier deployment:
# the default PyPI torch wheel bundles CUDA and is ~2.5GB; the CPU wheel is
# ~250MB. On Render's free 512MB RAM container, the CUDA wheel can cause OOM
# during pip install itself.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Now install everything else
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Only the runtime system libs (no gcc/g++ build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create runtime directories (data is mounted as a volume in docker-compose;
# on Render free tier with VECTOR_BACKEND=qdrant, these are empty placeholders)
RUN mkdir -p ./data/lancedb ./data/uploads ./logs

WORKDIR /app/backend

EXPOSE 8000

# Liveness probe — Render and docker-compose both poll this
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Single worker on free tier (512MB RAM — multiple workers would OOM)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
