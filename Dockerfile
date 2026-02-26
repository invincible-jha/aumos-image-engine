# ============================================================================
# Dockerfile — aumos-image-engine (CUDA-enabled multi-stage build)
# ============================================================================
# Uses NVIDIA CUDA base for GPU-accelerated image generation.
# Non-root runtime user for security.
# ============================================================================

# Stage 1: Build dependencies
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.11 /usr/bin/python3 && ln -sf python3 /usr/bin/python

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install PyTorch with CUDA support first, then remaining deps
RUN pip install --prefix=/install --no-warn-script-location \
    torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --prefix=/install --no-warn-script-location .

# Stage 2: Runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.11 /usr/bin/python3 && ln -sf python3 /usr/bin/python

# Security: non-root user
RUN groupadd -r aumos && useradd -r -g aumos -d /app -s /sbin/nologin aumos

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ /app/src/
WORKDIR /app

# Create model cache directory
RUN mkdir -p /app/model_cache && chown -R aumos:aumos /app

USER aumos

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/live'); r.raise_for_status()" || exit 1

# Start service
CMD ["uvicorn", "aumos_image_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
