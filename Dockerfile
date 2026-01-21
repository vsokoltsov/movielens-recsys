# syntax=docker/dockerfile:1.6

############################
# Base (shared)
############################
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/usr/local/bin:${PATH}"

WORKDIR /app

# OS deps (libgomp нужен implicit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN pip install --no-cache-dir uv

# dependencies layer (cache friendly)
COPY pyproject.toml uv.lock* ./
RUN uv venv /opt/venv

############################
# API image
############################
FROM base AS api

# install only what API needs
# (project deps + group api; no dev)
RUN uv sync --frozen --group api --no-dev

# copy only runtime code
COPY recsys/ /app/recsys/
COPY alembic.ini /app/alembic.ini
COPY recsys/alembic/ /app/recsys/alembic/

EXPOSE 8000

# run uvicorn directly (faster/cleaner than `uv run python -m ...`)
CMD ["uvicorn", "recsys.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]


############################
# ML image (optional)
############################
FROM base AS ml

# install api + train + notebooks (no dev)
RUN uv sync --frozen --group api --group train --group notebooks --no-dev

COPY recsys/ /app/recsys/
COPY alembic.ini /app/alembic.ini
COPY recsys/alembic/ /app/recsys/alembic/

COPY notebooks/ /app/notebooks/

EXPOSE 8888