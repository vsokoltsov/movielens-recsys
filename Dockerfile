# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/usr/local/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv venv /opt/venv && uv sync --frozen

COPY recsys/ /app/recsys/
EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "recsys.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ml
COPY notebooks/ /app/notebooks/
EXPOSE 8888