# syntax=docker/dockerfile:1.6
FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY infra/trainer/requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip setuptools wheel \
 && python -m pip install --prefer-binary -r /app/requirements.txt

RUN python -c "import pufferlib; print('pufferlib ok')"

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
