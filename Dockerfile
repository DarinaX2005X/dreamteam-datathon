# syntax=docker/dockerfile:1.7
FROM node:22.22.0-bookworm-slim@sha256:dd9d21971ec4395903fa6143c2b9267d048ae01ca6d3ea96f16cb30df6187d94 AS web-build
WORKDIR /src
COPY package.json package-lock.json ./
COPY web/package.json web/package.json
COPY landing/package.json landing/package.json
RUN npm ci
COPY web web
COPY landing landing
RUN npm run build

# PyTorch 2.6 CUDA 12.4 runtime is pinned to the x86_64 manifest verified for the L40S target.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee AS runtime

ARG APP_UID=1000
ARG APP_GID=1000
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NAVIDU_HOST=0.0.0.0 \
    NAVIDU_PORT=8000 \
    NAVIDU_DATA_DIR=/app/data \
    NAVIDU_MODELS_DIR=/app/models \
    NAVIDU_RULES_DIR=/app/rules \
    NAVIDU_WEB_DIST=/app/web/dist \
    USE_TF=0 \
    USE_FLAX=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "$APP_GID" lado \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --create-home --shell /usr/sbin/nologin lado

WORKDIR /app
COPY backend /tmp/backend
COPY deploy/requirements-common.txt deploy/requirements-gpu.txt /tmp/deploy/
RUN python -m pip install --no-cache-dir -r /tmp/deploy/requirements-gpu.txt '/tmp/backend[service]' \
    && rm -rf /tmp/backend
COPY rules rules
COPY contracts contracts
COPY models/manifest.json models/manifest.json
COPY scripts/preflight.py scripts/preflight.py
COPY --from=web-build /src/web/dist web/dist
RUN mkdir -p data models \
    && chown -R lado:lado /app

USER lado
EXPOSE 8000
VOLUME ["/app/data", "/app/models"]
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ui/health', timeout=2).read()"
CMD ["python", "-m", "navidu.service"]
