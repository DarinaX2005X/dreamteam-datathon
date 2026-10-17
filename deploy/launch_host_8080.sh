#!/usr/bin/env bash
set -euo pipefail

cd /data/dreamteam

python3 - <<'PY'
from pathlib import Path
import re

path = Path("docker-compose.yml")
text = path.read_text()

old = '''  volumes:
    - ./data:/app/data
    - ./models:/app/models:ro
  networks:
    - offline
  restart: "no"
'''
new = '''  volumes:
    - ./data:/app/data
    - ./models:/app/models:ro
  restart: "no"
'''
if old in text:
    text = text.replace(old, new, 1)

old = '''  api:
    <<: *runtime
    command: ["python", "-m", "navidu.service"]
    ports:
      - "${NAVIDU_BIND_ADDRESS:-127.0.0.1}:${NAVIDU_PORT:-8000}:8000"
'''
new = '''  api:
    <<: *runtime
    command: ["python", "-m", "navidu.service"]
    network_mode: host
    environment:
      NAVIDU_PORT: 8080
'''
if old not in text and 'network_mode: host' not in text:
    raise SystemExit("unexpected api service layout")
if old in text:
    text = text.replace(old, new, 1)

text = text.replace(
    "http://127.0.0.1:8000/api/ui/health",
    "http://127.0.0.1:8080/api/ui/health",
    1,
)

old = '''    command: ["sh", "-c", "python scripts/preflight.py --device $${NAVIDU_DEVICE:-auto} --require-models && exec python -m navidu.service.worker"]
    healthcheck:
'''
new = '''    command: ["sh", "-c", "python scripts/preflight.py --device $${NAVIDU_DEVICE:-auto} --require-models && exec python -m navidu.service.worker"]
    networks:
      - offline
    healthcheck:
'''
if old in text:
    text = text.replace(old, new, 1)

text = re.sub(
    r'(\n    networks:\n      - offline){2,}',
    '\n    networks:\n      - offline',
    text,
)

path.write_text(text)
PY

docker compose up -d --no-build --force-recreate api
