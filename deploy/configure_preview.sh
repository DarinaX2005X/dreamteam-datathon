#!/usr/bin/env bash
set -euo pipefail

cd /data/dreamteam

python3 - <<'PY'
from pathlib import Path

env_path = Path('.env')
lines = env_path.read_text().splitlines() if env_path.exists() else []
updates = {
    'NAVIDU_EXPERIMENTAL_ANALYSIS': '1',
    'NAVIDU_REQUIRE_SEMANTIC': '1',
    'NAVIDU_DEVICE': 'auto',
    'NAVIDU_SEMANTIC_MODEL': 'internvl3_5-4b',
    'NAVIDU_SEMANTIC_DTYPE': 'bf16',
}
seen = set()
result = []
for line in lines:
    key = line.split('=', 1)[0] if '=' in line and not line.lstrip().startswith('#') else None
    if key in updates:
        result.append(f'{key}={updates[key]}')
        seen.add(key)
    else:
        result.append(line)
for key, value in updates.items():
    if key not in seen:
        result.append(f'{key}={value}')
env_path.write_text('\n'.join(result) + '\n')

compose_path = Path('docker-compose.yml')
text = compose_path.read_text()

model_volume = '    - ./models:/app/models:ro\n'
overlay_volumes = (
    model_volume
    + '    - ./backend/navidu:/opt/conda/lib/python3.11/site-packages/navidu:ro\n'
    + '    - ./contracts:/app/contracts:ro\n'
)
if '/site-packages/navidu:ro' not in text:
    text = text.replace(model_volume, overlay_volumes, 1)

worker_network = '''    networks:
      - offline
    healthcheck:
'''
worker_gpu = '''    networks:
      - offline
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
'''
if 'driver: nvidia' not in text:
    if worker_network not in text:
        raise SystemExit('unexpected worker layout')
    text = text.replace(worker_network, worker_gpu, 1)

compose_path.write_text(text)
PY

docker compose config -q
docker compose up -d --no-build --force-recreate api worker
