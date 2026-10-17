#!/usr/bin/env bash
set -euo pipefail

cd /data/dreamteam
docker compose config -q

for _ in $(seq 1 10); do
  health=$(docker inspect lado-api-1 --format '{{.State.Health.Status}}')
  [[ "$health" == "healthy" ]] && break
  sleep 2
done
docker compose ps api

find scripts -maxdepth 2 -type f \( -iname '*model*' -o -iname '*fetch*' \) -print | sort

docker run --rm --gpus all lado-local:0.1.0 python -c \
  'import torch; print(f"cuda_available={torch.cuda.is_available()}"); print(f"device_count={torch.cuda.device_count()}"); print(f"device_0={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '\''none'\''}")'

python scripts/fetch_models.py --help
