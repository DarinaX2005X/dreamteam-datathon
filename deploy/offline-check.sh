#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 API_KEY VIDEO.mov [VIDEO.mov ...]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
api_key="$1"
shift
cd "$project_root"

# The Compose network is internal: containers can communicate and the host can reach the
# published API port, but neither service has an external network route.
docker compose up -d --no-build
python3 scripts/smoke_api.py \
  --base "http://127.0.0.1:${NAVIDU_PORT:-8000}" \
  --key "$api_key" --files "$@" --timeout 300
