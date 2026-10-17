#!/usr/bin/env bash
set -euo pipefail

cd /data/dreamteam

for _ in $(seq 1 20); do
  health=$(docker inspect lado-api-1 --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}')
  [[ "$health" == "healthy" ]] && break
  sleep 2
done

docker compose ps
docker inspect lado-api-1 --format 'id={{.Id}} network={{.HostConfig.NetworkMode}} health={{.State.Health.Status}}'
ss -ltnp '( sport = :8080 )'

for endpoint in / /app /api/ui/health; do
  code=$(curl -sS -o /tmp/lado-check -w '%{http_code}' --max-time 10 "http://127.0.0.1:8080${endpoint}")
  size=$(wc -c < /tmp/lado-check)
  printf '%s %s %sbytes\n' "$endpoint" "$code" "$size"
done

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
