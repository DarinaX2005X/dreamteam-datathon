#!/usr/bin/env bash
set -euo pipefail

cd /data/dreamteam
for _ in $(seq 1 15); do
  api_health=$(docker inspect lado-api-1 --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' 2>/dev/null || true)
  worker_state=$(docker inspect lado-worker-1 --format '{{.State.Status}}' 2>/dev/null || true)
  [[ "$api_health" == "healthy" && "$worker_state" == "running" ]] && break
  sleep 2
done

docker compose ps
for endpoint in / /app /api/ui/health; do
  code=$(curl -sS -o /tmp/lado-status-body -w '%{http_code}' --max-time 10 "http://127.0.0.1:8080${endpoint}")
  size=$(wc -c < /tmp/lado-status-body)
  printf '%s %s %sbytes\n' "$endpoint" "$code" "$size"
done

printf '%s\n' '--- api logs ---'
docker logs --tail 30 lado-api-1 2>&1
printf '%s\n' '--- worker logs ---'
docker logs --tail 50 lado-worker-1 2>&1
printf '%s\n' '--- gpu processes ---'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
printf '%s\n' '--- identities ---'
docker inspect lado-api-1 --format 'api_id={{.Id}} health={{.State.Health.Status}} network={{.HostConfig.NetworkMode}}'
docker inspect lado-worker-1 --format 'worker_id={{.Id}} state={{.State.Status}} gpu_requests={{json .HostConfig.DeviceRequests}}'
docker exec lado-api-1 sh -c 'printf "experimental=%s require_semantic=%s model=%s dtype=%s device=%s\n" "$NAVIDU_EXPERIMENTAL_ANALYSIS" "$NAVIDU_REQUIRE_SEMANTIC" "$NAVIDU_SEMANTIC_MODEL" "$NAVIDU_SEMANTIC_DTYPE" "$NAVIDU_DEVICE"'
printf '%s\n' '--- health body ---'
curl -sS --max-time 10 http://127.0.0.1:8080/api/ui/health
printf '\n'
