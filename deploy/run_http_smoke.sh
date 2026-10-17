#!/usr/bin/env bash
set -euo pipefail

cd /data/dreamteam
clip=$(cat /data/dreamteam/eval/runs/smoke-clip-path.txt)
state=/data/dreamteam/eval/runs/http-smoke-state.json
response=/data/dreamteam/eval/runs/http-smoke-upload-response.json

duration=$(docker run --rm -v "$clip:/tmp/input:ro" lado-local:0.1.0 \
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 /tmp/input)
size=$(stat -c %s "$clip")

if [[ ! -f "$state" ]]; then
  start=$(date +%s)
  http_code=$(curl -sS --max-time 120 -o "$response" -w '%{http_code}' \
    -F 'file=@'"$clip"';type=video/quicktime' http://127.0.0.1:8080/api/ui/checks)
  python3 - "$response" "$state" "$start" "$http_code" "$duration" "$size" <<'PY'
import json
import sys
from pathlib import Path

response_path, state_path, start, http_code, duration, size = sys.argv[1:]
payload = json.loads(Path(response_path).read_text())
if http_code != '201':
    error = payload.get('error', {}) if isinstance(payload, dict) else {}
    print(json.dumps({'upload_http': int(http_code), 'error_code': error.get('code', 'UNKNOWN')}), flush=True)
    raise SystemExit(1)
check_id = payload['check']['id']
state = {'check_id': check_id, 'start_epoch': int(start), 'duration_seconds': float(duration), 'size_bytes': int(size)}
Path(state_path).write_text(json.dumps(state))
print(json.dumps({'upload_http': 201, 'duration_seconds': round(float(duration), 3), 'size_bytes': int(size)}), flush=True)
PY
fi

last=''
while true; do
  now=$(date +%s)
  curl -sS --max-time 15 -o /tmp/lado-smoke-status -w '%{http_code}' \
    "http://127.0.0.1:8080/api/ui/checks/$(python3 -c 'import json; print(json.load(open("/data/dreamteam/eval/runs/http-smoke-state.json"))["check_id"])')" \
    > /tmp/lado-smoke-http
  result=$(python3 - /tmp/lado-smoke-status "$state" "$now" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
state = json.load(open(sys.argv[2]))
check = payload.get('check', {})
status = check.get('status', 'unknown')
progress = check.get('progress') or {}
error = check.get('error') or {}
out = {
    'status': status,
    'elapsed_seconds': int(sys.argv[3]) - state['start_epoch'],
    'phase': progress.get('phase'),
    'percent': progress.get('percent'),
}
if status in {'done', 'failed', 'cancelled'}:
    out['error_code'] = error.get('code')
    out['finding_count'] = len((check.get('finding_set') or {}).get('findings') or [])
    out['frame_count'] = len((check.get('artifacts') or {}).get('evidence_frames') or [])
print(json.dumps(out, separators=(',', ':')))
PY
)
  status=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "$result")
  elapsed=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["elapsed_seconds"])' "$result")
  if [[ "$result" != "$last" ]]; then
    printf '%s http=%s\n' "$result" "$(cat /tmp/lado-smoke-http)"
    last="$result"
  fi
  [[ "$status" == 'done' ]] && break
  if [[ "$status" == 'failed' || "$status" == 'cancelled' ]]; then
    exit 1
  fi
  if (( elapsed >= 300 )); then
    printf '%s\n' '{"status":"timeout","elapsed_seconds":300}'
    exit 124
  fi
  sleep 5
done
