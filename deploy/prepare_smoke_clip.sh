#!/usr/bin/env bash
set -euo pipefail

cd /data/dreamteam
find /data/dreamteam -maxdepth 1 -type f \( -name 'dataset-*.tar' -o -name 'dataset-*.tar.*' \) -printf '%f %s\n' | sort

subset=/data/dreamteam/dataset-6sha-e35f46baf71b-e41ee3941ea5.tar
python3 eval/prepare_transfer.py verify --archive "$subset"
python3 eval/prepare_transfer.py extract --archive "$subset" --destination /data/dreamteam

python3 - <<'PY'
import json
from pathlib import Path

manifest_path = Path('/data/dreamteam/eval/runs/transfer/manifests/ba5862dcf34d668d963a20488a582989d46a708bb1daa593b9230fac23ddb8bc.json')
if not manifest_path.is_file():
    raise SystemExit('verified development subset manifest is absent')
manifest = json.loads(manifest_path.read_text())
if manifest.get('reserved_final_holdout_video_sources') != 0 or manifest.get('development_video_sources') != 6:
    raise SystemExit('development subset isolation check failed')
videos = [item for item in manifest['files'] if item['path'].lower().endswith(('.mov', '.mp4', '.mkv', '.avi'))]
if not videos:
    raise SystemExit('no video in development subset')
videos.sort(key=lambda item: (item['size_bytes'], item['path']))
chosen = videos[0]
path = Path('/data/dreamteam') / chosen['path']
if not path.is_file() or path.stat().st_size != chosen['size_bytes']:
    raise SystemExit('selected development clip is absent or wrong size')
Path('/data/dreamteam/eval/runs/smoke-clip-path.txt').write_text(str(path) + '\n')
print(json.dumps({'development_candidates': len(videos), 'selected_size_bytes': chosen['size_bytes']}))
PY
