#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="$project_root/backend"
venv="$backend/.venv"
python_command="${PYTHON:-python3}"
fetch_models=0
skip_models=0

for argument in "$@"; do
  case "$argument" in
    --fetch-models) fetch_models=1 ;;
    --skip-models) skip_models=1 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done

cd "$project_root"
if [[ ! -x "$venv/bin/python" ]]; then
  "$python_command" -m venv "$venv"
fi
"$venv/bin/python" -m pip install --timeout 120 --retries 10 --upgrade pip
"$venv/bin/python" -m pip install --timeout 120 --retries 10 -e "$backend[service]"
"$venv/bin/python" -m pip install --timeout 120 --retries 10 -r deploy/requirements-cpu.txt

command -v ffmpeg >/dev/null || { echo "ffmpeg is required on PATH." >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is required on PATH." >&2; exit 1; }
command -v npm >/dev/null || { echo "Node.js/npm is required to build the Lado interface." >&2; exit 1; }
npm ci
npm run build

if (( fetch_models )); then
  [[ -f scripts/fetch_models.py ]] || { echo "scripts/fetch_models.py is not available yet." >&2; exit 1; }
  "$venv/bin/python" scripts/fetch_models.py
fi
if (( ! skip_models )); then
  "$venv/bin/python" scripts/preflight.py --require-models || {
    echo "Runtime/model preflight failed. Use --fetch-models after pins are available, or --skip-models only for API development." >&2
    exit 1
  }
fi
echo "Setup complete. Start with: $venv/bin/python scripts/run_local.py"
