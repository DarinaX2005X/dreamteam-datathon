#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="$project_root/backend"
venv="${VENV_PATH:-$backend/.venv}"
python_command="${PYTHON:-python3}"
device="${DEVICE:-auto}"
fetch_models=0
skip_models=0

for argument in "$@"; do
  case "$argument" in
    --fetch-models) fetch_models=1 ;;
    --skip-models) skip_models=1 ;;
    --device=cpu|--device=cuda|--device=auto) device="${argument#--device=}" ;;
    --venv=*) venv="${argument#--venv=}" ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done

cd "$project_root"
if [[ ! -x "$venv/bin/python" ]]; then
  "$python_command" -m venv "$venv"
fi
"$venv/bin/python" -m pip install --timeout 120 --retries 10 --upgrade pip
if [[ "$device" == auto ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then device=cuda; else device=cpu; fi
fi
[[ "$device" == cpu || "$device" == cuda ]] || { echo "Device must be auto, cpu, or cuda." >&2; exit 2; }
if [[ "$device" == cuda ]]; then torch_index="https://download.pytorch.org/whl/cu124"; else torch_index="https://download.pytorch.org/whl/cpu"; fi
echo "Installing the $device runtime in $venv"
"$venv/bin/python" -m pip install --timeout 120 --retries 10 --index-url "$torch_index" 'torch==2.6.0' 'torchvision==0.21.0'
"$venv/bin/python" -m pip install --timeout 120 --retries 10 -e "$backend[service,runtime,$device]"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required on PATH." >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is required on PATH." >&2; exit 1; }
command -v npm >/dev/null || { echo "Node.js/npm is required to build the Lado interface." >&2; exit 1; }
npm ci
npm run build

if (( fetch_models )); then
  [[ -f scripts/fetch_models.py ]] || { echo "scripts/fetch_models.py is not available yet." >&2; exit 1; }
  "$venv/bin/python" scripts/fetch_models.py --profile "$device"
fi
if (( ! skip_models )); then
  "$venv/bin/python" scripts/preflight.py --require-models || {
    echo "Runtime/model preflight failed. Use --fetch-models after pins are available, or --skip-models only for API development." >&2
    exit 1
  }
fi
echo "Setup complete. Start with: $venv/bin/python scripts/run_local.py"
