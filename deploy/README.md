# Lado deployment preparation

The production shape is one API process and one warm model worker sharing the same SQLite/data
directory and read-only model directory. The worker is the only process assigned a GPU. The service
does not send videos or extracted frames to an external AI service.

## Local preparation

Copy `.env.example` to `.env`, replace `NAVIDU_API_KEYS`, then fetch and verify the pinned model
manifest before starting. A strong key can be generated locally with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`.

On Windows (choose `cpu`, `cuda`, or leave `auto`; `-VenvPath` permits an isolated environment):

```powershell
.\scripts\setup_local.ps1 -Device auto -VenvPath backend/.venv -FetchModels
backend\.venv\Scripts\python.exe scripts\run_local.py
```

On Linux (`--device=cpu|cuda|auto`; `--venv` permits an isolated environment):

```bash
./scripts/setup_local.sh --device=auto --venv=backend/.venv --fetch-models
backend/.venv/bin/python scripts/run_local.py
```

The launch command prints `API bound` when HTTP is reachable. This is not a claim that inference is
ready. `/api/ui/health` must report `ready: true` and `model_status: ready` before accepting uploads.
No browser is opened. API and worker logs stay in `logs/`; job data stays in `data/`.

For a CPU-only container smoke image that does not pull CUDA layers, use
`docker build -f deploy/Dockerfile.cpu -t lado-local:cpu .`. It verifies packaging and HTTP
startup only; it does not establish the GPU model acceptance criteria below.

## NVIDIA Brev L40S preparation

Do not create or change an instance until the operator has paid for and selected it. For the planned
`massedcompute_L40S` VM, select the L40S 48 GB / x86_64 option, 625 GB disk, and VM mode. Brev says
GPU instances include Python, CUDA, Docker and Compose. Confirm rather than assume the delivered
machine:

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If the final command fails, follow NVIDIA's current Container Toolkit guide. It calls for installing
the toolkit, running `nvidia-ctk runtime configure --runtime=docker`, and restarting Docker. The
project setup and preflight scripts deliberately do not modify or restart the host daemon.

Keep the checkout, `data/`, models and exports under `/home/ubuntu/workspace`; Brev documents that
this location persists when an instance is stopped but is destroyed when it is deleted. Brev also
warns that credit exhaustion can eventually delete resources. Run `deploy/export_data.py` to a
user-selected mounted/local destination before that risk. The export excludes source videos by
default.

For the selected stoppable instance, run `bash deploy/brev-bootstrap.sh` after placing the checkout at
`/home/ubuntu/workspace/dreamteam`. Before an overnight stop, run `bash deploy/brev-stop-safe.sh`; it
stops the worker before the API, makes a SQLite backup under `backups/`, and flushes the persistent
workspace. Then run `brev stop INSTANCE_NAME` on the connecting computer. Resume with
`brev start INSTANCE_NAME`, `brev refresh`, reconnect, and run `bash deploy/brev-resume.sh`. Restart can
fail temporarily when the provider has no free GPU of the same type and region; the workspace stays
bound to that provider until capacity returns.

After the source and verified models are present:

```bash
cp .env.example .env
# edit NAVIDU_API_KEYS; keep .env private
python3 scripts/preflight.py --host-only --device cuda --min-vram-gib 40 --min-ram-gib 60 --min-disk-gib 80
docker compose -f docker-compose.yml -f deploy/docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f deploy/docker-compose.gpu.yml up -d
docker compose -f docker-compose.yml -f deploy/docker-compose.gpu.yml logs -f api worker
```

Use a Brev Secure Link for HTTP port `${NAVIDU_PORT:-8000}` when authenticated browser access is
appropriate, or expose the TCP port only to the deployer's IP for jury API access. `X-API-Key`
remains required on `/api/v1`. Do not expose SQLite, model volumes, or a second worker port.

Run the real jury smoke test with `.mov` paths only after health reports ready:

```bash
python3 scripts/smoke_api.py --base https://YOUR_ASSIGNED_ORIGIN --key "$NAVIDU_API_KEY" \
  --files /path/a.mov /path/b.mov --parallel 2 --timeout 300
```

For the five-file acceptance scenario, pass all five files and `--parallel 5`. The script validates
terminal schema fields, nonnegative timecodes, base64 and JPEG framing, and enforces the 300-second
deadline per upload/job. It reports measured values; it does not substitute fixture results.

## Checks still requiring the paid L40S

The L40S GPU identity and container access, real model warm-up, offline inference, five concurrent
145-second files, each job's 300-second bound, and the assigned public HTTPS origin must be checked
on the actual instance. Laptop CPU or RTX 2050 results do not establish those properties.

Primary references checked 2026-10-17:

- [NVIDIA Brev GPU instance lifecycle and persistence](https://docs.nvidia.com/brev/concepts/gpu-instances)
- [NVIDIA Brev Launchables, ports, secure links and launch parameters](https://docs.nvidia.com/brev/concepts/launchables)
- [NVIDIA Container Toolkit installation and Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose GPU allocation](https://docs.docker.com/compose/how-tos/gpu-support/)
- [Hugging Face pinned snapshot downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download)

