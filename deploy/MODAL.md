# Modal: bounded training and website

Brev is stopped. The user authorized the 32 development videos and their annotations
in their private Modal workspace on 2026-10-17. No holdout videos or annotations are
uploaded. No external AI API processes the videos.

## Current requirements

Use `.modal-venv` (Modal 1.6.2), authenticated to the user's workspace. Modal requires
a payment method for GPU functions even with free credits. Set Workspace Spend limit
to $0 to restrict net spending to credits; confirm the actual credit balance in Billing.
The CLI billing summary reports incurred costs, not remaining credits.

The image uploads only explicit source paths, public model manifests, rules, contracts,
and built web assets. Virtualenvs, credentials, local caches and videos are excluded.
Create `lado-runtime` from the ignored `.env.modal`, containing only NAVIDU_API_KEYS.

## CPU preparation

Public model download and SHA256 verification happen before allocating a GPU:

```powershell
.modal-venv/Scripts/python.exe -X utf8 -m modal run deploy/modal_training.py::setup --stage models
```

The private `lado-training-v1` Volume receives only the verified development archive.
`deploy/prepare_modal_data.py` filters the organizer metadata to the same 32 sources;
it keeps source SHA identities and the fixed 24/8 partition. The filtered metadata
has a new dataset fingerprint; the local original evaluation files are unchanged.
Do not upload the unfiltered archive or the full project directory.

```powershell
.modal-venv/Scripts/python.exe -X utf8 -m modal run deploy/modal_training.py::setup --stage data
```

## One bounded training run

```powershell
.modal-venv/Scripts/python.exe -X utf8 -m modal run deploy/modal_training.py::main --run-id modal-l4-lr5e5-e3-20261017
```

L4 24 GB, 4 physical CPU cores, at most 32 GiB RAM, one container, no retries or
warm pool. Three epochs at LR 5e-5, including continuation after an initially poor
epoch. Paired validation still selects only a genuine gain without losing any
previously correct source. Internal 90-minute budget; subprocess is stopped at
95 minutes with one minute to save; Modal terminates the function at 100 minutes.
GPU allocation stops when the function exits. The first real steps must confirm
that this recipe fits L4; no memory fit or accuracy improvement is claimed in advance.

Checkpoints are written before validation, committed to a private Volume, and copied
automatically to ignored `training/runs/<run-id>` on the local computer, including after
an ordinary training failure. Interrupted client connections may require manual export:

```powershell
.modal-venv/Scripts/python.exe -X utf8 -m modal volume get lado-training-v1 /runs/<run-id> training/runs/<run-id>
```

At rates checked 2026-10-17, the 100-minute GPU/CPU/RAM ceiling is approximately $2.08,
plus CPU setup, image building and any storage/network charges. Timeouts bound one
run, not all future activity or total account spend. Do not run the website GPU
concurrently with training. Do not upgrade the GPU without measured need and user agreement.

## Website

```powershell
npm run build
.modal-venv/Scripts/python.exe -X utf8 -m modal deploy deploy/modal_app.py
```

One L4 container owns API, worker and SQLite; min_containers=0, max_containers=1,
90-second idle window. The website currently uses the original 4B model with an
explicit experimental label. A trained adapter is not enabled automatically.
`NAVIDU_ALLOW_EXPERIMENTAL_API=1` explicitly enables the current model's actual results
through the required jury routes. The exact response JSON stays unchanged;
`X-Analysis-Mode: experimental` identifies the unvalidated quality status. Diagnostic,
incomplete and failed analyses still cannot produce a successful verdict.
The cold container waits for its own worker to become ready before accepting HTTP.

A successful POST starts a bounded queue-wait input so accepted jobs can continue if
the browser closes. After readiness the intake page stops background health polling;
before upload it checks service readiness again and allows up to three minutes to wake.
Opening the application or making API requests wakes the GPU; static browsing on this
combined deployment also uses it until idle shutdown. It is not an always-free CPU site.

Verify API readiness, a synthetic upload, persistence and scale-to-zero through HTTP
and Modal task state. UI and visual checks belong exclusively to the user. These cloud
lifecycle checks must actually pass before claiming successful deployment.

## References

- https://modal.com/pricing
- https://modal.com/docs/guide/budgets
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/webhooks
- https://modal.com/docs/guide/concurrent-inputs
