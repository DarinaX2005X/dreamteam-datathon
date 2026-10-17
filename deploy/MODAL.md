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
.modal-venv/Scripts/python.exe -X utf8 -m modal run deploy/modal_training.py::main --run-id modal-qwen-l4-lr5e5-20261017 --model-id qwen3-vl-4b
```

L4 24 GB, 4 physical CPU cores, at most 32 GiB RAM, one container, no retries or
warm pool. At most three epochs at LR 5e-5; stop after an epoch with no accepted
gain. Qwen uses its native processor with max_pixels=262144 per frame. Paired validation selects only a gain without losing any
previously correct source. Internal 90-minute budget; subprocess is stopped at
95 minutes with one minute to save; Modal terminates the function at 100 minutes.
GPU allocation stops when the function exits. The first real steps must confirm
that this recipe fits L4; no memory fit or accuracy improvement is claimed in advance.

The completed Qwen run `modal-qwen-l4-lr5e5-r3-20261017` fits L4 (16,573 MiB peak)
but did not improve validation: 4 TP / 0 TN / 4 FP / 0 FN. No adapter was selected.
See `eval/reports/modal-qwen-proposal-l4.json`. Its private checkpoint is retained.

Checkpoints are written before validation, committed to a private Volume, and copied
automatically to ignored `training/runs/<run-id>` on the local computer, including after
an ordinary training failure. Interrupted client connections may require manual export:

```powershell
.modal-venv/Scripts/python.exe -X utf8 -m modal volume get lado-training-v1 /runs/<run-id> training/runs
```

At rates checked 2026-10-17, the 100-minute GPU/CPU/RAM ceiling is approximately $2.08,
plus CPU setup, image building and any storage/network charges. Timeouts bound one
run, not all future activity or total account spend. The user subsequently requested
both the baseline Qwen website for their testing and Qwen training. The site retains
its own scale-to-zero L4; training uses a separate bounded L4 when the site is active.
Do not upgrade the GPU without measured need and user agreement.

## Website

```powershell
.modal-venv/Scripts/python.exe -X utf8 -m modal run -m deploy.modal_web_checks
.modal-venv/Scripts/python.exe -X utf8 -m modal deploy deploy/modal_app.py
```

One L4 container owns API, worker and SQLite; min_containers=0, max_containers=1,
90-second idle window. The website currently uses original Qwen3-VL-4B-Instruct with
native video, at most 64 sampled frames at 1 fps, and prompt v8-focused, with an
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

Confirmed: readiness HTTP 200, missing/invalid jury keys rejected with 401, valid
key accepted, full 65.3-second development video completed through API in 131.857
seconds. That positive video was missed. Technical completion does not establish
model quality. The user-only credentials are in ignored `.env.jury`; never commit it.

The final native-video v8 profile processed the 78.733-second clean development
source through the jury API in 230.439 seconds including a cold start. The response
was `done/no_violation`, with the exact required schema and experimental mode header.
The earlier version produced an extra-person false positive on this same source;
the physical/reflection consistency guard removed it. The measured time is below
the 300-second deadline for this request; it is not a guarantee for arbitrary videos.
HTTP readiness after another cold start took 63.875 seconds. A result and its JPEG
from the previous deployment remained readable after replacing the container.
Status reads after idle shutdown can also require a cold start; the inspection helper
allows up to 240 seconds for that GET within its overall 300-second budget.

## References

Code checks and processor checks run remotely after the user's local CPU restriction:

```powershell
.modal-venv/Scripts/python.exe -X utf8 -m modal run -m deploy.modal_checks
.modal-venv/Scripts/python.exe -X utf8 -m modal run -m deploy.modal_evaluation::cpu_preflight
.modal-venv/Scripts/python.exe -X utf8 -m modal run -m deploy.modal_evaluation::main --run-id <unique-id> --model-id qwen3-vl-4b --prompt-variant v8-focused --input-mode video --window-frames 64 --max-sources 2
```

Use module mode (`-m deploy.modal_evaluation`) so Modal includes its package imports.
Keep image inputs unchanged until the build finishes. Only explicit source paths enter
the image; private data remains in the authorized Volume. Native context and optional
detail crops are experimental profiles, not evidence of improved model quality.

The final v8 full development comparison completed all eight sources:
1 TP / 4 TN / 0 FP / 3 FN, maximum 97.794 seconds per analyzed video. The new
simultaneous-person criterion produced the finding; no cash events or object boxes
were accepted. An earlier run had an additional false positive; excluding people
with contradictory physical/reflection claims in the same frames removed it.
Uncertainty does not hide detector boxes or transfer to the physical source.
Qwen 8B fits the same L4 (20,491 MiB peak) but did not improve the
two-source comparison and is not deployed. See `eval/reports/modal-context-comparison.json`.
These repeatedly inspected development results are not an independent final estimate.
Holdout remains local and unused.

For old runs affected only by the `evidence/original` versus `frames/original` path
mismatch, `deploy.modal_evaluation::repair --run-id <existing-id>` validates saved
JPEGs and rebuilds the numeric projection on CPU, retaining original failure reports.
It neither reruns the model nor changes the saved findings.

- https://modal.com/pricing
- https://modal.com/docs/guide/budgets
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/webhooks
- https://modal.com/docs/guide/concurrent-inputs
