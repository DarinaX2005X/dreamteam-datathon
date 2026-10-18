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

The separate frozen-vision onset-transition probe is implemented in
`training/train_qwen_onset_probe.py` and `deploy/modal_onset_probe.py`. It has a
25-minute L4 ceiling, source-balanced losses, point-only positive supervision,
and a threshold selected from three training folds. Its first completed pilot
encoded 1,785 frames in a total of 132.138 seconds, but no OOF threshold met the
specificity/recall guard. Classifier validation was therefore skipped and no
checkpoint was selected. This did not improve production recall. The private
feature cache can support further bounded CPU analysis without video decoding.
See `eval/reports/modal-qwen-onset-probe.json` for the measured result.

Two fixed linear refinements reused that exact private cache on CPU only:
signed/absolute temporal differences, and the full pre/post/difference features.
Both fitted their training folds well but failed source-grouped OOF calibration;
neither met specificity >= 0.75 and recall >= 0.50 at any threshold. The run took
8.827 seconds with two CPUs and no GPU. Validation was not scored and no weights
were selected. `training/refine_qwen_onset_probe.py` and
`deploy/modal_onset_refinement.py` preserve the recipe; the numeric result is in
`eval/reports/modal-qwen-onset-refinement.json`. These experiments did not change
the deployed 2 TP / 4 TN / 0 FP / 2 FN development result.

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
.modal-venv/Scripts/python.exe -X utf8 -m modal deploy -m deploy.modal_app
```

One L4 container owns API, worker and SQLite; min_containers=0, max_containers=1,
90-second idle window. The website uses Qwen3-VL-8B-Instruct with native video,
at most 32 sampled frames at 1 fps, object-focused detail views, and prompt
v10-questions. Its base v8 observations are supplemented with separate factual
questions over nine-frame windows for the unattended-bag case scenario.
`lado-objects-v7-composite` supplies separately trained hand, cash,
identity-document and money-counter boxes. The profile has an explicit experimental
label: the eight-source development result is TP2/TN4/FP0/FN2 (previously
TP1/TN4/FP0/FN3), with all eight completing. This repeatedly used development
set is not an independent final accuracy estimate; two positives remain missed.

The 2026-10-18 question-clock correction uses window-relative timestamps in both
native video metadata and question text, while returned indices still map to source
PTS. A complete eight-source comparison preserved all verdicts and finding onsets;
it is a consistency fix, not a recall improvement. The published runtime is
`0.1.0+runtime.12cc82a195efa0d45a8d`. See
`eval/reports/modal-action-question-audit.json` for the rejected action-question
diagnostics and full comparison, and `eval/reports/pitch-redesign-release.json`
for the redesigned PDF/PPTX and GitHub-button HTTP checks. Browser and slide
visual review remain user-owned.
See `eval/reports/modal-question-verification.json`. A later paired diagnostic
tested 16-frame windows at 2 fps with an 8-frame stride on the two remaining
positive misses. Neither was recovered; both runs became slower, so that profile
was rejected. The deployed 32-frame profile is retained. Details and source-time
handling are recorded in `eval/reports/modal-temporal-window-audit.json`.
`NAVIDU_SEMANTIC_STRIDE_FRAMES=0` preserves the historical two-frame overlap;
an explicit stride is included in the runtime fingerprint.

Finding descriptions now show one onset time from the rule anchor. The complete
supporting interval remains available internally for evidence and sequence checks.
The deployed API confirmed the bag case at 16.0 seconds with the onset description.

The presentation is built from `presentation/pitch/source/build_deck.js`. Its final
PDF and editable PPTX are copied into `web/public/presentation/` before the remote
web build. The service exposes only `lado-pitch.pdf` and `lado-pitch.pptx` through
`/presentation/{filename}`; the source art and working documents are not published
through that route. PDF opens inline and PPTX downloads as an attachment. Verify
the actual response bytes against the final files, not only the HTTP status.

The identity
model has not established recall on separate development videos. A trained language
adapter is not enabled automatically.
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

The v10 update passed readiness and key checks. The previously missed bag case
returned one finding at 16 seconds and one evidence frame in 174.085 seconds
including cold startup. A short clean control returned no findings in 92.159 seconds.
Both public jury responses matched the schema; UI visual testing remains with the user.

Before the v10 update, the 8B/object profile passed readiness and key checks: HTTP 200,
missing/invalid keys rejected with 401, and an authenticated unknown job returned 404.
A 22.6-second positive development source completed with one finding and a valid JPEG
in 191.654 seconds from a cold request. The 78.733-second clean source completed
without findings but exceeded 300 seconds. After bounding database lease checks,
the repeat also exceeded that limit: perception took 323.5135 seconds, and the first
observed completed response was at 479.431 seconds including a cold POST of 104.797
seconds. The observation time is an upper bound, not an exact completion timestamp.
The user explicitly stopped further work on this long-video limit. Do not present
timings from the older 4B profile as timings of the currently deployed object profile.

Technical completion does not establish model quality. Current numeric evidence is
in `eval/reports/modal-question-verification.json`; prior object-profile results are
in `eval/reports/reviewed-spatial-objects.json`. The user-only credentials are in
ignored `.env.jury`; never commit it. Status reads after idle shutdown can also
require a cold start; the inspection helper allows up to 240 seconds for that GET
within its overall budget.

## References

Code checks and processor checks run remotely after the user's local CPU restriction:

```powershell
.modal-venv/Scripts/python.exe -X utf8 -m modal run -m deploy.modal_checks
.modal-venv/Scripts/python.exe -X utf8 -m modal run -m deploy.modal_evaluation::cpu_preflight
.modal-venv/Scripts/python.exe -X utf8 -m modal run -m deploy.modal_evaluation::main --run-id <unique-id> --model-id qwen3-vl-8b --prompt-variant v10-questions --input-mode video --window-frames 32 --max-pixels 262144 --detail-view --object-model lado-objects-v7-composite --max-new-tokens 1536 --max-sources 8
```

Use module mode (`-m deploy.modal_evaluation`) so Modal includes its package imports.
Keep image inputs unchanged until the build finishes. Only explicit source paths enter
the image; private data remains in the authorized Volume. Native context and optional
detail crops are experimental profiles, not evidence of improved model quality.

The earlier 4B/native64/v8 development comparison completed all eight sources:
1 TP / 4 TN / 0 FP / 3 FN, maximum 97.794 seconds per analyzed video. The new
simultaneous-person criterion produced the finding; no cash events or object boxes
were accepted. An earlier run had an additional false positive; excluding people
with contradictory physical/reflection claims in the same frames removed it.
Uncertainty does not hide detector boxes or transfer to the physical source.
Qwen 8B fit the same L4 (20,491 MiB peak) but did not improve that earlier
two-source comparison and was not selected then. See `eval/reports/modal-context-comparison.json`.
The current 8B/object/v10 profile and its full comparison are described above.
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
