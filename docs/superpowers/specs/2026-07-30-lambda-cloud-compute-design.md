# Lambda GPU cloud compute for analysis + eval runs — design

**Date:** 2026-07-30
**Status:** Approved in discussion; pending spec review
**Owner:** Ian (account), pipeline scripts in-repo

## Problem

The analysis pipeline is ~50–200× too slow on the 8 GB M2 Air: the default local
WASB ball backend costs ~5.2 s per 416×416 tile on CPU (~94 s per 1080p frame,
~344 s per 4K frame at 18/66 tiles), and tile batches >1 swap-thrash 8 GB RAM.
`BALL_DEVICE` auto-selects CUDA, so the same pipeline on a CUDA box runs the
designed path with no code changes. The open WASB Task-8 gate (measure the new
detector on full videos, score with `/eval`) also needs a CUDA machine; the
Windows box is available only intermittently and has sm_120 quirks.

Ian created a lambda.ai (Lambda) account. This design sets up on-demand GPU
compute there.

## Decisions (agreed 2026-07-30)

1. **Scope:** analysis + eval runs only. No training, no persistent/always-on
   server. (Training later would only change the GPU choice; nothing here
   precludes it.)
2. **GPU policy:** prefer `A6000` (48 GB, $1.09/hr as of 2026-07-30), fall back
   A10 ($1.29) then A100 ($1.99). Hard cap **$2.00/hr** enforced in-script;
   anything above requires asking Ian. US regions. Lambda bills by the minute.
3. **Fully ephemeral:** no persistent Lambda filesystem. Fresh instance per
   session, bootstrap from the public repo, rsync clips up and results back,
   terminate when done. Zero idle cost, no region lock-in.
4. **Approach A + C hybrid:** scripted lifecycle (up / run / down / status)
   plus a documented SSH tunnel so the box's web UI can be used interactively
   from the local browser.

## Access model & secrets

- **API key (user-owned, one-time):** Ian generates an API key in the Lambda
  console and saves it to `~/.config/crosscourt/lambda_api_key` (chmod 600).
  Scripts read that file (or `$LAMBDA_API_KEY` if set). The key is never
  committed, never sent to the box, never printed.
- **SSH keypair (script-generated, one-time):** `~/.ssh/id_ed25519` does not
  exist yet; `lambda_up.sh` generates it (no passphrase — it only grants access
  to Ian's own ephemeral instances) and registers the public key with Lambda as
  `crosscourt-mac` via `POST /ssh-keys` (skipped if already registered).
- **The box carries zero secrets.** The repo is public (`git clone` needs no
  auth); `BALL_DETECTOR` defaults to the committed local model (no
  `ROBOFLOW_API_KEY`); the coach LLM call happens on the Mac when a report is
  opened locally, so no `OPENAI_API_KEY` leaves the Mac.
- **Network exposure:** Flask on the box binds `127.0.0.1:5188` (the app
  default; bootstrap must NOT set `HOST`). All HTTP reaches it through an SSH
  tunnel. The app has no auth; it must never listen publicly.

## Lambda API usage

Base `https://cloud.lambdalabs.com/api/v1`, header
`Authorization: Bearer <key>`. Endpoints used: `GET /instance-types`
(availability per region), `POST /instance-operations/launch`,
`GET /instances`, `POST /instance-operations/terminate`, `GET|POST /ssh-keys`.
Exact instance-type identifiers (e.g. `gpu_1x_a6000`) are read from
`GET /instance-types` at runtime rather than hard-coded guesses; the
implementation verifies real names on first live call.

## Components (`scripts/lambda/`, committed, secret-free)

### `lambda_up.sh`
1. Read API key; generate/register SSH key if needed.
2. `GET /instance-types`; pick the first of A6000 → A10 → A100 with capacity
   in a US region and price ≤ $2.00/hr. No capacity → print the availability
   table and exit non-zero. Never substitute a pricier type.
3. Launch; poll `GET /instances` until ACTIVE; wait for SSH.
4. Bootstrap over SSH:
   - `git clone https://github.com/redcappyal/UCSC-AI-Project` (`--ref`
     optional, default `main`);
   - `python3 -m venv --system-site-packages .venv` — reuses Lambda Stack's
     CUDA torch (`torch` is unpinned in requirements.txt, so pip treats it as
     satisfied);
   - `pip install -r requirements.txt` (includes `rfdetr==1.8.3`);
   - pre-warm the rfdetr person checkpoint (first `load_person_detector()`
     downloads it);
   - **CUDA gate:** `import torch; assert torch.cuda.is_available()` and one
     416×416×9 tile through the committed WASB model asserting device `cuda`,
     printing the per-tile latency. Known risk: `rfdetr==1.8.3` could resolve
     a conflicting torch into the venv; the gate catches it, and the scripted
     fallback is `pip install torch --index-url` (CUDA wheel — datacenter
     bandwidth makes this ~a minute).
   - start Flask in tmux: `.venv/bin/python app.py` (defaults
     `127.0.0.1:5188`).
5. Write `~/.config/crosscourt/lambda_instance.json` (instance id, IP, type,
   price, launch time) for the other scripts.

### `lambda_run.sh <clip-path> [--calibration <json>] [--start S] [--end E] [--resume <run_id>]`
1. rsync the clip to the box (`ui_runs/uploads/` staging).
2. Through the tunnel, drive the production path: `POST /api/upload` (sha256
   dedup) → `POST /api/track` (calibration JSON required by the API; default =
   the newest local `ui_runs/<run>/calibration.json` whose `job.json`
   `video_path` basename contains the clip's sha256 — the by-hash naming makes
   this an exact match — else `--calibration` is mandatory; clip bounds
   default to the full video).
3. Poll `GET /api/track/status/<run_id>` every 5 s (server-side progress
   updates are throttled to 0.5 s; 500 ms polling is a browser-UX cadence,
   not needed here) until complete/failed.
4. rsync the finished `ui_runs/<run_id>/` back into the local `ui_runs/`.
   Reports open in the local UI; the source-video route keeps working because
   the same sha256 file already exists in the local `by-hash` store. The
   returned `job.json` carries box-local `video_path`/`run_dir` strings — the
   report reads run-dir artifacts, so this is display-metadata only, but the
   implementation must verify the local report view of a returned run (the
   `/verify` skill's planted-run check).
5. Print run id, wall time, and effective $ cost (minutes × price).

### `lambda_tunnel.sh`
Open `ssh -N -L 5188:127.0.0.1:5188 ubuntu@<ip>` and print
`http://localhost:5188` — the interactive (approach-C) path. Also used by
`lambda_run.sh` (shared helper; one tunnel at a time — the script reuses an
existing tunnel if the port is already forwarded, and `lambda_run` must not
kill a tunnel the user opened interactively).

### `lambda_down.sh`
`POST /instance-operations/terminate` for the tracked instance; poll until
terminated (billing stops only at termination — Lambda has no stopped state,
so an in-box shutdown would NOT stop billing; that is why there is no auto-TTL
inside the box). Then `GET /instances` and print anything else still running
on the account — the forgotten-box sweep ($26/day at $1.09/hr).

### `lambda_status.sh`
List account instances with type, region, uptime, and burn rate. Zero
instances prints an explicit "nothing billing" line.

## Data flow

- Up: clip via rsync over SSH (resumable; 90–550 MB typical, minutes on home
  broadband). No egress fees on Lambda's side.
- Back: the run directory only (CSV, hits, players/, report JSON — a few MB).
  Source videos never come back down.
- Eval (Task 8): `lambda_run` each eval source video; runs land in local
  `ui_runs/`; `/eval` scores locally against the newest `eval_set/BASELINE-*`
  with zero changes to eval tooling.

## Failure modes & cost hygiene

| Failure | Behavior |
|---|---|
| No capacity under cap | Print availability table, exit non-zero |
| Bootstrap/CUDA gate fails | Leave instance up, print SSH command + `lambda_down` reminder (terminating would re-bill launch minutes while debugging) |
| Tunnel drops mid-run | Job keeps running server-side (jobs persist via `job.json`); `lambda_run --resume <run_id>` re-attaches by polling status and re-syncing |
| Forgotten instance | `lambda_down` sweep + `lambda_status`; by-the-minute billing bounds the damage |
| Mac sleeps mid-run | Same as tunnel drop: resume path |

## Verification plan

1. `lambda_up.sh` end-to-end: fresh instance reaches the CUDA gate; recorded
   per-tile GPU latency printed (expectation: ~10–50 ms/tile vs 5.2 s CPU-Mac).
2. Smoke run: the 15 MB clip through `lambda_run.sh`; report opens in the
   local UI with hits present.
3. One real 4K capture (~200 MB): measure wall clock + cost; compare hit
   output sanity against a previous local run of the same clip.
4. `lambda_down.sh`: instance terminates; `lambda_status.sh` shows nothing
   billing.
5. Suite stays green (`pytest tests/`): scripts are additive; no pipeline code
   changes in this project.

## Non-goals

- No persistent Lambda storage, no always-on service, no training runs.
- No changes to detection/judging code, eval tooling, or the iOS app.
- No auto-scheduling; sessions are started and ended by a human (or an agent
  driving these scripts interactively).

## One-time user checklist (blocking implementation testing)

1. Lambda console → generate API key → save to
   `~/.config/crosscourt/lambda_api_key` (chmod 600).
2. Confirm billing/payment is set up in the Lambda console (launches will
   otherwise be rejected).
3. Say the word before the first live launch — every launch bills the card,
   so the first `lambda_up.sh` run happens with explicit go-ahead.
