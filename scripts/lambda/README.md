# Lambda GPU sessions

Ephemeral CUDA boxes for analysis + eval runs. Spec:
`docs/superpowers/specs/2026-07-30-lambda-cloud-compute-design.md`.

## One-time setup
1. Lambda console (https://cloud.lambda.ai) → API keys → generate; save it:
   `~/.config/crosscourt/lambda_api_key` (chmod 600). Never commit or paste it.
2. Billing set up in the console (launches are rejected otherwise).
3. First `lambda_up.sh` generates `~/.ssh/id_ed25519` and registers it as
   `crosscourt-mac`.

## A session
    scripts/lambda/lambda_up.sh                  # launch + bootstrap (~5 min, billing starts)
    # ^ SET A TIMER NOW — lambda_up prints the deadline and the overnight cost
    scripts/lambda/lambda_run.sh clip.mp4 \
        --stride 1 --inference-width 0           # rsync up, track on GPU, results into ui_runs/
    scripts/lambda/lambda_tunnel.sh              # optional: box UI at http://localhost:5199
    scripts/lambda/lambda_down.sh                # terminate = the ONLY thing that stops billing
    scripts/lambda/lambda_status.sh              # anytime: what is billing right now?

- **Set a phone timer the moment `lambda_up.sh` returns.** It prints the launch time,
  the hourly rate, a stop-by deadline and what the box costs if left overnight —
  because nothing else bounds a forgotten box. Lambda has no stopped state: the meter
  runs from the launch POST until `lambda_down.sh` terminates it, so a box forgotten
  over a weekend is $26–48/day. The sweep and `lambda_status.sh` only help someone
  who already remembered.
- GPU order: A6000 ($1.09/hr) → A10 ($1.29) → A100 ($1.99) → A100 SXM4 (`PREFER_TYPES`
  in `common.sh`, first-with-capacity wins — never a substitution), under a hard
  $2/hr cap (`PRICE_CAP_CENTS=200`, same file). No capacity under the cap → lambda_up
  prints the availability table and exits.
- `lambda_up.sh [git-ref]` takes an optional git ref (default `main`) — the branch or
  tag `bootstrap_remote.sh` clones and checks out on the box. Use it to run a branch
  that is not merged yet; the ref must exist on the **public remote**, not just locally.
- **Run at native fidelity: `--stride 1 --inference-width 0`.** Full resolution on every
  frame is the fidelity the GPU is rented for. Only one of those two flags still changes
  anything: `/api/track` defaults to `frame_stride=1` and `inference_width=960`
  (`app.py`), so the stride is already every frame and it is the **width** that quietly
  downgrades a run left alone. Pass `--stride 1` anyway — it costs nothing and keeps the
  invocation self-describing — but when a run comes back lower-fidelity than you meant,
  the width is the one to check.
- `--stride` and `--inference-width` override those defaults. `--inference-width` accepts
  only `0`, `640`, `960`, `1280` (`0` = native); anything else is rejected on the Mac,
  before the clip is rsynced to a box that is already billing. Under the default `rfdetr`
  ball backend the width binds — rfdetr resizes to it itself — so it is a real knob on
  every default run. It goes inert only under `BALL_DETECTOR=local`, where tiling owns
  scale and `MAX_BALL_FRAME_WIDTH` (1080p) is the cap that matters instead.
- Otherwise `lambda_run.sh` defaults to the full clip, with the calibration taken from
  your newest local run of the same video (`--calibration` to override; `--start` /
  `--end` to trim).
- The box's Flask listens on 127.0.0.1 only. Never start it with `HOST=0.0.0.0` on a
  cloud box: the app has no auth.
- The box holds zero secrets; the repo is public. OPENAI/ROBOFLOW keys stay on the Mac.

## When a script dies, check the meter before you retry
- **`lambda_up.sh` failed anywhere near the launch step?** Run
  `scripts/lambda/lambda_status.sh` BEFORE retrying `lambda_up.sh`. The launch POST can
  land server-side with its reply lost in transit, so a box may already be up and billing
  with nothing tracking it — and a blind retry buys a second one.
- **`lambda_down.sh` exited non-zero?** Something in the teardown was not confirmed — the
  terminate request failed in transit, or the status poll never saw `terminated`. The
  account sweep runs anyway (it is on an `EXIT` trap), so read its output: it, not the
  script's exit code, is what tells you whether the box is gone. If the sweep itself
  failed too, re-run `lambda_status.sh`, and terminate from the Lambda console if the
  instance is still listed. On an unconfirmed teardown the tracking file is kept on
  purpose, so `lambda_up.sh` still refuses to launch a second box; re-run
  `lambda_down.sh` once the API responds.
- **Unconfirmed teardown, but the sweep says nothing is running? The box is gone —
  delete the tracking file by hand:** `rm ~/.config/crosscourt/lambda_instance.json`.
  This is the exit from the likeliest wedge, and you will need it. Lambda commonly
  404s an instance shortly after it terminates, and a status can also sit on
  `terminating` past the poll's 2.5 min; either way `lambda_down.sh` never observes
  the literal `terminated` it requires, so it exits 1 and keeps the file — and
  re-running it just POSTs terminate against an id that is already gone, which fails
  the same way forever. **The account sweep is the authority, not the exit code.** If
  it lists nothing (or lists nothing at your instance id), the meter has stopped;
  delete the file and carry on. If it still lists the instance, do NOT delete the
  file — that box is still billing, so terminate it from the Lambda console first.
- **Bootstrap failed?** The box is left up on purpose for debugging — and is still
  billing. `lambda_down.sh` when you are done with it.
- A clean teardown is one you can read: `lambda_down.sh` ends with the same account sweep
  `lambda_status.sh` prints, and you want it to say
  `Nothing is running — nothing is billing.`

## Known rough edges
Deliberately open, not bugs anyone forgot:
- **The run driver is not detachable.** An ssh drop or a sleeping laptop aborts
  `lambda_run.sh` mid-poll while the job carries on running on the box. Re-attach with
  `scripts/lambda/lambda_run.sh --resume <run_id>` — the run id is printed the moment
  tracking starts.
- **A wrong `--resume` id polls forever.** If the id is bad, or the box lost the job to a
  Flask restart, the poll prints `waiting for status` every 5 s and never exits on its
  own. Ctrl-C it — the box keeps running, and keeps billing — then look at the box
  (`lambda_tunnel.sh`) or tear it down.
