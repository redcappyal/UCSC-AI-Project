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
    scripts/lambda/lambda_run.sh clip.mp4 \
        --stride 1 --inference-width 0           # rsync up, track on GPU, results into ui_runs/
    scripts/lambda/lambda_tunnel.sh              # optional: box UI at http://localhost:5199
    scripts/lambda/lambda_down.sh                # terminate = the ONLY thing that stops billing
    scripts/lambda/lambda_status.sh              # anytime: what is billing right now?

- GPU order: A6000 ($1.09/hr) → A10 ($1.29) → A100 ($1.99); hard cap $2/hr.
  No capacity → lambda_up prints the availability table and exits.
- **Run at native fidelity: `--stride 1 --inference-width 0`.** Every frame at full
  resolution is the fidelity the GPU is rented for and what the WASB ball detector was
  trained for — it tiles at native resolution and ignores `inference_width` outright
  (`job_runner.py`, `track_segments`). `/api/track`'s own defaults are laptop-shaped,
  `frame_stride=4` and `inference_width=960` (`app.py`), so left alone the coarse pass
  sees one frame in four and the numbers are not comparable to a local
  native-resolution eval. Take the server defaults only for a deliberately quick look.
- `--stride` and `--inference-width` override those defaults. `--inference-width` accepts
  only `0`, `640`, `960`, `1280` (`0` = native); anything else is rejected on the Mac,
  before the clip is rsynced to a box that is already billing. Under the default `local`
  (WASB) ball backend the width is inert either way — it bites only if `BALL_DETECTOR` is
  set to `rfdetr`, which is why passing `0` explicitly is worth the keystrokes.
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
- **`lambda_down.sh` exited non-zero?** It stopped before its account sweep, so nothing
  has confirmed the box is gone. Run `lambda_status.sh`, and terminate from the Lambda
  console if the instance is still listed.
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
