#!/usr/bin/env bash
# Analyze one clip on the box: rsync up -> /api/upload -> /api/track -> poll
# -> rsync the finished run back into local ui_runs/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
  echo "usage: lambda_run.sh <clip> [--calibration cal.json] [--start S] [--end E] [--stride N]" >&2
  echo "       lambda_run.sh --resume <run_id>" >&2
  exit 1
}

CLIP="" CALIBRATION="" START_T="0" END_T="" STRIDE="" RESUME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --calibration) CALIBRATION="$2"; shift 2 ;;
    --start) START_T="$2"; shift 2 ;;
    --end) END_T="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    -*) usage ;;
    *) CLIP="$1"; shift ;;
  esac
done
[ -n "$RESUME" ] || [ -n "$CLIP" ] || usage
require_instance

REMOTE_APP="http://127.0.0.1:5188"
poll_and_sync() { # poll_and_sync RUN_ID
  local run_id="$1" status="" response=""
  while true; do
    response="$(box_ssh "curl -s $REMOTE_APP/api/track/status/$run_id")"
    # A transient blip can hand back an empty/partial body, and json_field
    # raises on that. Aborting a multi-hour run over one bad tick would be
    # absurd — treat an unreadable poll as "no news" and ask again.
    status="$(json_field "$response" status 2>/dev/null)" || status=""
    if [ -z "$status" ]; then
      sleep 5
      continue
    fi
    printf '\r%s: %s frame %s/%s   ' "$status" \
      "$(json_field "$response" stage)" \
      "$(json_field "$response" processed_frames)" \
      "$(json_field "$response" total_frames)"
    case "$status" in
      complete) echo "" ; break ;;
      failed) echo "" ; echo "Run failed: $(json_field "$response" error)" >&2; exit 1 ;;
      cancelled) echo "" ; echo "Run was cancelled on the box." >&2; exit 1 ;;
    esac
    sleep 5
  done
  echo "Syncing run $run_id back..."
  mkdir -p "$REPO_ROOT/ui_runs"
  box_rsync "$REMOTE_USER@$INSTANCE_IP:UCSC-AI-Project/ui_runs/$run_id/" \
    "$REPO_ROOT/ui_runs/$run_id/"
  python3 - <<PY
import json, time
meta = json.load(open("$INSTANCE_FILE"))
hours = (time.time() - meta["launched_at"]) / 3600
print(f"Run $run_id synced to ui_runs/. Session so far: "
      f"{hours:.2f} h x \${meta['price_cents']/100:.2f}/hr = \${hours * meta['price_cents'] / 100:.2f}")
PY
}

if [ -n "$RESUME" ]; then
  poll_and_sync "$RESUME"
  exit 0
fi

[ -f "$CLIP" ] || { echo "Clip not found: $CLIP" >&2; exit 1; }
if [ -z "$CALIBRATION" ]; then
  CALIBRATION="$(python3 "$REPO_ROOT/lambda_cloud.py" find-calibration "$CLIP" \
    --ui-runs "$REPO_ROOT/ui_runs")" || exit 2
  echo "Using calibration: $CALIBRATION"
fi
[ -f "$CALIBRATION" ] || { echo "Calibration not found: $CALIBRATION" >&2; exit 1; }

clip_base="$(basename "$CLIP")"
echo "Uploading clip ($(du -h "$CLIP" | cut -f1 | tr -d ' '))..."
box_ssh mkdir -p incoming
box_rsync "$CLIP" "$REMOTE_USER@$INSTANCE_IP:incoming/$clip_base"
scp "${SSH_OPTS[@]}" "$CALIBRATION" "$REMOTE_USER@$INSTANCE_IP:/tmp/crosscourt_cal.json"

upload_response="$(box_ssh "curl -s -F video_file=@incoming/$clip_base $REMOTE_APP/api/upload")"
video_id="$(json_field "$upload_response" video_id)"
duration="$(json_field "$upload_response" duration)"
[ -n "$video_id" ] || { echo "Upload failed: $upload_response" >&2; exit 1; }
box_ssh "rm -f incoming/$clip_base"
if [ -z "$END_T" ]; then
  END_T="${duration:-359999}"   # /api/track clamps end to the video's last frame
fi

track_response="$(box_ssh "curl -s -F video_id=$video_id \
  -F 'calibration_json=</tmp/crosscourt_cal.json' \
  -F start_time=$START_T -F end_time=$END_T \
  ${STRIDE:+-F frame_stride=$STRIDE} \
  $REMOTE_APP/api/track")"
run_id="$(json_field "$track_response" run_id)"
[ -n "$run_id" ] || { echo "Track submit failed: $track_response" >&2; exit 1; }
echo "Tracking started: run $run_id (resume later with --resume $run_id)"
poll_and_sync "$run_id"
