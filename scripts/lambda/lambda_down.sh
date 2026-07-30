#!/usr/bin/env bash
# Terminate the tracked instance (billing stops ONLY at termination), then
# sweep the account for anything else still running.
set -euo pipefail
# Resolved once, absolute, BEFORE the trap is armed: inside the trap
# BASH_SOURCE[0] is whatever file the shell was last reading — common.sh, not this
# script — and the sweep is the backstop that must not depend on that coincidence
# (or on the cwd the operator happened to invoke from).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# The sweep is the forgotten-box backstop, so it must survive every abort path
# in this script — a failed terminate POST is exactly when the operator needs
# to know what is still billing. || true: the sweep is best-effort (it calls
# the API too), and must never mask this script's own exit status.
trap 'echo "--- account sweep"; "$SCRIPT_DIR/lambda_status.sh" || true' EXIT

if [ -f "$INSTANCE_FILE" ]; then
  require_instance
  echo "Terminating $INSTANCE_ID..."
  # api() is fail-closed (curl --fail prints nothing on >=400), so a bare call
  # would abort here under errexit with no diagnostic — and this POST can land
  # server-side with its reply lost in transit, exactly like the launch POST.
  # Capture the exit code instead, same idiom as lambda_up.sh's launch step.
  terminate_status=0
  api POST /instance-operations/terminate "{\"instance_ids\": [\"$INSTANCE_ID\"]}" >/dev/null \
    || terminate_status=$?
  if [ "$terminate_status" -ne 0 ]; then
    echo "Terminate request FAILED — instance $INSTANCE_ID MAY still be up and billing." >&2
    echo "The account sweep below shows what is actually running; terminate from the Lambda console if it is still listed." >&2
    exit 1
  fi

  # Only a status of `terminated` actually OBSERVED here licenses deleting the
  # tracking file: that file is lambda_up.sh's one-box guard, and dropping it on
  # an unconfirmed teardown is how an unreachable API becomes two boxes billing.
  confirmed=no
  status=""
  for _ in $(seq 1 30); do
    status="$(json_field "$(api GET "/instances/$INSTANCE_ID" || echo '{}')" data.status)"
    if [ "$status" = "terminated" ]; then confirmed=yes; fi
    # Spelled out rather than `[ -z "$status" ] || [ "$status" = terminated ] && break`.
    # That compound parses as `(A || B) && break`, which is the wanted semantics — but
    # it is one edit from the classic `A && B || C` misread, and the explicit `if`
    # states the intent unambiguously. Not on the script whose job is stopping the
    # meter. (An empty status means the GET itself failed, so it stops the poll
    # without confirming anything — hence the check below.)
    if [ -z "$status" ] || [ "$status" = "terminated" ]; then break; fi
    echo "  status=$status"
    sleep 5
  done

  if [ "$confirmed" != "yes" ]; then
    echo "Terminate requested but NOT confirmed (last status: '$status') — keeping $INSTANCE_FILE so lambda_up.sh still refuses to launch a second box." >&2
    echo "Check the sweep below; re-run scripts/lambda/lambda_down.sh once the API responds." >&2
    exit 1
  fi

  python3 - <<PY
import json, time
meta = json.load(open("$INSTANCE_FILE"))
hours = (time.time() - meta["launched_at"]) / 3600
print(f"Terminated. Session: {hours:.2f} h x \${meta['price_cents']/100:.2f}/hr "
      f"= \${hours * meta['price_cents'] / 100:.2f}")
PY
  rm -f "$INSTANCE_FILE"
  # Close the tunnel, and only now that this box is confirmed dead — unconditional
  # it also ran on the no-tracking-file branch, where nothing was terminated.
  # Pinning $INSTANCE_IP keeps it off a tunnel to some other box; an empty ip (a box
  # that never finished provisioning, so never had a tunnel) degrades to the
  # port-only match, still safe here because the only box this script knows about
  # is the one just terminated.
  pkill -f "ssh.*-L $TUNNEL_PORT:127.0.0.1:$REMOTE_PORT.*$INSTANCE_IP" 2>/dev/null || true
else
  echo "No tracked instance file — sweeping the account anyway."
fi
