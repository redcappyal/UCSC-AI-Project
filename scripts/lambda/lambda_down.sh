#!/usr/bin/env bash
# Terminate the tracked instance (billing stops ONLY at termination), then
# sweep the account for anything else still running.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ -f "$INSTANCE_FILE" ]; then
  require_instance
  echo "Terminating $INSTANCE_ID..."
  api POST /instance-operations/terminate "{\"instance_ids\": [\"$INSTANCE_ID\"]}" > /dev/null
  for _ in $(seq 1 30); do
    status="$(json_field "$(api GET "/instances/$INSTANCE_ID" || echo '{}')" data.status)"
    # Spelled out rather than `[ -z "$status" ] || [ "$status" = terminated ] && break`.
    # That compound is measurably equivalent here — bash parses it `(A || B) && break`,
    # and errexit stays quiet because the failing test is a non-final member of an
    # &&/|| list — but it is one edit from the classic `A && B || C` misread and it
    # leaves $? at 1 after the loop. Not on the script whose job is stopping the meter.
    if [ -z "$status" ] || [ "$status" = "terminated" ]; then break; fi
    echo "  status=$status"
    sleep 5
  done
  python3 - <<PY
import json, time
meta = json.load(open("$INSTANCE_FILE"))
hours = (time.time() - meta["launched_at"]) / 3600
print(f"Terminated. Session: {hours:.2f} h x \${meta['price_cents']/100:.2f}/hr "
      f"= \${hours * meta['price_cents'] / 100:.2f}")
PY
  rm -f "$INSTANCE_FILE"
else
  echo "No tracked instance file — sweeping the account anyway."
fi

# Close any tunnel to a dead box.
pkill -f "ssh.*-L $TUNNEL_PORT:127.0.0.1:$REMOTE_PORT" 2>/dev/null || true

echo "--- account sweep"
"$(dirname "${BASH_SOURCE[0]}")/lambda_status.sh"
