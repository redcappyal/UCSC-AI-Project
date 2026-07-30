#!/usr/bin/env bash
# Launch one instance (A6000 -> A10 -> A100 under $2/hr), bootstrap, verify CUDA.
set -euo pipefail
# Resolved once, absolute: BASH_SOURCE[0] changes meaning inside a sourced file,
# and every later use of this path (the bootstrap scp) must be immune to cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REF="${1:-main}"   # optional git ref to bootstrap on the box (default: main)

# --- one box at a time
if [ -f "$INSTANCE_FILE" ]; then
  existing_id="$(python3 -c "import json;print(json.load(open('$INSTANCE_FILE'))['id'])")"
  # An unreachable API is not evidence the box is gone. A 404 on a GC'd id lands
  # in this refusal path by design — deleting tracking requires a human (or
  # lambda_down.sh), never a guess, because guessing wrong bills a second box.
  info="$(api GET "/instances/$existing_id")" || {
    echo "Cannot verify existing instance $existing_id (API error above) — refusing to launch another box. Check scripts/lambda/lambda_status.sh; if the old box is confirmed gone, delete $INSTANCE_FILE and re-run." >&2
    exit 1
  }
  status=""
  status="$(json_field "$info" data.status 2>/dev/null)" || status=""
  # Invariant shared with lambda_down.sh: ONLY an explicitly observed 'terminated'
  # licenses deleting the tracking file. An empty status is not evidence the box is
  # gone — it is a 2xx we could not read (shape change, `status: null`, partial
  # `data`) — and letting it fall through would launch a second box while the
  # tracked one is possibly still billing, leaving the OLD one untracked. The two
  # scripts must agree here or the one-box guard has a hole at exactly the moment
  # the API is behaving strangely.
  if [ "$status" != "terminated" ]; then
    echo "Instance $existing_id reports status '$status' (not 'terminated') — use it, or scripts/lambda/lambda_down.sh first." >&2
    echo "If it is genuinely gone (scripts/lambda/lambda_status.sh shows nothing), delete $INSTANCE_FILE and re-run." >&2
    exit 1
  fi
  rm -f "$INSTANCE_FILE"
fi

# --- ssh key: generate + register once
mkdir -p "$CONFIG_DIR" && chmod 700 "$CONFIG_DIR"
registered="$(api GET /ssh-keys | python3 -c "
import json, sys
names = [k['name'] for k in json.load(sys.stdin).get('data', [])]
print('yes' if '$SSH_KEY_NAME' in names else 'no')")"
if [ "$registered" = "yes" ] && [ ! -f "$SSH_KEY" ]; then
  echo "'$SSH_KEY_NAME' is registered with Lambda but $SSH_KEY is missing locally." >&2
  echo "Delete the key in the Lambda console (SSH keys) and re-run." >&2
  exit 1
fi
if [ ! -f "$SSH_KEY" ]; then
  ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "$SSH_KEY_NAME"
fi
if [ "$registered" = "no" ]; then
  pubkey="$(cat "$SSH_KEY.pub")"
  add_response="$(api POST /ssh-keys "{\"name\": \"$SSH_KEY_NAME\", \"public_key\": \"$pubkey\"}")"
  if [ -n "$(json_field "$add_response" error.message)" ]; then
    echo "Registering SSH key failed: $(json_field "$add_response" error.message)" >&2
    exit 1
  fi
  echo "Registered SSH key '$SSH_KEY_NAME' with Lambda."
fi

# --- pick type/region under the cap
# api() is fail-closed (curl --fail), so this pipeline fails for transport/auth
# errors too, not just "no capacity". Only pick-type's own exit 2 means no
# capacity; anything else would be a false diagnosis.
pick_status=0
picked="$(api GET /instance-types | python3 "$REPO_ROOT/lambda_cloud.py" pick-type \
  --prefer "$PREFER_TYPES" --cap-cents "$PRICE_CAP_CENTS")" || pick_status=$?
if [ "$pick_status" -eq 2 ]; then
  echo "No capacity under the cap right now; try again later or ask Ian about raising it." >&2
  exit 2
elif [ "$pick_status" -ne 0 ]; then
  echo "Instance-type query failed (not a capacity issue) — see the curl/python error above." >&2
  exit 1
fi
itype="${picked%% *}"
region="${picked##* }"
echo "Launching $itype in $region..."

# Billing starts the moment this POST lands on the server, so its failure mode is
# the expensive one. api() is fail-closed (curl --fail prints nothing on >=400), so
# a bare assignment would abort here under errexit with no id, no tracking file and
# no warning — and a POST can succeed server-side while its reply is lost (30 s
# timeout on a hardware-provisioning endpoint, reset, edge 502). Capture the exit
# code instead, same idiom as the pick step, and say plainly that a box may exist.
launch_status=0
launch_response="$(api POST /instance-operations/launch "{
  \"region_name\": \"$region\", \"instance_type_name\": \"$itype\",
  \"ssh_key_names\": [\"$SSH_KEY_NAME\"], \"quantity\": 1,
  \"name\": \"crosscourt-analysis\"}")" || launch_status=$?
if [ "$launch_status" -ne 0 ]; then
  echo "Launch request failed in transit — a box MAY have started and be billing." >&2
  echo "Run scripts/lambda/lambda_status.sh BEFORE retrying." >&2
  exit 1
fi
# json_field raises IndexError (not empty) on a 2xx whose instance_ids is empty, so
# guard this assignment too; the message below prints the whole response, which is
# strictly more useful than the traceback it replaces.
instance_id=""
instance_id="$(json_field "$launch_response" data.instance_ids.0 2>/dev/null)" || instance_id=""
if [ -z "$instance_id" ]; then
  echo "Launch response carried no instance id: $launch_response" >&2
  exit 1
fi
echo "Instance $instance_id launching (billing has started — lambda_down.sh stops it)."

# One clock read, shared by both writes below: launched_at means "when billing
# started", and the final write happens up to 15 min later, after provisioning.
launched_at="$(date +%s)"

# Provisional record, written before any further api call: the file must exist from
# the first billable moment so lambda_down.sh can always find the box even if this
# script dies mid-poll. The full write below overwrites it with the real ip + price.
CC_INSTANCE_FILE="$INSTANCE_FILE" CC_ID="$instance_id" CC_IP="" CC_TYPE="$itype" \
CC_REGION="$region" CC_PRICE_CENTS=0 CC_LAUNCHED_AT="$launched_at" \
python3 - <<'PY'
# Quoted heredoc + os.environ: shell values never become Python source, so a quote
# or backslash in one cannot break the write — and breaking *this* write is what the
# provisional record exists to prevent. tmp + os.replace so an interrupt mid-write
# leaves the old file, never a zero-byte one that every reader then chokes on.
import json, os
path = os.environ["CC_INSTANCE_FILE"]
with open(path + ".tmp", "w") as handle:
    json.dump({"id": os.environ["CC_ID"], "ip": os.environ["CC_IP"],
               "type": os.environ["CC_TYPE"], "region": os.environ["CC_REGION"],
               "price_cents": int(os.environ["CC_PRICE_CENTS"] or 0),
               "launched_at": int(os.environ["CC_LAUNCHED_AT"])}, handle)
os.replace(path + ".tmp", path)
PY

# --- wait for active + ip (15 min cap)
ip=""
status=""
for _ in $(seq 1 90); do
  sleep 10
  # Every parse in this loop is fail-open, and that is the point: the box is
  # ALREADY billing from here on, so one transient 502/timeout must not abort the
  # script — recovery would re-pay the provisioning minutes. An unreadable tick is
  # just "no news"; re-poll. (Same `|| echo '{}'` guard lambda_down.sh uses.)
  info="$(api GET "/instances/$instance_id" || echo '{}')"
  status=""; status="$(json_field "$info" data.status 2>/dev/null)" || status=""
  ip=""; ip="$(json_field "$info" data.ip 2>/dev/null)" || ip=""
  echo "  status=${status:-?} ip=${ip:-...}"
  if [ "$status" = "active" ] && [ -n "$ip" ]; then
    break
  fi
done
if [ "$status" != "active" ] || [ -z "$ip" ]; then
  echo "Instance not active after 15 min — check lambda_status.sh; terminate with lambda_down.sh." >&2
  exit 1
fi

# Same fail-open guard, same reason: the box is up and billing, and the price is
# only used for the cost lines. Losing it to a blip must not abort the session.
price_cents=""
price_cents="$(json_field "$(api GET /instance-types || echo '{}')" \
  "data.$itype.instance_type.price_cents_per_hour" 2>/dev/null)" || price_cents=""
if [ -z "$price_cents" ]; then
  echo "warning: could not read the $itype price — session cost lines will read \$0.00." >&2
  price_cents=0
fi
# Same env-fed, atomic write as the provisional one above; only ip/price_cents differ.
# launched_at is the value read at launch, not now — provisioning took up to 15 min.
CC_INSTANCE_FILE="$INSTANCE_FILE" CC_ID="$instance_id" CC_IP="$ip" CC_TYPE="$itype" \
CC_REGION="$region" CC_PRICE_CENTS="$price_cents" CC_LAUNCHED_AT="$launched_at" \
python3 - <<'PY'
import json, os
path = os.environ["CC_INSTANCE_FILE"]
with open(path + ".tmp", "w") as handle:
    json.dump({"id": os.environ["CC_ID"], "ip": os.environ["CC_IP"],
               "type": os.environ["CC_TYPE"], "region": os.environ["CC_REGION"],
               "price_cents": int(os.environ["CC_PRICE_CENTS"] or 0),
               "launched_at": int(os.environ["CC_LAUNCHED_AT"])}, handle)
os.replace(path + ".tmp", path)
PY

# --- wait for ssh, then bootstrap
INSTANCE_IP="$ip"
for _ in $(seq 1 30); do
  if box_ssh true 2>/dev/null; then break; fi
  sleep 10
done
scp "${SSH_OPTS[@]}" "$SCRIPT_DIR/bootstrap_remote.sh" "$REMOTE_USER@$ip:/tmp/bootstrap_remote.sh"
if ! box_ssh bash /tmp/bootstrap_remote.sh "$REF"; then
  echo "" >&2
  echo "Bootstrap FAILED. Instance left up for debugging (billing continues):" >&2
  # printf %q, not ${SSH_OPTS[*]}: the -i/-o arguments are paths that may contain
  # spaces, and this line exists to be copy-pasted into a shell.
  ssh_cmd="$(printf 'ssh '; printf '%q ' "${SSH_OPTS[@]}"; printf '%q' "$REMOTE_USER@$ip")"
  echo "  $ssh_cmd" >&2
  echo "  scripts/lambda/lambda_down.sh   # when done" >&2
  exit 1
fi

echo ""
# Nothing outside this terminal bounds a forgotten box: Lambda has no stopped
# state, so the meter runs from the launch POST until termination, and both the
# lambda_down sweep and lambda_status only help an operator who already
# remembered. Naming the wall-clock deadline and the overnight number here, while
# they are still watching, is the cheapest bound that exists today. (A local
# watchdog is the next increment — see the spec's failure-mode table.)
CC_PRICE_CENTS="$price_cents" CC_LAUNCHED_AT="$launched_at" python3 - <<'PY'
import os, time
price = int(os.environ["CC_PRICE_CENTS"] or 0) / 100
started = int(os.environ["CC_LAUNCHED_AT"])
clock = lambda offset: time.strftime("%H:%M", time.localtime(started + offset))
print(f"Billing since {clock(0)} local at ${price:.2f}/hr — "
      f"stop by {clock(3 * 3600)} or this session costs ${price * 3:.2f}.")
print(f"Left running overnight (24 h) it costs ${price * 24:.2f}. SET A TIMER NOW.")
PY
echo ""
echo "Ready. Next:"
echo "  scripts/lambda/lambda_run.sh <clip.mp4>     # analyze a clip"
echo "  scripts/lambda/lambda_tunnel.sh             # browse the box UI"
echo "  scripts/lambda/lambda_down.sh               # STOP BILLING when done"
