#!/usr/bin/env bash
# Launch one instance (A6000 -> A10 -> A100 under $2/hr), bootstrap, verify CUDA.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
REF="${1:-main}"

# --- one box at a time
if [ -f "$INSTANCE_FILE" ]; then
  existing_id="$(python3 -c "import json;print(json.load(open('$INSTANCE_FILE'))['id'])")"
  status="$(json_field "$(api GET "/instances/$existing_id" || echo '{}')" data.status)"
  if [ -n "$status" ] && [ "$status" != "terminated" ]; then
    echo "Instance $existing_id is already '$status' — use it, or lambda_down.sh first." >&2
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

launch_response="$(api POST /instance-operations/launch "{
  \"region_name\": \"$region\", \"instance_type_name\": \"$itype\",
  \"ssh_key_names\": [\"$SSH_KEY_NAME\"], \"quantity\": 1,
  \"name\": \"crosscourt-analysis\"}")"
instance_id="$(json_field "$launch_response" data.instance_ids.0)"
if [ -z "$instance_id" ]; then
  echo "Launch failed: $(json_field "$launch_response" error.message)" >&2
  exit 1
fi
echo "Instance $instance_id launching (billing has started — lambda_down.sh stops it)."

# --- wait for active + ip (15 min cap)
ip=""
for _ in $(seq 1 90); do
  sleep 10
  info="$(api GET "/instances/$instance_id")"
  status="$(json_field "$info" data.status)"
  ip="$(json_field "$info" data.ip)"
  echo "  status=$status ip=${ip:-...}"
  if [ "$status" = "active" ] && [ -n "$ip" ]; then
    break
  fi
done
if [ "$status" != "active" ] || [ -z "$ip" ]; then
  echo "Instance not active after 15 min — check lambda_status.sh; terminate with lambda_down.sh." >&2
  exit 1
fi

price_cents="$(json_field "$(api GET /instance-types)" "data.$itype.instance_type.price_cents_per_hour")"
python3 - <<PY
import json, time
json.dump({"id": "$instance_id", "ip": "$ip", "type": "$itype",
           "region": "$region", "price_cents": int("$price_cents" or 0),
           "launched_at": int(time.time())},
          open("$INSTANCE_FILE", "w"))
PY

# --- wait for ssh, then bootstrap
INSTANCE_IP="$ip"
for _ in $(seq 1 30); do
  if box_ssh true 2>/dev/null; then break; fi
  sleep 10
done
scp "${SSH_OPTS[@]}" "$(dirname "${BASH_SOURCE[0]}")/bootstrap_remote.sh" "$REMOTE_USER@$ip:/tmp/bootstrap_remote.sh"
if ! box_ssh bash /tmp/bootstrap_remote.sh "$REF"; then
  echo "" >&2
  echo "Bootstrap FAILED. Instance left up for debugging (billing continues):" >&2
  echo "  ssh ${SSH_OPTS[*]} $REMOTE_USER@$ip" >&2
  echo "  scripts/lambda/lambda_down.sh   # when done" >&2
  exit 1
fi

echo ""
echo "Ready. Next:"
echo "  scripts/lambda/lambda_run.sh <clip.mp4>     # analyze a clip"
echo "  scripts/lambda/lambda_tunnel.sh             # browse the box UI"
echo "  scripts/lambda/lambda_down.sh               # STOP BILLING when done"
