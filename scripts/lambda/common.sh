#!/usr/bin/env bash
# Shared config + helpers for scripts/lambda/*. Source, don't execute.
set -euo pipefail

API_BASE="https://cloud.lambdalabs.com/api/v1"
CONFIG_DIR="$HOME/.config/crosscourt"
KEY_FILE="$CONFIG_DIR/lambda_api_key"
INSTANCE_FILE="$CONFIG_DIR/lambda_instance.json"
KNOWN_HOSTS="$CONFIG_DIR/lambda_known_hosts"
REPO_URL="https://github.com/redcappyal/UCSC-AI-Project"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_KEY_NAME="crosscourt-mac"
REMOTE_USER="ubuntu"
REMOTE_PORT=5188
TUNNEL_PORT="${CROSSCOURT_TUNNEL_PORT:-5199}"
PREFER_TYPES="gpu_1x_a6000,gpu_1x_a10,gpu_1x_a100,gpu_1x_a100_sxm4"
PRICE_CAP_CENTS=200

# Multiplex ssh so per-poll commands reuse one connection.
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$KNOWN_HOSTS" -o ConnectTimeout=10
  -o ControlMaster=auto -o ControlPath="$CONFIG_DIR/ssh-%r@%h" -o ControlPersist=10m)

api_key() {
  if [ -n "${LAMBDA_API_KEY:-}" ]; then
    printf '%s' "$LAMBDA_API_KEY"
  elif [ -f "$KEY_FILE" ]; then
    cat "$KEY_FILE"
  else
    echo "No Lambda API key: create $KEY_FILE (see scripts/lambda/README.md)" >&2
    return 1
  fi
}

api() { # api METHOD PATH [JSON_BODY]
  local method="$1" path="$2" body="${3:-}"
  # Assign on its own line: inside `local ...=$(api_key)` the declaration's own
  # exit status masks api_key's return 1, losing the missing-key diagnostic.
  local key
  key="$(api_key)" || return 1
  # --fail (not --fail-with-body): an HTTP error body on stdout would parse as an
  # empty account and print a false all-clear downstream.
  local args=(-sS --fail -m 30 -X "$method" -H "Authorization: Bearer $key")
  if [ -n "$body" ]; then
    args+=(-H "Content-Type: application/json" -d "$body")
  fi
  curl "${args[@]}" "$API_BASE$path"
}

json_field() { # json_field JSON dotted.path  (prints empty on missing)
  # JSON travels via stdin, never a Python literal: embedding the payload in
  # source would mangle \u escapes and choke on quotes.
  printf '%s' "$1" | python3 -c "
import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1].split('.'):
    if isinstance(value, list):
        value = value[int(key)]
    else:
        value = value.get(key, '') if isinstance(value, dict) else ''
print(value if value is not None else '')
" "$2"
}

require_instance() {
  if [ ! -f "$INSTANCE_FILE" ]; then
    echo "No tracked instance ($INSTANCE_FILE missing) — run scripts/lambda/lambda_up.sh first." >&2
    exit 1
  fi
  INSTANCE_ID="$(python3 -c "import json;print(json.load(open('$INSTANCE_FILE'))['id'])")"
  INSTANCE_IP="$(python3 -c "import json;print(json.load(open('$INSTANCE_FILE'))['ip'])")"
  export INSTANCE_ID INSTANCE_IP
}

box_ssh() { ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$INSTANCE_IP" "$@"; }

box_rsync() { # box_rsync SRC... DST  (caller writes user@host: prefixes)
  rsync -a --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}" "$@"
}

ensure_tunnel() {
  if nc -z 127.0.0.1 "$TUNNEL_PORT" 2>/dev/null; then
    return 0
  fi
  ssh "${SSH_OPTS[@]}" -f -N -o ExitOnForwardFailure=yes \
    -L "$TUNNEL_PORT:127.0.0.1:$REMOTE_PORT" "$REMOTE_USER@$INSTANCE_IP"
}
