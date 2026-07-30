#!/usr/bin/env bash
# Interactive path: forward the box UI to http://localhost:5199 (local 5188
# stays free for the locally running Flask).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_instance
ensure_tunnel
echo "Box UI: http://localhost:$TUNNEL_PORT  (tunnel to $INSTANCE_IP:$REMOTE_PORT)"
echo "Tunnel persists in the background; lambda_down.sh cleans it up."
