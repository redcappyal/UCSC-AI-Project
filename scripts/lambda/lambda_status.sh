#!/usr/bin/env bash
# Read-only: list every instance on the account and its burn rate.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

api GET /instances | python3 -c "
import json, sys
data = json.load(sys.stdin).get('data', [])
if not data:
    print('Nothing is running — nothing is billing.')
for inst in data:
    itype = inst.get('instance_type', {})
    price = itype.get('price_cents_per_hour', 0) / 100
    print(f\"{inst.get('id','?'):36s} {itype.get('name','?'):18s} \"
          f\"{inst.get('region',{}).get('name','?'):12s} {inst.get('status','?'):10s} \"
          f\"ip={inst.get('ip') or '-':15s} \${price:.2f}/hr\")
"
