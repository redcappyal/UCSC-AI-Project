#!/usr/bin/env bash
# Read-only: list every instance on the account and its burn rate.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

api GET /instances | python3 -c "
import json, sys
payload = json.load(sys.stdin)
# .get('data', []) would turn a 200 in an unexpected shape into 'nothing is
# billing' — and this line is the last word on whether money is burning (it is
# also what lambda_down.sh's EXIT trap prints). Show the payload and fail instead.
if not isinstance(payload, dict) or 'data' not in payload:
    print('Unexpected /instances response (no data key) — CANNOT say whether anything is billing:')
    print(json.dumps(payload)[:2000])
    print('Check https://cloud.lambda.ai directly.')
    sys.exit(1)
data = payload['data'] or []
if not data:
    print('Nothing is running — nothing is billing.')
for inst in data:
    itype = inst.get('instance_type', {})
    price = itype.get('price_cents_per_hour', 0) / 100
    print(f\"{inst.get('id','?'):36s} {itype.get('name','?'):18s} \"
          f\"{inst.get('region',{}).get('name','?'):12s} {inst.get('status','?'):10s} \"
          f\"ip={inst.get('ip') or '-':15s} \${price:.2f}/hr\")
"
