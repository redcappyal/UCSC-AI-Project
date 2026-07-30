# Lambda Cloud Compute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scripted, ephemeral Lambda GPU sessions that run the existing analysis
pipeline on CUDA and land results back in the local `ui_runs/`.

**Architecture:** A stdlib-only Python helper (`lambda_cloud.py`) holds the two
pieces of real logic (instance-type selection, default-calibration lookup) so
they are unit-testable; five thin bash scripts in `scripts/lambda/` do the
plumbing (Lambda REST API via curl, ssh/rsync, box-side curl against the box's
own localhost Flask). Nothing persists on Lambda between sessions.

**Tech Stack:** bash (macOS /bin/bash 3.2-compatible), curl, ssh/rsync,
Python 3 stdlib only for the helper, Lambda Cloud API v1, pytest.

**Spec:** `docs/superpowers/specs/2026-07-30-lambda-cloud-compute-design.md`

## Global Constraints

- GPU preference order, exact API names: `gpu_1x_a6000` → `gpu_1x_a10` →
  `gpu_1x_a100` → `gpu_1x_a100_sxm4`. Hard price cap **200 cents/hr**; never
  launch above it. US regions only (region names starting `us-`).
- Fully ephemeral: no Lambda filesystems; every session starts with launch +
  bootstrap and ends with terminate.
- Zero secrets on the box; repo cloned anonymously (public). The API key is
  read from `~/.config/crosscourt/lambda_api_key` (or `$LAMBDA_API_KEY`) and
  never printed, logged, or copied to the box.
- Flask on the box binds `127.0.0.1:5188` (app default — bootstrap must not
  set `HOST`). It is reached only via ssh (box-side curl) or the ssh tunnel.
- Tunnel LOCAL port is **5199** (remote 5188) so it never collides with a
  locally running Flask. (Spec update included in Task 5.)
- `lambda_cloud.py` imports stdlib only (no cv2/torch/flask) so it runs under
  plain `python3` on both the Mac and the box.
- Scripts must run under macOS bash 3.2: no `mapfile`, no `declare -A`, no
  `${var,,}`.
- Repo hook: editing `lambda_cloud.py` auto-runs `tests/test_lambda_cloud.py`
  (PostToolUse); a failing test comes back as a blocked edit.
- Launching bills Ian's card. Any step that launches a real instance requires
  his explicit go-ahead at that moment (Task 7 only).

## File Structure

- `lambda_cloud.py` (create, repo root) — selection + calibration logic, CLI.
- `tests/test_lambda_cloud.py` (create) — unit tests, no network.
- `scripts/lambda/common.sh` (create) — shared config/helpers, sourced by all.
- `scripts/lambda/lambda_status.sh` (create) — read-only account view.
- `scripts/lambda/bootstrap_remote.sh` (create) — runs ON the box.
- `scripts/lambda/lambda_up.sh` (create) — launch + bootstrap.
- `scripts/lambda/lambda_tunnel.sh` (create) — interactive UI tunnel.
- `scripts/lambda/lambda_run.sh` (create) — clip up → track → results back.
- `scripts/lambda/lambda_down.sh` (create) — terminate + sweep.
- `scripts/lambda/README.md` (create) — runbook.
- `docs/superpowers/specs/2026-07-30-lambda-cloud-compute-design.md` (modify,
  Task 5 only: tunnel port note).

---

### Task 1: `lambda_cloud.py` — instance-type selection

**Files:**
- Create: `lambda_cloud.py`
- Test: `tests/test_lambda_cloud.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `pick_instance_type(payload: dict, preferences: list, price_cap_cents: int, region_prefix: str = "us-") -> tuple | None` — `(type_name, region_name)` or `None`.
  - `availability_rows(payload: dict) -> list` — `(name, price_dollars: float, us_regions: list)` sorted by price, `gpu_1x_*` only.
  - CLI: `python3 lambda_cloud.py pick-type --prefer a,b,c --cap-cents 200` reads the `GET /instance-types` JSON on stdin; prints `"<name> <region>"` exit 0, or an availability table to stderr exit 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lambda_cloud.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import lambda_cloud

# Trimmed real response shape (live API, 2026-07-30).
FIXTURE = {
    "data": {
        "gpu_1x_a6000": {
            "instance_type": {"name": "gpu_1x_a6000", "price_cents_per_hour": 109},
            "regions_with_capacity_available": [],
        },
        "gpu_1x_a10": {
            "instance_type": {"name": "gpu_1x_a10", "price_cents_per_hour": 129},
            "regions_with_capacity_available": [
                {"name": "us-east-1"}, {"name": "us-west-1"},
            ],
        },
        "gpu_1x_a100": {
            "instance_type": {"name": "gpu_1x_a100", "price_cents_per_hour": 199},
            "regions_with_capacity_available": [{"name": "europe-central-1"}],
        },
        "gpu_1x_a100_sxm4": {
            "instance_type": {"name": "gpu_1x_a100_sxm4", "price_cents_per_hour": 199},
            "regions_with_capacity_available": [{"name": "us-east-1"}],
        },
        "gpu_1x_h100_pcie": {
            "instance_type": {"name": "gpu_1x_h100_pcie", "price_cents_per_hour": 329},
            "regions_with_capacity_available": [{"name": "us-west-3"}],
        },
        "gpu_8x_a100": {
            "instance_type": {"name": "gpu_8x_a100", "price_cents_per_hour": 1592},
            "regions_with_capacity_available": [{"name": "us-east-1"}],
        },
    }
}

PREFS = ["gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_a100", "gpu_1x_a100_sxm4"]


def test_pick_prefers_earlier_entries_with_us_capacity():
    # a6000 has no capacity -> a10 (which does) wins over the a100s.
    assert lambda_cloud.pick_instance_type(FIXTURE, PREFS, 200) == (
        "gpu_1x_a10", "us-east-1",
    )


def test_pick_takes_first_preference_when_available():
    payload = json.loads(json.dumps(FIXTURE))
    payload["data"]["gpu_1x_a6000"]["regions_with_capacity_available"] = [
        {"name": "us-west-1"},
    ]
    assert lambda_cloud.pick_instance_type(payload, PREFS, 200) == (
        "gpu_1x_a6000", "us-west-1",
    )


def test_pick_ignores_non_us_regions():
    # gpu_1x_a100 has only europe capacity; sxm4 has us-east-1.
    payload = json.loads(json.dumps(FIXTURE))
    payload["data"]["gpu_1x_a10"]["regions_with_capacity_available"] = []
    assert lambda_cloud.pick_instance_type(payload, PREFS, 200) == (
        "gpu_1x_a100_sxm4", "us-east-1",
    )


def test_pick_respects_price_cap():
    # Cap below every candidate -> None, even with capacity present.
    assert lambda_cloud.pick_instance_type(FIXTURE, PREFS, 100) is None


def test_pick_returns_none_when_nothing_available():
    payload = json.loads(json.dumps(FIXTURE))
    for entry in payload["data"].values():
        entry["regions_with_capacity_available"] = []
    assert lambda_cloud.pick_instance_type(payload, PREFS, 200) is None


def test_pick_never_selects_types_outside_preferences():
    # h100 has US capacity and would fit a high cap; it is not in PREFS.
    assert lambda_cloud.pick_instance_type(FIXTURE, PREFS, 400) == (
        "gpu_1x_a10", "us-east-1",
    )


def test_availability_rows_sorted_by_price_single_gpu_only():
    rows = lambda_cloud.availability_rows(FIXTURE)
    names = [name for name, _, _ in rows]
    assert names == [
        "gpu_1x_a6000", "gpu_1x_a10", "gpu_1x_a100", "gpu_1x_a100_sxm4",
        "gpu_1x_h100_pcie",
    ]
    a10 = rows[1]
    assert a10[1] == 1.29
    assert a10[2] == ["us-east-1", "us-west-1"]


def test_cli_pick_type_success(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(Path(lambda_cloud.__file__)), "pick-type",
         "--prefer", ",".join(PREFS), "--cap-cents", "200"],
        input=json.dumps(FIXTURE), capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "gpu_1x_a10 us-east-1"


def test_cli_pick_type_no_capacity_exit_2():
    payload = json.loads(json.dumps(FIXTURE))
    for entry in payload["data"].values():
        entry["regions_with_capacity_available"] = []
    proc = subprocess.run(
        [sys.executable, str(Path(lambda_cloud.__file__)), "pick-type",
         "--prefer", ",".join(PREFS), "--cap-cents", "200"],
        input=json.dumps(payload), capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "gpu_1x_a6000" in proc.stderr  # availability table printed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lambda_cloud.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lambda_cloud'`
(Run from the main checkout, which has `.venv`; in a worktree without a venv
use the main repo's `.venv/bin/python` by absolute path.)

- [ ] **Step 3: Implement selection + CLI**

Create `lambda_cloud.py`:

```python
"""Lambda Cloud helpers for scripts/lambda/*.

Stdlib only — runs under plain python3 on the Mac and on the box. Holds the
logic worth unit-testing; the bash scripts stay plumbing.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def _entry_type(entry):
    # Real responses nest under "instance_type"; tolerate a flat entry.
    return entry.get("instance_type", entry)


def _us_regions(entry, region_prefix):
    return [
        r["name"]
        for r in entry.get("regions_with_capacity_available", [])
        if r.get("name", "").startswith(region_prefix)
    ]


def pick_instance_type(payload, preferences, price_cap_cents, region_prefix="us-"):
    """First preferred type with capacity in a matching region under the cap.

    Preference order is authoritative: a cheaper or better-stocked type that
    appears later in `preferences` never beats an earlier one that has any
    qualifying region. Types not listed in `preferences` are never chosen.
    """
    data = payload.get("data", {})
    for name in preferences:
        entry = data.get(name)
        if entry is None:
            continue
        if int(_entry_type(entry).get("price_cents_per_hour", 1 << 30)) > price_cap_cents:
            continue
        regions = _us_regions(entry, region_prefix)
        if regions:
            return name, regions[0]
    return None


def availability_rows(payload, region_prefix="us-"):
    """(name, $/hr, matching regions) for gpu_1x_* types, cheapest first."""
    rows = []
    for name, entry in payload.get("data", {}).items():
        if not name.startswith("gpu_1x_"):
            continue
        price = int(_entry_type(entry).get("price_cents_per_hour", 0)) / 100.0
        rows.append((name, price, _us_regions(entry, region_prefix)))
    return sorted(rows, key=lambda row: (row[1], row[0]))


def file_sha256(path, chunk_bytes=1024 * 1024):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def find_default_calibration(clip_path, ui_runs_dir):
    """Newest local run of the same video (by sha256) -> its calibration.json.

    The by-hash upload store names videos `<sha256><ext>` and /api/track
    records that path in job.json, so "same video" is an exact substring match
    on the basename. Returns a Path or None.
    """
    digest = file_sha256(clip_path)
    ui_runs = Path(ui_runs_dir)
    if not ui_runs.is_dir():
        return None
    run_dirs = sorted(
        (d for d in ui_runs.iterdir() if (d / "job.json").exists()),
        key=lambda d: d.name, reverse=True,
    )
    for run_dir in run_dirs:
        try:
            job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        video_path = job.get("video_path", "")
        if digest in Path(video_path).name:
            calibration = run_dir / "calibration.json"
            if calibration.exists():
                return calibration
    return None


def _cmd_pick_type(args):
    payload = json.load(sys.stdin)
    picked = pick_instance_type(
        payload, args.prefer.split(","), args.cap_cents, args.region_prefix,
    )
    if picked is None:
        print(
            f"No preferred instance type available under "
            f"{args.cap_cents / 100:.2f} $/hr in '{args.region_prefix}*' regions:",
            file=sys.stderr,
        )
        for name, price, regions in availability_rows(payload, args.region_prefix):
            print(
                f"  {name:24s} ${price:5.2f}/hr  {regions or 'no capacity'}",
                file=sys.stderr,
            )
        return 2
    print(f"{picked[0]} {picked[1]}")
    return 0


def _cmd_find_calibration(args):
    calibration = find_default_calibration(args.clip, args.ui_runs)
    if calibration is None:
        print(
            f"No local run of this video found under {args.ui_runs}; "
            f"pass --calibration explicitly.",
            file=sys.stderr,
        )
        return 2
    print(calibration)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pick = sub.add_parser("pick-type", help="choose instance type from stdin JSON")
    pick.add_argument("--prefer", required=True, help="comma-separated type names, in order")
    pick.add_argument("--cap-cents", type=int, required=True)
    pick.add_argument("--region-prefix", default="us-")
    pick.set_defaults(func=_cmd_pick_type)

    cal = sub.add_parser("find-calibration", help="default calibration for a clip")
    cal.add_argument("clip")
    cal.add_argument("--ui-runs", default="ui_runs")
    cal.set_defaults(func=_cmd_find_calibration)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_lambda_cloud.py -q`
Expected: `9 passed` (the `find_default_calibration` tests come in Task 2; at
this point only the 9 selection/CLI tests above exist).

- [ ] **Step 5: Commit**

```bash
git add lambda_cloud.py tests/test_lambda_cloud.py
git commit -m "feat(cloud): instance-type selection helper for Lambda scripts"
```

---

### Task 2: `lambda_cloud.py` — default-calibration lookup

**Files:**
- Modify: `lambda_cloud.py` (already contains the implementation from Task 1 —
  this task adds only tests; implementation and tests were co-designed, but a
  reviewer gates them separately: selection logic vs filesystem lookup logic)
- Test: `tests/test_lambda_cloud.py`

**Interfaces:**
- Consumes: `file_sha256`, `find_default_calibration` (Task 1 file).
- Produces: verified behavior other tasks rely on: `find-calibration` CLI
  prints the calibration path (exit 0) or exit 2 with a stderr hint.

- [ ] **Step 1: Write the failing-or-passing tests (append to `tests/test_lambda_cloud.py`)**

```python
def _make_run(ui_runs, run_id, video_name, with_calibration=True):
    run_dir = ui_runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "job.json").write_text(
        json.dumps({"video_path": f"/anywhere/by-hash/{video_name}"}),
        encoding="utf-8",
    )
    if with_calibration:
        (run_dir / "calibration.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_find_calibration_newest_matching_run_wins(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    digest = lambda_cloud.file_sha256(clip)
    ui_runs = tmp_path / "ui_runs"
    _make_run(ui_runs, "1753900000000", f"{digest}.mp4")
    newest = _make_run(ui_runs, "1753990000000", f"{digest}.mp4")
    _make_run(ui_runs, "1753995000000", "0123deadbeef.mp4")  # other video, newer
    found = lambda_cloud.find_default_calibration(clip, ui_runs)
    assert found == newest / "calibration.json"


def test_find_calibration_skips_runs_without_calibration(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    digest = lambda_cloud.file_sha256(clip)
    ui_runs = tmp_path / "ui_runs"
    older = _make_run(ui_runs, "1753900000000", f"{digest}.mp4")
    _make_run(ui_runs, "1753990000000", f"{digest}.mp4", with_calibration=False)
    assert lambda_cloud.find_default_calibration(clip, ui_runs) == older / "calibration.json"


def test_find_calibration_none_when_no_match(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    ui_runs = tmp_path / "ui_runs"
    _make_run(ui_runs, "1753900000000", "0123deadbeef.mp4")
    assert lambda_cloud.find_default_calibration(clip, ui_runs) is None


def test_cli_find_calibration_exit_2_when_missing(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    proc = subprocess.run(
        [sys.executable, str(Path(lambda_cloud.__file__)), "find-calibration",
         str(clip), "--ui-runs", str(tmp_path / "ui_runs")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "--calibration" in proc.stderr
```

- [ ] **Step 2: Run the full helper test file**

Run: `.venv/bin/python -m pytest tests/test_lambda_cloud.py -q`
Expected: `13 passed`. If any of the four new tests fail, fix
`find_default_calibration` (not the tests) until green.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: previous green count + 14 new, same 1 skipped / 1 deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/test_lambda_cloud.py
git commit -m "test(cloud): cover default-calibration lookup"
```

---

### Task 3: `common.sh` + `lambda_status.sh`

**Files:**
- Create: `scripts/lambda/common.sh`
- Create: `scripts/lambda/lambda_status.sh`

**Interfaces:**
- Consumes: `~/.config/crosscourt/lambda_api_key` (exists, chmod 600).
- Produces (sourced by Tasks 4–6):
  - vars: `API_BASE`, `CONFIG_DIR`, `KEY_FILE`, `INSTANCE_FILE`, `REPO_URL`,
    `REPO_ROOT`, `SSH_KEY`, `SSH_OPTS` (array), `TUNNEL_PORT` (default 5199,
    env-overridable via `CROSSCOURT_TUNNEL_PORT`), `REMOTE_PORT=5188`,
    `PREFER_TYPES`, `PRICE_CAP_CENTS=200`.
  - functions: `api METHOD PATH [JSON_BODY]` (curl, auth header, 30s timeout),
    `json_field JSON_STRING DOTTED.PATH` (python3 one-liner),
    `require_instance` (loads `INSTANCE_ID`/`INSTANCE_IP` from
    `INSTANCE_FILE` or exits 1 with "run lambda_up.sh first"),
    `box_ssh CMD...` / `box_rsync SRC DST` (ssh/rsync with `SSH_OPTS` +
    ControlMaster multiplexing), `ensure_tunnel` (idempotent
    `ssh -f -N -L $TUNNEL_PORT:127.0.0.1:$REMOTE_PORT`).

- [ ] **Step 1: Write `scripts/lambda/common.sh`**

```bash
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
  local args=(-sS -m 30 -X "$method" -H "Authorization: Bearer $(api_key)")
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
```

- [ ] **Step 2: Write `scripts/lambda/lambda_status.sh`**

```bash
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
```

- [ ] **Step 3: Syntax-check both and make executable**

Run: `bash -n scripts/lambda/common.sh && bash -n scripts/lambda/lambda_status.sh && chmod +x scripts/lambda/lambda_status.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Live read-only verification (free, no launch)**

Run: `scripts/lambda/lambda_status.sh`
Expected: `Nothing is running — nothing is billing.` (touches only
`GET /instances` with the saved key).

- [ ] **Step 5: Commit**

```bash
git add scripts/lambda/common.sh scripts/lambda/lambda_status.sh
git commit -m "feat(cloud): shared lambda helpers + read-only status script"
```

---

### Task 4: `bootstrap_remote.sh` + `lambda_up.sh`

**Files:**
- Create: `scripts/lambda/bootstrap_remote.sh`
- Create: `scripts/lambda/lambda_up.sh`

**Interfaces:**
- Consumes: `common.sh` vars/functions (Task 3), `lambda_cloud.py pick-type`
  (Task 1).
- Produces: a running, bootstrapped instance; `$INSTANCE_FILE` JSON with keys
  `id`, `ip`, `type`, `region`, `price_cents`, `launched_at` (epoch seconds)
  — `lambda_run.sh`/`lambda_down.sh` read exactly these keys.

- [ ] **Step 1: Write `scripts/lambda/bootstrap_remote.sh`** (runs ON the box)

```bash
#!/usr/bin/env bash
# Runs on the Lambda instance as ubuntu. Idempotent.
set -euo pipefail
REF="${1:-main}"

cd "$HOME"
if [ ! -d UCSC-AI-Project ]; then
  git clone --branch "$REF" https://github.com/redcappyal/UCSC-AI-Project
fi
cd UCSC-AI-Project
git fetch origin "$REF" && git checkout "$REF" && git pull --ff-only origin "$REF" || true

# Reuse Lambda Stack's CUDA torch; requirements.txt leaves torch unpinned so
# pip sees it satisfied. rfdetr==1.8.3 could still resolve its own torch —
# the CUDA gate below catches that.
if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo "--- CUDA gate"
.venv/bin/python - <<'PY'
import time
import numpy as np
import torch
assert torch.cuda.is_available(), (
    "CUDA unavailable — torch was clobbered by a dependency. Fix: "
    ".venv/bin/pip install --force-reinstall torch")
import ball_model
runner = ball_model.load_detector()
device = getattr(runner, "device", "cpu")
assert device == "cuda", f"ball model landed on {device!r}, expected cuda"
tile = (np.random.rand(416, 416, 9) * 255).astype("uint8")
runner.run_batch([tile])  # warmup pays graph optimization
start = time.perf_counter()
runner.run_batch([tile])
print(f"CUDA gate OK: {(time.perf_counter() - start) * 1000:.1f} ms/tile on {torch.cuda.get_device_name(0)}")
PY

echo "--- person detector pre-warm (non-fatal)"
.venv/bin/python -c "import person_model; d = person_model.load_person_detector(); print('person detector:', 'ready' if d else 'unavailable')" || true

echo "--- flask"
pkill -f "python app.py" 2>/dev/null || true
sleep 1
nohup .venv/bin/python app.py > "$HOME/flask.log" 2>&1 &
for _ in $(seq 1 20); do
  if curl -sf http://127.0.0.1:5188/api/health > /dev/null; then
    echo "Flask up on 127.0.0.1:5188 (log: ~/flask.log)"
    exit 0
  fi
  sleep 1
done
echo "Flask did not come up; tail ~/flask.log:" >&2
tail -20 "$HOME/flask.log" >&2
exit 1
```

- [ ] **Step 2: Write `scripts/lambda/lambda_up.sh`**

```bash
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
keys_response="$(api GET /ssh-keys)"
registered="$(python3 -c "
import json
names = [k['name'] for k in json.loads('''$keys_response''').get('data', [])]
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
picked="$(api GET /instance-types | python3 "$REPO_ROOT/lambda_cloud.py" pick-type \
  --prefer "$PREFER_TYPES" --cap-cents "$PRICE_CAP_CENTS")" || {
  echo "No capacity under the cap right now; try again later or ask Ian about raising it." >&2
  exit 2
}
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
```

- [ ] **Step 3: Syntax-check + executable bits**

Run: `bash -n scripts/lambda/bootstrap_remote.sh && bash -n scripts/lambda/lambda_up.sh && chmod +x scripts/lambda/bootstrap_remote.sh scripts/lambda/lambda_up.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Live read-only check of the selection plumbing (no launch)**

Run:
```bash
bash -c 'source scripts/lambda/common.sh && api GET /instance-types | python3 "$REPO_ROOT/lambda_cloud.py" pick-type --prefer "$PREFER_TYPES" --cap-cents "$PRICE_CAP_CENTS"'
```
Expected: one line like `gpu_1x_a10 us-east-1` (exit 0), or the availability
table on stderr (exit 2) if nothing qualifies right now. Either outcome proves
the key, the API call, and the selection pipe end-to-end using only the free
read-only endpoint.
**Do NOT run lambda_up.sh in this task** — launching bills the card and is
Task 7's gated step.

- [ ] **Step 5: Commit**

```bash
git add scripts/lambda/bootstrap_remote.sh scripts/lambda/lambda_up.sh
git commit -m "feat(cloud): lambda_up launch + remote bootstrap with CUDA gate"
```

---

### Task 5: `lambda_tunnel.sh` + `lambda_run.sh` (+ spec port note)

**Files:**
- Create: `scripts/lambda/lambda_tunnel.sh`
- Create: `scripts/lambda/lambda_run.sh`
- Modify: `docs/superpowers/specs/2026-07-30-lambda-cloud-compute-design.md`
  (tunnel section: local port 5199, reason: local Flask owns 5188)

**Interfaces:**
- Consumes: `common.sh` (`require_instance`, `box_ssh`, `box_rsync`,
  `ensure_tunnel`, `TUNNEL_PORT`), `lambda_cloud.py find-calibration`,
  `$INSTANCE_FILE` keys `price_cents`/`launched_at`.
- Produces: completed runs synced into local `ui_runs/<run_id>/`.

- [ ] **Step 1: Write `scripts/lambda/lambda_tunnel.sh`**

```bash
#!/usr/bin/env bash
# Interactive path: forward the box UI to http://localhost:5199 (local 5188
# stays free for the locally running Flask).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_instance
ensure_tunnel
echo "Box UI: http://localhost:$TUNNEL_PORT  (tunnel to $INSTANCE_IP:$REMOTE_PORT)"
echo "Tunnel persists in the background; lambda_down.sh cleans it up."
```

- [ ] **Step 2: Write `scripts/lambda/lambda_run.sh`**

```bash
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
    status="$(json_field "$response" status)"
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
```

- [ ] **Step 3: Syntax-check + executable bits**

Run: `bash -n scripts/lambda/lambda_tunnel.sh && bash -n scripts/lambda/lambda_run.sh && chmod +x scripts/lambda/lambda_tunnel.sh scripts/lambda/lambda_run.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Update the spec's tunnel port**

In `docs/superpowers/specs/2026-07-30-lambda-cloud-compute-design.md`, in the
`lambda_tunnel.sh` component section, replace
`ssh -N -L 5188:127.0.0.1:5188 ubuntu@<ip>` and `http://localhost:5188` with
`ssh -N -L 5199:127.0.0.1:5188 ubuntu@<ip>` and `http://localhost:5199`, and
append this sentence to that paragraph: "Local port 5199 (not 5188) so the
tunnel never collides with a locally running Flask; override with
`CROSSCOURT_TUNNEL_PORT`."

- [ ] **Step 5: Verify the public_job response actually carries `run_id`**

Run: `grep -n '"run_id"' app.py | head -3`
Expected: at least one hit inside `public_job` (the /api/track response body).
If absent, find the actual key with `grep -n "def public_job" -A 20 app.py`
and adjust the `json_field "$track_response" run_id` line to match.

- [ ] **Step 6: Commit**

```bash
git add scripts/lambda/lambda_tunnel.sh scripts/lambda/lambda_run.sh docs/superpowers/specs/2026-07-30-lambda-cloud-compute-design.md
git commit -m "feat(cloud): lambda_run clip pipeline + UI tunnel on local 5199"
```

---

### Task 6: `lambda_down.sh` + runbook

**Files:**
- Create: `scripts/lambda/lambda_down.sh`
- Create: `scripts/lambda/README.md`

**Interfaces:**
- Consumes: `common.sh`, `$INSTANCE_FILE`.
- Produces: terminated instance, removed `$INSTANCE_FILE`, account sweep.

- [ ] **Step 1: Write `scripts/lambda/lambda_down.sh`**

```bash
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
    [ -z "$status" ] || [ "$status" = "terminated" ] && break
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
```

- [ ] **Step 2: Write `scripts/lambda/README.md`**

```markdown
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
    scripts/lambda/lambda_up.sh            # launch + bootstrap (~5 min, billing starts)
    scripts/lambda/lambda_run.sh clip.mp4  # rsync up, track on GPU, results into ui_runs/
    scripts/lambda/lambda_tunnel.sh        # optional: box UI at http://localhost:5199
    scripts/lambda/lambda_down.sh          # terminate = the ONLY thing that stops billing
    scripts/lambda/lambda_status.sh        # anytime: what is billing right now?

- GPU order: A6000 ($1.09/hr) → A10 ($1.29) → A100 ($1.99); hard cap $2/hr.
  No capacity → lambda_up prints the availability table and exits.
- `lambda_run.sh` defaults: full clip, server-default stride (4), calibration
  from your newest local run of the same video (`--calibration` to override,
  `--stride`, `--start`, `--end` available).
- Mac slept / tunnel dropped mid-run? The job keeps running on the box:
  `lambda_run.sh --resume <run_id>`.
- Bootstrap failure leaves the box up for debugging — it is still billing;
  `lambda_down.sh` when done.
- The box's Flask listens on 127.0.0.1 only. Never start it with HOST=0.0.0.0
  on a cloud box: the app has no auth.
- The box holds zero secrets; the repo is public. OPENAI/ROBOFLOW keys stay
  on the Mac.
```

- [ ] **Step 3: Syntax-check + executable bit**

Run: `bash -n scripts/lambda/lambda_down.sh && chmod +x scripts/lambda/lambda_down.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: Full suite still green**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: same green count as end of Task 2 (scripts are not collected).

- [ ] **Step 5: Commit**

```bash
git add scripts/lambda/lambda_down.sh scripts/lambda/README.md
git commit -m "feat(cloud): lambda_down terminate+sweep and session runbook"
```

---

### Task 7: Live verification session (GATED — bills the card)

**Files:** none created; measurements recorded in `scripts/lambda/README.md`.

**Interfaces:**
- Consumes: everything above, Ian's explicit go-ahead **at execution time**.

- [ ] **Step 1: Get the explicit go-ahead**

Ask Ian: "Launching now will start billing (~$1.09–1.99/hr, expected total
$1–3 for this verification). Go?" Do not proceed on silence.

- [ ] **Step 2: Launch and bootstrap**

Run: `scripts/lambda/lambda_up.sh`
Expected: type/region line, launch id, status polls to `active`, bootstrap
output ending `CUDA gate OK: <n> ms/tile on <GPU name>` and
`Flask up on 127.0.0.1:5188`. Record the ms/tile number.

- [ ] **Step 3: Smoke clip (15 MB)**

Run: `scripts/lambda/lambda_run.sh "ui_runs/uploads/by-hash/7698ab13e87985d57e551bb81d8455f63427621d20b126bb315e4c20f6b53a21.mov"`
Expected: upload, `Tracking started: run <id>`, progress to `complete`, sync,
cost line. Then open the local app (`.venv/bin/python app.py`, existing
workflow) and confirm the new run appears with hits in the report.

- [ ] **Step 4: One real 4K capture**

Run: `scripts/lambda/lambda_run.sh "ui_runs/uploads/by-hash/91c71e99f769ba3e43903a6e3a422da0c12a466d43b43e21301110edc92891a6.mp4"`
Expected: completes end-to-end; note wall-clock minutes and $ from the cost
line. Sanity-compare hit count against any previous local run of the same
video (`ls ui_runs/*/detected_hits.json` + the report view).

- [ ] **Step 5: Tear down and verify nothing bills**

Run: `scripts/lambda/lambda_down.sh`
Expected: `Terminated. Session: ...` then the sweep prints
`Nothing is running — nothing is billing.`

- [ ] **Step 6: Record results**

Append measured numbers (ms/tile, wall clock + cost for both clips) to
`scripts/lambda/README.md` under a `## Measured 2026-07-30` heading; commit:

```bash
git add scripts/lambda/README.md
git commit -m "docs(cloud): record first live session measurements"
```
