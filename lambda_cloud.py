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
