"""Seed silver rally labels for the rally-boundary eval axis.

Writes `eval_set/rally_labels.jsonl`, one row per labeled video:

    {"video_path", "video_sha", "fps", "rallies": [{"start_s","end_s"}],
     "provenance", "verified": false}

**These labels are silver and unverified.** They are produced by clustering
*human-labeled hit frames* into rallies on a fixed gap, so what they encode is
a human's hit labels under a clustering rule -- not a human's judgement of
where the rallies were. `verified: false` says so on every row, and the human
spot-check is a gate that lives in the design spec, not in this script.

They come from human labels rather than detector output on purpose. Seeding
from `detected_hits.json` would score the rally segmenter against the very
recall problem it was built to route around, and would flatter it exactly
where it matters least.

Two sources, in order of preference:

  1. `*_wall_hits.csv` with a `.meta.json` sidecar -- the offline labeler's
     output (label_hits.py). Real human labels. The sidecar is required, not
     optional: without the fps and the video sha the frame numbers are
     anonymous integers, which is the same point the README makes about the
     eval set as a whole.
  2. `ui_runs/*/` runs whose source video is still on disk, recomputing the
     hit-derived rallies. Only usable on a machine that has `ui_runs/`, which
     is gitignored -- absent from a fresh worktree.

Usage:

    .venv/Scripts/python.exe tools/seed_rally_labels.py
    .venv/Scripts/python.exe tools/seed_rally_labels.py --video-root /path/to/videos
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_rally_boundaries import cluster_hit_frames_into_rallies


def _read_hit_frames(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [
            int(row["hit_frame"])
            for row in csv.DictReader(handle)
            if row.get("hit_frame", "").strip()
        ]


def _resolve_video(recorded_path, video_root):
    """Where the labeled video actually lives on this machine, if anywhere.

    Sidecars record the absolute path from the machine that did the labeling
    -- typically a Mac. On any other machine that path will not exist, which
    is a fact to report rather than an error to raise.
    """
    candidate = Path(recorded_path)
    if candidate.exists():
        return candidate
    if video_root:
        local = Path(video_root) / candidate.name
        if local.exists():
            return local
    return candidate


def seed_from_label_csvs(repo_root, video_root=None):
    rows = []
    skipped = []
    for csv_path in sorted(repo_root.glob("*wall_hits.csv")):
        meta_path = csv_path.with_suffix(".meta.json")
        if not meta_path.exists():
            skipped.append(
                (csv_path.name, "no .meta.json sidecar - frame numbers index "
                                "into nothing without fps and a video sha")
            )
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fps = float(meta.get("fps") or 0)
        if fps <= 0:
            skipped.append((csv_path.name, "sidecar records no usable fps"))
            continue

        rallies = cluster_hit_frames_into_rallies(_read_hit_frames(csv_path), fps)
        if not rallies:
            skipped.append((csv_path.name, "no run of hits long enough to be a rally"))
            continue

        rows.append({
            "video_path": str(_resolve_video(meta.get("video_path", ""), video_root)),
            "video_sha": meta.get("video_sha"),
            "fps": fps,
            "labels_from": csv_path.name,
            "rallies": [
                {"start_s": round(r["start_s"], 3), "end_s": round(r["end_s"], 3)}
                for r in rallies
            ],
            "hit_count": sum(r["hit_count"] for r in rallies),
            "provenance": "silver-human-hit-clustered",
            "verified": False,
        })

    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval_set/rally_labels.jsonl")
    parser.add_argument(
        "--video-root",
        help="Directory holding the labeled videos on THIS machine, if the "
             "paths recorded in the sidecars point somewhere else.",
    )
    args = parser.parse_args()

    rows, skipped = seed_from_label_csvs(REPO_ROOT, args.video_root)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} labeled video(s) to {out_path}")
    for row in rows:
        present = "present" if Path(row["video_path"]).exists() else "NOT on this machine"
        print(f"  {row['labels_from']}: {len(row['rallies'])} rallies "
              f"from {row['hit_count']} human hits - video {present}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")


if __name__ == "__main__":
    main()
