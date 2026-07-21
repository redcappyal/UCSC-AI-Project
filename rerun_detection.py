"""Replay fusion detection over stored runs into a mirror dir for eval A/B.

Real run artifacts are never modified. Audio windows are not persisted in
run dirs, so replays run without audio evidence in BOTH arms - this
compares the trajectory + emission change, holding everything else fixed.

Usage:
  python rerun_detection.py --runs-dir ui_runs --out-dir /tmp/eval2d
  python rerun_detection.py --runs-dir ui_runs --out-dir /tmp/eval3d --use-3d
Then: python build_eval_set.py --runs-dir <out-dir> ... and eval_line_calls.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import court_model
from event_engine import detect_events_fused


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["source_frame"]))
    return rows


def replay_run(run_dir, out_dir, use_3d):
    rows = load_rows(run_dir / "ball_coordinates.csv")
    calibration = None
    calibration_path = run_dir / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera = camera_info = None
    if use_3d and calibration is not None:
        camera, camera_info = court_model.solve_camera_model(calibration)
    hits = detect_events_fused(rows, audio_windows=None,
                               calibration=calibration, camera=camera)
    for hit in hits:
        hit["frame"] = hit["hit_frame"]  # build_eval_set matches on "frame"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ground_truth.json", "corrections.json", "calibration.json"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    payload = {"hits": hits, "replay": {"use_3d": bool(use_3d and camera),
                                        "camera_info": camera_info}}
    (out_dir / "detected_hits.json").write_text(
        json.dumps(payload, default=float), encoding="utf-8")
    return len(hits), camera is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("ui_runs"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--use-3d", action="store_true")
    args = parser.parse_args()

    replayed, skipped = 0, []
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        has_labels = (run_dir / "ground_truth.json").exists() or \
                     (run_dir / "corrections.json").exists()
        if not has_labels or not (run_dir / "ball_coordinates.csv").exists():
            skipped.append(run_dir.name)
            continue
        try:
            count, solved = replay_run(run_dir, args.out_dir / run_dir.name,
                                       args.use_3d)
        except Exception as error:  # a bad run must not sink the sweep
            print(f"  ERROR {run_dir.name}: {error}")
            skipped.append(run_dir.name)
            continue
        print(f"  {run_dir.name}: {count} hits"
              + (" [3D]" if solved else " [2D]"))
        replayed += 1
    print(f"{replayed} runs replayed, {len(skipped)} skipped: {skipped}")


if __name__ == "__main__":
    main()
