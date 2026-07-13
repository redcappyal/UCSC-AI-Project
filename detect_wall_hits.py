import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_BALL_CSV = Path(__file__).with_name("ball_coordinates.csv")
DEFAULT_OUTPUT_CSV = Path(__file__).with_name("detected_hits.csv")
MAX_GAP_FRAMES = 3
MAX_JUMP_PX_PER_FRAME = 200.0
SMOOTH_WINDOW = 3
MIN_GAP_FRAMES = 10
TOP_K = 20
REQUIRED_COLUMNS = ("source_frame", "timestamp_seconds", "detected", "x_center", "y_center")


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_detected_positions(csv_path):
    frames = []
    timestamps = []
    positions = []
    duplicate_count = 0

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

        for row in reader:
            if not parse_bool(row["detected"]):
                continue
            if not row["x_center"] or not row["y_center"]:
                continue

            position = (float(row["x_center"]), float(row["y_center"]))

            # Videos with duplicated frames repeat the exact same box; keeping the
            # repeats would make the velocity alternate between zero and double.
            if positions and position == positions[-1]:
                duplicate_count += 1
                continue

            frames.append(int(row["source_frame"]))
            timestamps.append(float(row["timestamp_seconds"]))
            positions.append(position)

    return (
        np.array(frames, dtype=np.int64),
        np.array(timestamps, dtype=np.float64),
        np.array(positions, dtype=np.float64),
        duplicate_count,
    )


def split_into_tracks(frames, positions, max_gap, max_jump):
    if len(frames) == 0:
        return []

    frame_deltas = np.diff(frames)
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    # A displacement far larger than the ball could travel in the elapsed frames
    # means the detector switched to a different object, not that the ball moved.
    teleported = displacements / frame_deltas > max_jump
    break_points = np.where((frame_deltas > max_gap) | teleported)[0] + 1
    starts = np.concatenate([[0], break_points])
    ends = np.concatenate([break_points, [len(frames)]])
    return list(zip(starts, ends))


def smooth_positions(positions, window):
    if window <= 1 or len(positions) < window:
        return positions

    kernel = np.ones(window) / window
    smoothed = np.empty_like(positions)
    half = window // 2

    for axis in range(positions.shape[1]):
        padded = np.pad(positions[:, axis], half, mode="edge")
        smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")

    return smoothed


def compute_candidates(frames, timestamps, positions, tracks, smooth_window):
    candidates = []

    for start, end in tracks:
        track_frames = frames[start:end]
        track_times = timestamps[start:end]
        track_positions = smooth_positions(positions[start:end], smooth_window)

        if len(track_frames) < 3:
            continue

        after_gap = start > 0

        dt = np.diff(track_times)[:, np.newaxis]
        velocities = np.diff(track_positions, axis=0) / dt
        delta_v = np.diff(velocities, axis=0)
        dv_magnitudes = np.linalg.norm(delta_v, axis=1)

        for i, dv_magnitude in enumerate(dv_magnitudes):
            sample = i + 1
            candidates.append(
                {
                    "hit_frame": int(track_frames[sample]),
                    "timestamp_seconds": float(track_times[sample]),
                    "dv_magnitude": float(dv_magnitude),
                    "speed_before": float(np.linalg.norm(velocities[sample - 1])),
                    "speed_after": float(np.linalg.norm(velocities[sample])),
                    "after_gap": after_gap and sample <= 2,
                }
            )

    return candidates


def pick_peaks(candidates, min_gap, top_k, threshold):
    ranked = sorted(candidates, key=lambda c: c["dv_magnitude"], reverse=True)
    picked = []

    for candidate in ranked:
        if threshold is not None and candidate["dv_magnitude"] < threshold:
            break
        if any(abs(candidate["hit_frame"] - p["hit_frame"]) < min_gap for p in picked):
            continue

        picked.append(candidate)
        if threshold is None and len(picked) >= top_k:
            break

    return picked


def detect_hits(
    csv_path,
    *,
    max_gap=MAX_GAP_FRAMES,
    max_jump=MAX_JUMP_PX_PER_FRAME,
    smooth=SMOOTH_WINDOW,
    min_gap=MIN_GAP_FRAMES,
    top_k=TOP_K,
    threshold=None,
):
    frames, timestamps, positions, _ = load_detected_positions(csv_path)
    tracks = split_into_tracks(frames, positions, max_gap, max_jump)
    candidates = compute_candidates(frames, timestamps, positions, tracks, smooth)
    hits = pick_peaks(candidates, min_gap, top_k, threshold)
    return sorted(hits, key=lambda hit: hit["hit_frame"])


def save_hits(output_path, hits):
    fieldnames = [
        "hit_frame",
        "timestamp_seconds",
        "dv_magnitude",
        "speed_before",
        "speed_after",
        "after_gap",
    ]
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for hit in sorted(hits, key=lambda h: h["hit_frame"]):
            writer.writerow(
                {
                    **hit,
                    "timestamp_seconds": f"{hit['timestamp_seconds']:.6f}",
                    "dv_magnitude": f"{hit['dv_magnitude']:.3f}",
                    "speed_before": f"{hit['speed_before']:.3f}",
                    "speed_after": f"{hit['speed_after']:.3f}",
                }
            )


def print_hits(hits):
    header = f"{'rank':>4}  {'frame':>8}  {'time (s)':>10}  {'|dv| px/s':>12}  {'v before':>10}  {'v after':>10}  after_gap"
    print(header)
    print("-" * len(header))
    for rank, hit in enumerate(hits, start=1):
        print(
            f"{rank:>4}  {hit['hit_frame']:>8}  {hit['timestamp_seconds']:>10.3f}  "
            f"{hit['dv_magnitude']:>12.1f}  {hit['speed_before']:>10.1f}  "
            f"{hit['speed_after']:>10.1f}  {'yes' if hit['after_gap'] else ''}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Find candidate wall-hit frames by ranking the largest changes in the "
            "ball's per-frame velocity vector."
        )
    )
    parser.add_argument(
        "--ball-csv",
        type=Path,
        default=DEFAULT_BALL_CSV,
        help=f"CSV from modelEval.py. Defaults to {DEFAULT_BALL_CSV.name}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output hit CSV. Defaults to {DEFAULT_OUTPUT_CSV.name}.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=MAX_GAP_FRAMES,
        help="Largest detection gap (in frames) allowed inside one track.",
    )
    parser.add_argument(
        "--max-jump",
        type=float,
        default=MAX_JUMP_PX_PER_FRAME,
        help=(
            "Largest plausible ball movement in pixels per elapsed frame. "
            "Bigger jumps are treated as detector mistakes and split the track."
        ),
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTH_WINDOW,
        help="Moving-average window for positions. 1 disables smoothing.",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=MIN_GAP_FRAMES,
        help="Minimum frame spacing between reported hits.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="How many hits to report when no threshold is given.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Report every hit with |dv| at or above this value instead of the top k.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    frames, timestamps, positions, duplicate_count = load_detected_positions(args.ball_csv)
    tracks = split_into_tracks(frames, positions, args.max_gap, args.max_jump)
    candidates = compute_candidates(frames, timestamps, positions, tracks, args.smooth)
    hits = pick_peaks(candidates, args.min_gap, args.top_k, args.threshold)

    print(f"Detected samples: {len(frames)} (dropped {duplicate_count} consecutive duplicate position(s))")
    print(f"Tracks (split at gaps > {args.max_gap} frames or jumps > {args.max_jump:g} px/frame): {len(tracks)}")
    print(f"Candidate samples with a velocity change: {len(candidates)}")
    print()

    if not hits:
        print("No hits found. Try lowering --threshold or checking the input CSV.")
        return

    print_hits(hits)
    save_hits(args.output, hits)
    print(f"\nSaved {len(hits)} hit(s) to {args.output}")
    print("Frames flagged after_gap follow a detection gap; the true impact may be inside the gap.")


if __name__ == "__main__":
    main()
