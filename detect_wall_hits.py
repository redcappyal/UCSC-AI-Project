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
MIN_DV_PX_PER_SECOND = 200.0
MIN_TURN_DEGREES = 25.0
IMPACT_FIT_SAMPLES = 4
GAP_BRIDGE_SECONDS = 0.5
MAX_IMPACT_MISMATCH_PX = 40.0
REQUIRED_COLUMNS = ("source_frame", "timestamp_seconds", "detected", "x_center", "y_center")


def turn_angle_degrees(v_before, v_after):
    norms = np.linalg.norm(v_before) * np.linalg.norm(v_after)
    if norms < 1e-9:
        return 0.0
    cos_angle = np.clip(float(v_before @ v_after) / norms, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_detected_positions_from_rows(rows):
    frames = []
    timestamps = []
    positions = []
    duplicate_count = 0

    for row in rows:
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


def load_detected_positions(csv_path):
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing_text}")

        return load_detected_positions_from_rows(reader)


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


def compute_candidates(frames, timestamps, positions, tracks, smooth_window, max_jump=MAX_JUMP_PX_PER_FRAME):
    candidates = []
    track_velocities = {}

    for track_index, (start, end) in enumerate(tracks):
        track_frames = frames[start:end]
        track_times = timestamps[start:end]
        track_positions = smooth_positions(positions[start:end], smooth_window)

        if len(track_frames) >= 2:
            dt = np.diff(track_times)[:, np.newaxis]
            track_velocities[track_index] = np.diff(track_positions, axis=0) / dt

        if len(track_frames) < 3:
            continue

        after_gap = start > 0
        velocities = track_velocities[track_index]
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
                    "turn_degrees": turn_angle_degrees(velocities[sample - 1], velocities[sample]),
                    "after_gap": bool(after_gap and sample <= 2),
                    "track_index": track_index,
                    "sample_global_index": start + sample,
                }
            )

    # A bounce hidden entirely inside a detection gap produces no intra-track
    # velocity change; compare the motion on either side of each short gap.
    for track_index in range(1, len(tracks)):
        if track_index - 1 not in track_velocities or track_index not in track_velocities:
            continue

        prev_start, prev_end = tracks[track_index - 1]
        start, end = tracks[track_index]
        gap_seconds = float(timestamps[start] - timestamps[prev_end - 1])
        if not (0 < gap_seconds <= GAP_BRIDGE_SECONDS):
            continue

        # A displacement implying more than max_jump px/frame means the track
        # split was a detector switch (teleport), not a detection gap.
        frame_delta = float(frames[start] - frames[prev_end - 1])
        displacement = float(np.linalg.norm(positions[start] - positions[prev_end - 1]))
        if frame_delta <= 0 or displacement / frame_delta > max_jump:
            continue

        v_before = track_velocities[track_index - 1][-1]
        v_after = track_velocities[track_index][0]
        candidates.append(
            {
                "hit_frame": int(frames[start]),
                "timestamp_seconds": float(timestamps[start]),
                "dv_magnitude": float(np.linalg.norm(v_after - v_before)),
                "speed_before": float(np.linalg.norm(v_before)),
                "speed_after": float(np.linalg.norm(v_after)),
                "turn_degrees": turn_angle_degrees(v_before, v_after),
                "after_gap": True,
                "track_index": track_index,
                "sample_global_index": start,
            }
        )

    return candidates


def fit_linear_motion(times, positions):
    """Least-squares fit p(t) = p0 + v*t; returns (p0, v) or None."""
    if len(times) < 2:
        return None

    design = np.stack([np.ones_like(times), times], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
    return coefficients[0], coefficients[1]


def _impact_from_index_sets(timestamps, positions, in_indices, out_indices):
    if len(in_indices) < 2 or len(out_indices) < 2:
        return None

    fit_in = fit_linear_motion(timestamps[in_indices], positions[in_indices])
    fit_out = fit_linear_motion(timestamps[out_indices], positions[out_indices])
    if fit_in is None or fit_out is None:
        return None

    p_in, v_in = fit_in
    p_out, v_out = fit_out

    # Closest approach in time of the two fitted motions. A spatial line
    # intersection would be ill-conditioned when the ball returns along the
    # same image-space line; the relative velocity w is large exactly then.
    w = v_in - v_out
    ww = float(w @ w)
    if ww < 1e-9:
        return None

    d = p_in - p_out
    t_star = float(-(d @ w) / ww)

    # The impact must lie between the last incoming and first outgoing sample.
    t_star = min(max(t_star, float(timestamps[in_indices[-1]])), float(timestamps[out_indices[0]]))

    pos_in = p_in + v_in * t_star
    pos_out = p_out + v_out * t_star
    mismatch = float(np.linalg.norm(pos_in - pos_out))
    if mismatch > MAX_IMPACT_MISMATCH_PX:
        return None

    point = (pos_in + pos_out) / 2
    return {
        "impact_x": float(point[0]),
        "impact_y": float(point[1]),
        "impact_time": t_star,
        "impact_mismatch_px": mismatch,
    }


def estimate_impact(timestamps, positions, tracks, hit):
    """Estimate where the ball met the wall for a picked hit, or None.

    Fits incoming/outgoing motion around the hit sample. For hits right after
    a detection gap it also tries the gap-impact hypothesis (incoming motion
    from the previous track's tail), keeping whichever fit agrees best.
    """
    track_index = hit["track_index"]
    pivot = hit["sample_global_index"]
    track_start, track_end = tracks[track_index]

    previous_tail = []
    if track_index > 0:
        prev_start, prev_end = tracks[track_index - 1]
        gap_seconds = timestamps[track_start] - timestamps[prev_end - 1]
        if 0 <= gap_seconds <= GAP_BRIDGE_SECONDS:
            previous_tail = list(range(max(prev_start, prev_end - IMPACT_FIT_SAMPLES), prev_end))

    estimates = []

    # Hypothesis A: impact at the hit sample (excluded from both fits).
    in_indices = list(range(max(track_start, pivot - IMPACT_FIT_SAMPLES), pivot))
    if len(in_indices) < 2 and previous_tail:
        in_indices = previous_tail[len(in_indices) - IMPACT_FIT_SAMPLES :] + in_indices
    out_indices = list(range(pivot + 1, min(track_end, pivot + 1 + IMPACT_FIT_SAMPLES)))
    estimate = _impact_from_index_sets(timestamps, positions, in_indices, out_indices)
    if estimate is not None:
        estimates.append(estimate)

    # Hypothesis B: impact inside the detection gap before this track.
    if hit.get("after_gap") and previous_tail:
        out_indices = list(range(track_start, min(track_end, track_start + IMPACT_FIT_SAMPLES)))
        estimate = _impact_from_index_sets(timestamps, positions, previous_tail, out_indices)
        if estimate is not None:
            estimates.append(estimate)

    if not estimates:
        return None

    return min(estimates, key=lambda e: e["impact_mismatch_px"])


def is_significant(candidate):
    # A wall bounce turns the ball sharply within one sample; flight-path
    # curvature (gravity) turns it only a few degrees per sample, even near
    # the arc's apex where the ball is slow. The |dv| floor rejects
    # direction flips from detector jitter on a slow or held ball.
    return (
        candidate["dv_magnitude"] >= MIN_DV_PX_PER_SECOND
        and candidate["turn_degrees"] >= MIN_TURN_DEGREES
    )


def pick_peaks(candidates, min_gap, top_k, threshold):
    ranked = sorted(candidates, key=lambda c: c["dv_magnitude"], reverse=True)
    picked = []

    for candidate in ranked:
        if threshold is not None and candidate["dv_magnitude"] < threshold:
            break
        # Significance depends on the candidate's own speed, so it is not
        # monotonic in |dv|: keep scanning rather than stopping.
        if threshold is None and not is_significant(candidate):
            continue
        if any(abs(candidate["hit_frame"] - p["hit_frame"]) < min_gap for p in picked):
            continue

        picked.append(candidate)
        if threshold is None and len(picked) >= top_k:
            break

    return picked


def detect_hits_from_positions(
    frames,
    timestamps,
    positions,
    *,
    max_gap=MAX_GAP_FRAMES,
    max_jump=MAX_JUMP_PX_PER_FRAME,
    smooth=SMOOTH_WINDOW,
    min_gap=MIN_GAP_FRAMES,
    top_k=TOP_K,
    threshold=None,
):
    tracks = split_into_tracks(frames, positions, max_gap, max_jump)
    candidates = compute_candidates(frames, timestamps, positions, tracks, smooth)
    hits = pick_peaks(candidates, min_gap, top_k, threshold)

    # timestamp = frame / fps, so the frame at any time is time * fps.
    time_span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    fps = (frames[-1] - frames[0]) / time_span if time_span > 0 else 30.0

    for hit in hits:
        impact = estimate_impact(timestamps, positions, tracks, hit)
        hit.pop("track_index", None)
        hit.pop("sample_global_index", None)
        if impact is not None:
            impact["impact_frame"] = impact["impact_time"] * fps
            hit.update(impact)

    return sorted(hits, key=lambda hit: hit["hit_frame"])


def detect_hits_from_rows(rows, **kwargs):
    frames, timestamps, positions, _ = load_detected_positions_from_rows(rows)
    return detect_hits_from_positions(frames, timestamps, positions, **kwargs)


def detect_hits(csv_path, **kwargs):
    frames, timestamps, positions, _ = load_detected_positions(csv_path)
    return detect_hits_from_positions(frames, timestamps, positions, **kwargs)


def save_hits(output_path, hits):
    fieldnames = [
        "hit_frame",
        "timestamp_seconds",
        "dv_magnitude",
        "speed_before",
        "speed_after",
        "turn_degrees",
        "after_gap",
        "impact_frame",
        "impact_x",
        "impact_y",
        "impact_time",
        "impact_mismatch_px",
    ]
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for hit in sorted(hits, key=lambda h: h["hit_frame"]):
            row = {
                **hit,
                "timestamp_seconds": f"{hit['timestamp_seconds']:.6f}",
                "dv_magnitude": f"{hit['dv_magnitude']:.3f}",
                "speed_before": f"{hit['speed_before']:.3f}",
                "speed_after": f"{hit['speed_after']:.3f}",
                "turn_degrees": f"{hit['turn_degrees']:.1f}",
            }
            for key in ("impact_frame", "impact_x", "impact_y", "impact_time", "impact_mismatch_px"):
                if key in hit:
                    row[key] = f"{hit[key]:.3f}"
            writer.writerow(row)


def print_hits(hits):
    header = (
        f"{'rank':>4}  {'frame':>8}  {'time (s)':>10}  {'|dv| px/s':>12}  {'v before':>10}  "
        f"{'v after':>10}  {'turn deg':>9}  {'impact @':>10}  after_gap"
    )
    print(header)
    print("-" * len(header))
    for rank, hit in enumerate(hits, start=1):
        impact_text = f"{hit['impact_frame']:.1f}" if "impact_frame" in hit else "-"
        print(
            f"{rank:>4}  {hit['hit_frame']:>8}  {hit['timestamp_seconds']:>10.3f}  "
            f"{hit['dv_magnitude']:>12.1f}  {hit['speed_before']:>10.1f}  "
            f"{hit['speed_after']:>10.1f}  {hit['turn_degrees']:>9.1f}  "
            f"{impact_text:>10}  {'yes' if hit['after_gap'] else ''}"
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
        help=(
            "Report every hit with |dv| at or above this value. Overrides the "
            "default mode, which keeps only hits whose |dv| rivals the ball speed."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()

    frames, timestamps, positions, duplicate_count = load_detected_positions(args.ball_csv)
    tracks = split_into_tracks(frames, positions, args.max_gap, args.max_jump)
    candidates = compute_candidates(frames, timestamps, positions, tracks, args.smooth)
    hits = detect_hits_from_positions(
        frames,
        timestamps,
        positions,
        max_gap=args.max_gap,
        max_jump=args.max_jump,
        smooth=args.smooth,
        min_gap=args.min_gap,
        top_k=args.top_k,
        threshold=args.threshold,
    )

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
