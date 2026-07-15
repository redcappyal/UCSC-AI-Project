import argparse
import csv
import json
from dataclasses import dataclass
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

# Which bounce detector localizes the impact for each picked hit. Set to
# "legacy_sign_flip" to restore the pre-two-stage impact fit exactly.
BOUNCE_DETECTOR = "two_stage"
TWO_STAGE_DEFAULTS = {
    "min_trajectory_points": 7,
    "window_half_width": 7,
    "min_segment_points": 3,
    "min_direction_change_deg": 20.0,
    "min_split_improvement": 0.30,
}
# Samples kept on each side of a picked hit when handing the two-stage
# detector its per-event trajectory slice. Matching the fitting half-width
# keeps the stage-1 distance search anchored to this event; a wider slice
# lets the minimum drift onto post-bounce flight toward the other line.
EVENT_WINDOW_HALF_WIDTH = TWO_STAGE_DEFAULTS["window_half_width"]


def normalize_x_range(wall_x_range):
    if wall_x_range is None:
        return None
    left, right = wall_x_range
    left = float(left)
    right = float(right)
    if right < left:
        left, right = right, left
    if right <= left:
        return None
    return left, right


def is_inside_x_range(x, wall_x_range):
    if wall_x_range is None:
        return True
    left, right = wall_x_range
    return left <= float(x) <= right


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
                    "candidate_x": float(track_positions[sample][0]),
                    "candidate_y": float(track_positions[sample][1]),
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
                "candidate_x": float(positions[start][0]),
                "candidate_y": float(positions[start][1]),
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


@dataclass(frozen=True)
class BounceResult:
    impact_index: int  # index into the trajectory of the nearest sample
    impact_t: float  # sub-frame impact time, in trajectory-timestamp seconds
    impact_xy: tuple  # sub-frame impact position in pixels
    method: str
    diagnostics: dict


def calibration_line_coefficients(calibration):
    """Slope/intercept (y = m*x + b) for each named calibrated line."""
    coefficients = {}
    for index, line in enumerate(calibration.get("lines", [])):
        endpoints = line.get("endpoints") or []
        if len(endpoints) < 2:
            continue
        (x1, y1), (x2, y2) = endpoints[0], endpoints[1]
        if float(x1) == float(x2):
            # Vertical lines have no y = m*x + b form; wall lines never are.
            continue
        slope = (float(y2) - float(y1)) / (float(x2) - float(x1))
        intercept = float(y1) - slope * float(x1)
        coefficients[line.get("name") or f"line_{index}"] = (slope, intercept)
    return coefficients


def fit_segment_with_residual(times, positions):
    """fit_linear_motion plus the fit's summed squared residual, or None."""
    fit = fit_linear_motion(times, positions)
    if fit is None:
        return None
    p0, v = fit
    predicted = p0 + times[:, np.newaxis] * v
    return p0, v, float(np.sum((positions - predicted) ** 2))


def detect_bounce_legacy(trajectory, calibration, config=None, diagnostics_out=None):
    """The pre-two-stage detector behind the strategy interface, unchanged.

    Runs the original candidate/peak/impact-fit machinery on the trajectory
    and reports its strongest event. `calibration` is unused; registered as
    "legacy_sign_flip", the config name for the pre-two-stage method.
    """
    frames, timestamps, positions = trajectory
    frames = np.asarray(frames, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    tracks = split_into_tracks(frames, positions, MAX_GAP_FRAMES, MAX_JUMP_PX_PER_FRAME)
    candidates = compute_candidates(frames, timestamps, positions, tracks, SMOOTH_WINDOW)
    hits = pick_peaks(candidates, MIN_GAP_FRAMES, TOP_K, None)
    if not hits:
        if isinstance(diagnostics_out, dict):
            diagnostics_out["rejected"] = "no significant velocity change"
        return None

    hit = max(hits, key=lambda h: h["dv_magnitude"])
    index = int(hit["sample_global_index"])
    impact = estimate_impact(timestamps, positions, tracks, hit)

    diagnostics = {
        key: hit[key]
        for key in ("dv_magnitude", "turn_degrees", "speed_before", "speed_after", "after_gap")
    }
    if impact is None:
        impact_t = float(timestamps[index])
        impact_xy = (float(positions[index][0]), float(positions[index][1]))
    else:
        impact_t = float(impact["impact_time"])
        impact_xy = (float(impact["impact_x"]), float(impact["impact_y"]))
        diagnostics["impact_mismatch_px"] = impact["impact_mismatch_px"]
    if isinstance(diagnostics_out, dict):
        diagnostics_out.update(diagnostics)

    return BounceResult(
        impact_index=index,
        impact_t=impact_t,
        impact_xy=impact_xy,
        method="legacy_sign_flip",
        diagnostics=diagnostics,
    )


def detect_bounce_two_stage(trajectory, calibration, config=None, diagnostics_out=None):
    """Two-stage bounce detector.

    Stage 1 localizes the impact at the trajectory's closest approach to a
    calibrated wall line (the lines lie on the impact surface, so this works
    even for flat drives whose image-space velocity barely changes). Stage 2
    refines it with a two-segment least-squares fit of x(t) and y(t) inside a
    window around the candidate and intersects the segments for a sub-frame
    impact. Returns None, with the reason in `diagnostics_out`, when there is
    no clear break: a smooth fit failing silently is worse than no detection.
    """
    cfg = dict(TWO_STAGE_DEFAULTS)
    if config:
        cfg.update(config)
    diagnostics = diagnostics_out if isinstance(diagnostics_out, dict) else {}

    _, timestamps, positions = trajectory
    timestamps = np.asarray(timestamps, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    count = len(timestamps)
    if count < cfg["min_trajectory_points"]:
        diagnostics["rejected"] = f"only {count} trajectory point(s)"
        return None

    lines = calibration_line_coefficients(calibration) if calibration else {}
    if not lines:
        diagnostics["rejected"] = "no usable calibrated lines"
        return None

    # Stage 1: perpendicular pixel distance to each line, minimum over lines.
    names = list(lines)
    distances = np.stack(
        [
            np.abs(m * positions[:, 0] - positions[:, 1] + b) / np.sqrt(m * m + 1)
            for m, b in (lines[name] for name in names)
        ]
    )
    per_point = distances.min(axis=0)
    candidate = int(np.argmin(per_point))
    nearest_line = names[int(np.argmin(distances[:, candidate]))]

    # Stage 2 fits only a window around the candidate; other events in the
    # buffer (a floor bounce, a nick) would corrupt a global fit.
    half = cfg["window_half_width"]
    lo = max(0, candidate - half)
    hi = min(count, candidate + half + 1)
    min_points = cfg["min_segment_points"]

    diagnostics.update(
        {
            "nearest_line": nearest_line,
            "min_line_distance_px": float(per_point[candidate]),
            "window": (int(lo), int(hi - 1)),
        }
    )

    single = fit_segment_with_residual(timestamps[lo:hi], positions[lo:hi])
    best = None
    for k in range(lo + min_points - 1, hi - min_points + 1):
        fit_a = fit_segment_with_residual(timestamps[lo : k + 1], positions[lo : k + 1])
        fit_b = fit_segment_with_residual(timestamps[k:hi], positions[k:hi])
        if fit_a is None or fit_b is None:
            continue
        total = fit_a[2] + fit_b[2]
        if best is None or total < best[0]:
            best = (total, k, fit_a, fit_b)

    if single is None or best is None:
        diagnostics["rejected"] = "window too small for a two-segment fit"
        return None

    single_ssr = single[2]
    total, k_star, (p_a, v_a, ssr_a), (p_b, v_b, ssr_b) = best
    diagnostics.update(
        {
            "split_index": int(k_star),
            "single_fit_ssr": single_ssr,
            "split_ssr": total,
            "rms_residual_in": float(np.sqrt(ssr_a / (k_star - lo + 1))),
            "rms_residual_out": float(np.sqrt(ssr_b / (hi - k_star))),
        }
    )

    if single_ssr < 1e-9 or total > (1.0 - cfg["min_split_improvement"]) * single_ssr:
        kept = 100.0 * total / max(single_ssr, 1e-12)
        needed = 100.0 * (1.0 - cfg["min_split_improvement"])
        diagnostics["rejected"] = (
            f"no clear break: two-segment fit keeps {kept:.1f}% of the "
            f"single-fit residual (needs <= {needed:.0f}%)"
        )
        return None

    angle = turn_angle_degrees(v_a, v_b)
    diagnostics["direction_change_deg"] = angle

    # Near-parallel segments mean the split improvement was noise, not a
    # bounce; emitting the candidate centroid here would hallucinate one.
    if angle < cfg["min_direction_change_deg"]:
        diagnostics["rejected"] = (
            f"direction change {angle:.1f} deg is below "
            f"{cfg['min_direction_change_deg']:g} deg (near-parallel segments)"
        )
        return None

    # Sub-frame impact: closest approach in time of the two fitted motions
    # (same reasoning as _impact_from_index_sets).
    w = v_a - v_b
    ww = float(w @ w)
    t_star = None
    if ww >= 1e-9:
        t_candidate = float(-((p_a - p_b) @ w) / ww)
        t_low = float(timestamps[max(lo, k_star - 1)])
        t_high = float(timestamps[min(hi - 1, k_star + 1)])
        if t_low <= t_candidate <= t_high:
            t_star = t_candidate

    if t_star is None:
        diagnostics["fallback"] = "no reliable intersection"
        impact_index = int(k_star)
        impact_t = float(timestamps[k_star])
        impact_xy = (float(positions[k_star][0]), float(positions[k_star][1]))
    else:
        impact_t = t_star
        point = ((p_a + v_a * t_star) + (p_b + v_b * t_star)) / 2
        impact_xy = (float(point[0]), float(point[1]))
        impact_index = int(np.argmin(np.abs(timestamps - impact_t)))

    return BounceResult(
        impact_index=impact_index,
        impact_t=impact_t,
        impact_xy=impact_xy,
        method="two_stage",
        diagnostics=dict(diagnostics),
    )


BOUNCE_DETECTORS = {
    "legacy_sign_flip": detect_bounce_legacy,
    "two_stage": detect_bounce_two_stage,
}


def detect_bounce(trajectory, calibration, config=None, diagnostics_out=None):
    """Run the configured bounce detector on one event's trajectory buffer.

    `trajectory` is the (frames, timestamps, positions) triple produced by
    load_detected_positions*; `calibration` is the parsed calibration.json
    dict (the legacy method ignores it). Returns BounceResult or None.
    """
    name = (config or {}).get("bounce_detector", BOUNCE_DETECTOR)
    return BOUNCE_DETECTORS[name](trajectory, calibration, config, diagnostics_out)


def is_significant(candidate):
    # A wall bounce turns the ball sharply within one sample; flight-path
    # curvature (gravity) turns it only a few degrees per sample, even near
    # the arc's apex where the ball is slow. The |dv| floor rejects
    # direction flips from detector jitter on a slow or held ball.
    return (
        candidate["dv_magnitude"] >= MIN_DV_PX_PER_SECOND
        and candidate["turn_degrees"] >= MIN_TURN_DEGREES
    )


def score_candidate(candidate):
    dv_score = candidate["dv_magnitude"] / MIN_DV_PX_PER_SECOND
    turn_score = candidate["turn_degrees"] / MIN_TURN_DEGREES
    speed_score = min(candidate["speed_before"], candidate["speed_after"]) / MIN_DV_PX_PER_SECOND

    # Ranking, not filtering: keep the old thresholds in is_significant(), but
    # prefer candidates with both a sharp turn and meaningful ball speed.
    return float(
        0.55 * dv_score
        + 0.35 * turn_score
        + 0.10 * min(speed_score, 4.0)
    )


def pick_peaks(candidates, min_gap, top_k, threshold):
    for candidate in candidates:
        candidate["score"] = score_candidate(candidate)

    ranked = sorted(candidates, key=lambda c: (c["score"], c["dv_magnitude"]), reverse=True)
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
    wall_x_range=None,
    calibration=None,
    bounce_detector=None,
):
    wall_x_range = normalize_x_range(wall_x_range)
    detector = bounce_detector or BOUNCE_DETECTOR
    tracks = split_into_tracks(frames, positions, max_gap, max_jump)
    candidates = compute_candidates(frames, timestamps, positions, tracks, smooth)
    candidates = [
        candidate
        for candidate in candidates
        if is_inside_x_range(candidate["candidate_x"], wall_x_range)
    ]
    hits = pick_peaks(candidates, min_gap, top_k, threshold)

    # timestamp = frame / fps, so the frame at any time is time * fps.
    time_span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    fps = (frames[-1] - frames[0]) / time_span if time_span > 0 else 30.0

    # The two-stage detector localizes against the calibrated wall lines;
    # without a calibration the legacy impact fit is the only option.
    two_stage = detector == "two_stage" and calibration is not None

    for hit in hits:
        if two_stage:
            pivot = hit["sample_global_index"]
            lo = max(0, pivot - EVENT_WINDOW_HALF_WIDTH)
            hi = min(len(frames), pivot + EVENT_WINDOW_HALF_WIDTH + 1)
            diagnostics = {}
            result = detect_bounce_two_stage(
                (frames[lo:hi], timestamps[lo:hi], positions[lo:hi]),
                calibration,
                diagnostics_out=diagnostics,
            )
            hit.pop("track_index", None)
            hit.pop("sample_global_index", None)
            hit["method"] = "two_stage"
            hit["diagnostics"] = diagnostics
            if result is not None:
                hit.update(
                    {
                        "impact_x": result.impact_xy[0],
                        "impact_y": result.impact_xy[1],
                        "impact_time": result.impact_t,
                        # Display seeks a captured frame, not a sub-frame time.
                        "impact_frame": float(frames[lo + result.impact_index]),
                    }
                )
            continue

        impact = estimate_impact(timestamps, positions, tracks, hit)
        hit.pop("track_index", None)
        hit.pop("sample_global_index", None)
        if impact is not None:
            impact["impact_frame"] = impact["impact_time"] * fps
            hit.update(impact)

    hits = [
        hit
        for hit in hits
        if is_inside_x_range(hit.get("impact_x", hit["candidate_x"]), wall_x_range)
    ]
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
        "score",
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
                "score": f"{hit['score']:.3f}",
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
        f"{'rank':>4}  {'frame':>8}  {'time (s)':>10}  {'score':>8}  {'|dv| px/s':>12}  {'v before':>10}  "
        f"{'v after':>10}  {'turn deg':>9}  {'impact @':>10}  after_gap"
    )
    print(header)
    print("-" * len(header))
    for rank, hit in enumerate(hits, start=1):
        impact_text = f"{hit['impact_frame']:.1f}" if "impact_frame" in hit else "-"
        print(
            f"{rank:>4}  {hit['hit_frame']:>8}  {hit['timestamp_seconds']:>10.3f}  "
            f"{hit['score']:>8.2f}  "
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
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration.json with the fitted wall lines; required by the two_stage detector.",
    )
    parser.add_argument(
        "--bounce-detector",
        choices=sorted(BOUNCE_DETECTORS),
        default=None,
        help=f"Impact localization method. Defaults to {BOUNCE_DETECTOR}.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    calibration = None
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))

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
        calibration=calibration,
        bounce_detector=args.bounce_detector,
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
