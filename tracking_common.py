"""Shared ball-tracking constants and pure helpers.

Importing this module has no side effects (no network clients, no model
loads); anything here is safe for the Flask app, the CLIs, and tests.
"""

import math

import cv2


CONFIDENCE_THRESHOLD = 0.40
BALL_CLASS_NAMES = {"ball", "squash ball", "squash-ball", "squash_ball"}

# The frame rate the frame-counted constants below were tuned at. Kept in step
# with detect_wall_hits.REFERENCE_FPS -- both describe the same capture.
# Pixel-domain constants here (MOTION_TRACK_*_PX, and the FLOOR_REBOUND_*_PX
# thresholds) are deliberately left unscaled: the ball tier only admits footage
# near the resolution they were tuned at (capabilities.BALL_MIN_WIDTH_PX).
REFERENCE_FPS = 60.0

MOTION_TRACK_WINDOW_FRAMES = 5
MOTION_TRACK_MIN_DETECTIONS = 2
MOTION_TRACK_MIN_SPAN_PX = 8.0
MOTION_TRACK_MIN_PATH_PX = 10.0
MOTION_TRACK_MAX_STEP_PX = 80.0
MOTION_TRACK_STATIONARY_RADIUS_PX = 8.0
MOTION_TRACK_BONUS = 0.30
STATIONARY_TRACK_PENALTY = 0.40
TRAJECTORY_FILL_MAX_GAP_FRAMES = 6
TRAJECTORY_FILL_EDGE_MARGIN_PX = 24.0
FLOOR_REBOUND_SEARCH_FRAMES = 4
FLOOR_REBOUND_WINDOW_FRAMES = 3
FLOOR_REBOUND_MIN_DETECTIONS_PER_SIDE = 2
FLOOR_REBOUND_MIN_Y_SPEED_PX_PER_FRAME = 1.25
FLOOR_REBOUND_MIN_REVERSAL_PX_PER_FRAME = 3.0
FLOOR_REBOUND_MIN_PROMINENCE_PX = 3.0
FLOOR_REBOUND_FLOOR_SEAM_MARGIN_PX = 36.0
FLOOR_REBOUND_LOOKBACK_FRAMES = 24
FLOOR_REBOUND_CONTINUATION_WINDOW_FRAMES = 6
FLOOR_REBOUND_MIN_HORIZONTAL_RETENTION = 0.50
FLOOR_REBOUND_MAX_CANDIDATE_TURN_DEGREES = 35.0
FLOOR_REBOUND_MIN_CANDIDATE_SPEED_RATIO = 0.45
FLOOR_REBOUND_MAX_PATH_GAP_FRAMES = 6
FLOOR_REBOUND_NEARBY_IMPACT_SEARCH_FRAMES = 8
FLOOR_REBOUND_NEARBY_IMPACT_TURN_DEGREES = 45.0
CSV_FIELDNAMES = [
    "source_frame",
    "timestamp_seconds",
    "detected",
    "class_name",
    "confidence",
    "x_center",
    "y_center",
    "width",
    "height",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]


def find_predictions(obj):
    predictions = []

    if isinstance(obj, dict):
        if all(key in obj for key in ("x", "y", "width", "height")):
            predictions.append(obj)

        for value in obj.values():
            predictions.extend(find_predictions(value))

    elif isinstance(obj, list):
        for item in obj:
            predictions.extend(find_predictions(item))

    return predictions


def prediction_class_name(prediction):
    return str(
        prediction.get(
            "class",
            prediction.get("class_name", prediction.get("name", "object")),
        )
    )


def is_ball_prediction(prediction):
    class_name = prediction_class_name(prediction).strip().lower()
    return class_name in BALL_CLASS_NAMES


def candidate_ball_predictions(predictions, confidence_threshold=CONFIDENCE_THRESHOLD):
    valid_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("confidence", 1.0) >= confidence_threshold
    ]
    ball_predictions = [
        prediction for prediction in valid_predictions if is_ball_prediction(prediction)
    ]

    if not ball_predictions:
        ball_predictions = valid_predictions

    return ball_predictions


def select_ball_prediction(predictions, confidence_threshold=CONFIDENCE_THRESHOLD):
    ball_predictions = candidate_ball_predictions(predictions, confidence_threshold)
    if not ball_predictions:
        return None

    return max(ball_predictions, key=lambda prediction: prediction.get("confidence", 1.0))


def prediction_distance(a, b):
    return ((float(a["x"]) - float(b["x"])) ** 2 + (float(a["y"]) - float(b["y"])) ** 2) ** 0.5


def _trajectory_y_point(row):
    if not row or not row.get("detected"):
        return None
    try:
        y = float(row["y"])
    except (KeyError, TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def _trajectory_xy_point(row):
    if not row or not row.get("detected"):
        return None
    try:
        x = float(row["x"])
        y = float(row["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _linear_slope(points):
    """Least-squares y/frame slope for a short trajectory segment."""
    if len(points) < 2:
        return None
    mean_frame = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_frame) ** 2 for point in points)
    if denominator <= 1e-9:
        return None
    return sum(
        (point[0] - mean_frame) * (point[1] - mean_y)
        for point in points
    ) / denominator


def detect_floor_rebound(
    rows_by_frame,
    frame,
    *,
    search_frames=FLOOR_REBOUND_SEARCH_FRAMES,
    window_frames=FLOOR_REBOUND_WINDOW_FRAMES,
    min_detections_per_side=FLOOR_REBOUND_MIN_DETECTIONS_PER_SIDE,
    min_y_speed_px_per_frame=FLOOR_REBOUND_MIN_Y_SPEED_PX_PER_FRAME,
    min_reversal_px_per_frame=FLOOR_REBOUND_MIN_REVERSAL_PX_PER_FRAME,
    min_prominence_px=FLOOR_REBOUND_MIN_PROMINENCE_PX,
    search_after_frames=None,
):
    """Detect a clear ground-bounce signature near a candidate frame.

    Image y increases downward. A floor rebound therefore has positive y
    velocity before impact and negative y velocity after impact, forming a
    local y maximum. Searching a few frames around the model prediction
    tolerates small timing errors while the speed and prominence gates avoid
    vetoing noisy or nearly flat front-wall trajectories.
    """
    search_frames = max(0, int(search_frames))
    if search_after_frames is None:
        search_after_frames = search_frames
    search_after_frames = max(0, int(search_after_frames))
    window_frames = max(1, int(window_frames))
    min_detections_per_side = max(2, int(min_detections_per_side))
    best_stats = {
        "is_floor_rebound": False,
        "center_frame": None,
        "vy_before_px_per_frame": None,
        "vy_after_px_per_frame": None,
        "reversal_px_per_frame": 0.0,
        "prominence_px": 0.0,
        "peak_frame": None,
        "peak_x": None,
        "peak_y": None,
    }
    best_floor_stats = None

    for center in range(
        int(frame) - search_frames,
        int(frame) + search_after_frames + 1,
    ):
        before = []
        after = []
        neighborhood = []
        for sample_frame in range(center - window_frames, center + window_frames + 1):
            y = _trajectory_y_point(rows_by_frame.get(sample_frame))
            if y is None:
                continue
            point = (sample_frame, y)
            neighborhood.append(point)
            if sample_frame < center:
                before.append(point)
            elif sample_frame > center:
                after.append(point)

        if (
            len(before) < min_detections_per_side
            or len(after) < min_detections_per_side
        ):
            continue

        vy_before = _linear_slope(before)
        vy_after = _linear_slope(after)
        if vy_before is None or vy_after is None:
            continue

        peak_frame, peak_y = max(neighborhood, key=lambda point: point[1])
        peak_row = rows_by_frame.get(peak_frame) or {}
        try:
            peak_x = float(peak_row["x"])
        except (KeyError, TypeError, ValueError):
            peak_x = None
        if peak_x is not None and not math.isfinite(peak_x):
            peak_x = None
        shoulder_y = max(before[0][1], after[-1][1])
        prominence = peak_y - shoulder_y
        reversal = vy_before - vy_after
        stats = {
            "is_floor_rebound": bool(
                vy_before >= float(min_y_speed_px_per_frame)
                and vy_after <= -float(min_y_speed_px_per_frame)
                and reversal >= float(min_reversal_px_per_frame)
                and prominence >= float(min_prominence_px)
            ),
            "center_frame": center,
            "vy_before_px_per_frame": float(vy_before),
            "vy_after_px_per_frame": float(vy_after),
            "reversal_px_per_frame": float(reversal),
            "prominence_px": float(prominence),
            "peak_frame": int(peak_frame),
            "peak_x": peak_x,
            "peak_y": float(peak_y),
        }
        if stats["is_floor_rebound"]:
            if best_floor_stats is None or (
                stats["prominence_px"],
                stats["reversal_px_per_frame"],
                -abs(center - int(frame)),
            ) > (
                best_floor_stats["prominence_px"],
                best_floor_stats["reversal_px_per_frame"],
                -abs(best_floor_stats["center_frame"] - int(frame)),
            ):
                best_floor_stats = stats
            continue
        if stats["reversal_px_per_frame"] > best_stats["reversal_px_per_frame"]:
            best_stats = stats

    if best_floor_stats is not None:
        return True, best_floor_stats
    return False, best_stats


def floor_rebound_is_near_floor_seam(
    stats,
    bottom_left,
    bottom_right,
    *,
    margin_px=FLOOR_REBOUND_FLOOR_SEAM_MARGIN_PX,
):
    """Require a rebound peak to lie at/below the calibrated wall-floor seam."""
    if not stats.get("is_floor_rebound"):
        return False
    try:
        peak_x = float(stats["peak_x"])
        peak_y = float(stats["peak_y"])
        left_x = float(bottom_left.x)
        left_y = float(bottom_left.y)
        right_x = float(bottom_right.x)
        right_y = float(bottom_right.y)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (
        peak_x,
        peak_y,
        left_x,
        left_y,
        right_x,
        right_y,
    )):
        return False

    dx = right_x - left_x
    if abs(dx) <= 1e-9:
        seam_y = (left_y + right_y) / 2.0
    else:
        progress = (peak_x - left_x) / dx
        seam_y = left_y + progress * (right_y - left_y)
    return peak_y >= seam_y - max(0.0, float(margin_px))


def _trajectory_velocity_around_frame(rows_by_frame, frame, window_frames):
    """Estimate image-plane velocity immediately before and after a frame."""
    frame = int(frame)
    window_frames = max(1, int(window_frames))
    center = _trajectory_xy_point(rows_by_frame.get(frame))
    if center is None:
        return None

    before = []
    after = []
    for sample_frame in range(frame - window_frames, frame + window_frames + 1):
        point = _trajectory_xy_point(rows_by_frame.get(sample_frame))
        if point is None:
            continue
        sample = (sample_frame, point)
        if sample_frame <= frame:
            before.append(sample)
        if sample_frame >= frame:
            after.append(sample)

    if len(before) < 2 or len(after) < 2:
        return None

    vx_before = _linear_slope([(sample_frame, point[0]) for sample_frame, point in before])
    vy_before = _linear_slope([(sample_frame, point[1]) for sample_frame, point in before])
    vx_after = _linear_slope([(sample_frame, point[0]) for sample_frame, point in after])
    vy_after = _linear_slope([(sample_frame, point[1]) for sample_frame, point in after])
    if None in (vx_before, vy_before, vx_after, vy_after):
        return None

    speed_before = math.hypot(vx_before, vy_before)
    speed_after = math.hypot(vx_after, vy_after)
    if speed_before <= 1e-9 or speed_after <= 1e-9:
        turn_degrees = 0.0
        speed_ratio = 0.0
    else:
        cosine = (
            vx_before * vx_after + vy_before * vy_after
        ) / (speed_before * speed_after)
        turn_degrees = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        speed_ratio = min(speed_before, speed_after) / max(speed_before, speed_after)

    return {
        "vx_before_px_per_frame": float(vx_before),
        "vy_before_px_per_frame": float(vy_before),
        "vx_after_px_per_frame": float(vx_after),
        "vy_after_px_per_frame": float(vy_after),
        "speed_before_px_per_frame": float(speed_before),
        "speed_after_px_per_frame": float(speed_after),
        "speed_ratio": float(speed_ratio),
        "turn_degrees": float(turn_degrees),
    }


def candidate_on_floor_rebound_trajectory(
    rows_by_frame,
    frame,
    bottom_left,
    bottom_right,
    *,
    lookback_frames=FLOOR_REBOUND_LOOKBACK_FRAMES,
    continuation_window_frames=FLOOR_REBOUND_CONTINUATION_WINDOW_FRAMES,
    min_horizontal_retention=FLOOR_REBOUND_MIN_HORIZONTAL_RETENTION,
    max_candidate_turn_degrees=FLOOR_REBOUND_MAX_CANDIDATE_TURN_DEGREES,
    min_candidate_speed_ratio=FLOOR_REBOUND_MIN_CANDIDATE_SPEED_RATIO,
    max_path_gap_frames=FLOOR_REBOUND_MAX_PATH_GAP_FRAMES,
    nearby_impact_search_frames=FLOOR_REBOUND_NEARBY_IMPACT_SEARCH_FRAMES,
    nearby_impact_turn_degrees=FLOOR_REBOUND_NEARBY_IMPACT_TURN_DEGREES,
    seam_margin_px=FLOOR_REBOUND_FLOOR_SEAM_MARGIN_PX,
    min_candidate_delay_frames=FLOOR_REBOUND_SEARCH_FRAMES + 1,
):
    """Identify a false hit on the approach to or exit from a ground bounce.

    A larger blind rebound window removes real wall contacts that happen soon
    after a floor bounce. This attribution is deliberately stricter: the floor
    event must be at the calibrated seam, retain horizontal momentum, connect
    continuously to the candidate, and the candidate itself must have no
    visible impact turn or abrupt speed change.
    """
    frame = int(frame)
    lookback_frames = max(1, int(lookback_frames))
    continuation_window_frames = max(2, int(continuation_window_frames))
    min_candidate_delay_frames = max(1, int(min_candidate_delay_frames))
    max_path_gap_frames = max(1, int(max_path_gap_frames))
    nearby_impact_search_frames = max(0, int(nearby_impact_search_frames))

    is_rebound, rebound_stats = detect_floor_rebound(
        rows_by_frame,
        frame,
        search_frames=lookback_frames,
        search_after_frames=lookback_frames,
    )
    peak_frame = rebound_stats.get("peak_frame")
    if not is_rebound or peak_frame is None:
        return False, {"reason": "no_nearby_floor_rebound"}

    frame_offset_from_rebound = frame - int(peak_frame)
    distance_from_rebound = abs(frame_offset_from_rebound)
    if (
        distance_from_rebound < min_candidate_delay_frames
        or distance_from_rebound > lookback_frames
    ):
        return False, {
            "reason": "rebound_outside_delay_range",
            "frame_offset_from_rebound": frame_offset_from_rebound,
            "distance_from_rebound": distance_from_rebound,
            "rebound": rebound_stats,
        }
    if not floor_rebound_is_near_floor_seam(
        rebound_stats,
        bottom_left,
        bottom_right,
        margin_px=seam_margin_px,
    ):
        return False, {
            "reason": "rebound_not_at_floor_seam",
            "frame_offset_from_rebound": frame_offset_from_rebound,
            "distance_from_rebound": distance_from_rebound,
            "rebound": rebound_stats,
        }

    rebound_velocity = _trajectory_velocity_around_frame(
        rows_by_frame,
        peak_frame,
        continuation_window_frames,
    )
    candidate_velocity = _trajectory_velocity_around_frame(
        rows_by_frame,
        frame,
        continuation_window_frames,
    )
    if rebound_velocity is None or candidate_velocity is None:
        return False, {
            "reason": "insufficient_velocity_context",
            "frame_offset_from_rebound": frame_offset_from_rebound,
            "distance_from_rebound": distance_from_rebound,
            "rebound": rebound_stats,
        }

    vx_before = rebound_velocity["vx_before_px_per_frame"]
    vx_after = rebound_velocity["vx_after_px_per_frame"]
    horizontal_speed = max(abs(vx_before), abs(vx_after))
    if horizontal_speed <= 1.0:
        horizontal_retention = 1.0 if abs(vx_after - vx_before) <= 1.0 else 0.0
        horizontal_direction_continues = horizontal_retention > 0.0
    else:
        horizontal_retention = min(abs(vx_before), abs(vx_after)) / horizontal_speed
        horizontal_direction_continues = vx_before * vx_after > 0.0
    if (
        not horizontal_direction_continues
        or horizontal_retention < max(0.0, float(min_horizontal_retention))
    ):
        return False, {
            "reason": "horizontal_motion_changed_at_rebound",
            "frame_offset_from_rebound": frame_offset_from_rebound,
            "distance_from_rebound": distance_from_rebound,
            "horizontal_retention": float(horizontal_retention),
            "rebound": rebound_stats,
            "rebound_velocity": rebound_velocity,
        }

    path_start = min(int(peak_frame), frame)
    path_end = max(int(peak_frame), frame)
    path_frames = [
        sample_frame
        for sample_frame in range(path_start, path_end + 1)
        if _trajectory_xy_point(rows_by_frame.get(sample_frame)) is not None
    ]
    if len(path_frames) < 4:
        return False, {
            "reason": "insufficient_continuation_path",
            "frame_offset_from_rebound": frame_offset_from_rebound,
            "distance_from_rebound": distance_from_rebound,
            "rebound": rebound_stats,
        }
    largest_gap = max(
        (next_frame - previous_frame)
        for previous_frame, next_frame in zip(path_frames, path_frames[1:])
    )
    if largest_gap > max_path_gap_frames:
        return False, {
            "reason": "continuation_path_gap",
            "frame_offset_from_rebound": frame_offset_from_rebound,
            "distance_from_rebound": distance_from_rebound,
            "largest_path_gap_frames": int(largest_gap),
            "rebound": rebound_stats,
        }

    candidate_turn = candidate_velocity["turn_degrees"]
    candidate_speed_ratio = candidate_velocity["speed_ratio"]
    candidate_has_impact_evidence = (
        candidate_turn >= max(0.0, float(max_candidate_turn_degrees))
        or candidate_speed_ratio < max(0.0, float(min_candidate_speed_ratio))
    )
    if candidate_has_impact_evidence:
        return False, {
            "reason": "candidate_has_impact_evidence",
            "frame_offset_from_rebound": frame_offset_from_rebound,
            "distance_from_rebound": distance_from_rebound,
            "horizontal_retention": float(horizontal_retention),
            "largest_path_gap_frames": int(largest_gap),
            "rebound": rebound_stats,
            "rebound_velocity": rebound_velocity,
            "candidate_velocity": candidate_velocity,
        }

    nearby_impact = None
    for impact_frame in range(
        frame - nearby_impact_search_frames,
        frame + nearby_impact_search_frames + 1,
    ):
        if abs(impact_frame - int(peak_frame)) <= FLOOR_REBOUND_WINDOW_FRAMES:
            continue
        impact_velocity = _trajectory_velocity_around_frame(
            rows_by_frame,
            impact_frame,
            continuation_window_frames,
        )
        if impact_velocity is None:
            continue
        if (
            impact_velocity["turn_degrees"]
            >= max(0.0, float(nearby_impact_turn_degrees))
        ):
            if (
                nearby_impact is None
                or impact_velocity["turn_degrees"]
                > nearby_impact["velocity"]["turn_degrees"]
            ):
                nearby_impact = {
                    "frame": int(impact_frame),
                    "velocity": impact_velocity,
                }
    if nearby_impact is not None:
        return False, {
            "reason": "candidate_near_non_floor_impact",
            "frame_offset_from_rebound": frame_offset_from_rebound,
            "distance_from_rebound": distance_from_rebound,
            "horizontal_retention": float(horizontal_retention),
            "largest_path_gap_frames": int(largest_gap),
            "rebound": rebound_stats,
            "rebound_velocity": rebound_velocity,
            "candidate_velocity": candidate_velocity,
            "nearby_impact": nearby_impact,
        }

    return True, {
        "reason": (
            "smooth_floor_rebound_continuation"
            if frame_offset_from_rebound > 0
            else "smooth_floor_rebound_approach"
        ),
        "frame_offset_from_rebound": frame_offset_from_rebound,
        "distance_from_rebound": distance_from_rebound,
        "horizontal_retention": float(horizontal_retention),
        "largest_path_gap_frames": int(largest_gap),
        "rebound": rebound_stats,
        "rebound_velocity": rebound_velocity,
        "candidate_velocity": candidate_velocity,
    }


def nearest_prediction(point, predictions):
    if not predictions:
        return None
    return min(predictions, key=lambda prediction: prediction_distance(point, prediction))


def linked_predictions_in_direction(
    frame,
    candidate,
    candidates_by_frame,
    window_frames,
    direction,
    max_step_px=MOTION_TRACK_MAX_STEP_PX,
):
    linked = []
    current_frame = frame
    current_prediction = candidate
    nearby_frames = [
        other_frame
        for other_frame in sorted(candidates_by_frame)
        if 0 < (other_frame - frame) * direction <= window_frames
    ]
    if direction < 0:
        nearby_frames.reverse()

    for other_frame in nearby_frames:
        frame_gap = abs(other_frame - current_frame)
        if frame_gap <= 0:
            continue

        nearest = nearest_prediction(current_prediction, candidates_by_frame[other_frame])
        if nearest is None:
            continue

        max_distance = max_step_px * frame_gap
        if prediction_distance(current_prediction, nearest) > max_distance:
            continue

        linked.append((other_frame, nearest))
        current_frame = other_frame
        current_prediction = nearest

    return linked


def candidate_motion_stats(frame, candidate, candidates_by_frame, window_frames):
    points = (
        linked_predictions_in_direction(
            frame,
            candidate,
            candidates_by_frame,
            window_frames,
            direction=-1,
        )
        + [(frame, candidate)]
        + linked_predictions_in_direction(
            frame,
            candidate,
            candidates_by_frame,
            window_frames,
            direction=1,
        )
    )

    points.sort(key=lambda item: item[0])
    if not points:
        return {"detected": 0, "span_px": 0.0, "path_px": 0.0}

    xs = [float(prediction["x"]) for _, prediction in points]
    ys = [float(prediction["y"]) for _, prediction in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    path = 0.0
    for (_, prev), (_, cur) in zip(points, points[1:]):
        path += prediction_distance(prev, cur)

    return {
        "detected": len(points),
        "span_px": (x_span * x_span + y_span * y_span) ** 0.5,
        "path_px": path,
    }


def candidate_stationary_stats(
    frame,
    candidate,
    candidates_by_frame,
    window_frames,
    stationary_radius_px=MOTION_TRACK_STATIONARY_RADIUS_PX,
):
    points = [(frame, candidate)]
    for other_frame in sorted(candidates_by_frame):
        if other_frame == frame or abs(other_frame - frame) > window_frames:
            continue

        nearby = [
            prediction
            for prediction in candidates_by_frame[other_frame]
            if prediction_distance(candidate, prediction) <= stationary_radius_px
        ]
        if nearby:
            points.append((other_frame, nearest_prediction(candidate, nearby)))

    points.sort(key=lambda item: item[0])
    xs = [float(prediction["x"]) for _, prediction in points]
    ys = [float(prediction["y"]) for _, prediction in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    path = 0.0
    for (_, prev), (_, cur) in zip(points, points[1:]):
        path += prediction_distance(prev, cur)

    return {
        "detected": len(points),
        "span_px": (x_span * x_span + y_span * y_span) ** 0.5,
        "path_px": path,
    }


def motion_consistency_score(stats):
    if stats["detected"] < MOTION_TRACK_MIN_DETECTIONS:
        return 0.0
    if stats["span_px"] <= MOTION_TRACK_MIN_SPAN_PX and stats["path_px"] <= MOTION_TRACK_MIN_PATH_PX:
        return -STATIONARY_TRACK_PENALTY

    span_score = min(1.0, stats["span_px"] / max(1.0, MOTION_TRACK_MIN_SPAN_PX * 4))
    path_score = min(1.0, stats["path_px"] / max(1.0, MOTION_TRACK_MIN_PATH_PX * 4))
    return MOTION_TRACK_BONUS * max(span_score, path_score)


def is_stationary_candidate(stats):
    return (
        stats["detected"] >= MOTION_TRACK_MIN_DETECTIONS
        and stats["span_px"] <= MOTION_TRACK_MIN_SPAN_PX
        and stats["path_px"] <= MOTION_TRACK_MIN_PATH_PX
    )


def scaled_window_frames(fps):
    """MOTION_TRACK_WINDOW_FRAMES expressed for `fps`, floored at 2.

    Two points is the minimum that defines a direction at all, so the floor is
    the point below which the "motion consistent" test stops being about
    motion. Identity at REFERENCE_FPS; a missing frame rate falls back there.
    """
    if not fps or fps <= 0:
        return MOTION_TRACK_WINDOW_FRAMES
    return max(2, int(round(MOTION_TRACK_WINDOW_FRAMES * fps / REFERENCE_FPS)))


def select_motion_consistent_ball_predictions(
    predictions_by_frame,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    window_frames=MOTION_TRACK_WINDOW_FRAMES,
    require_track_support=False,
    minimum_confidence=None,
):
    """Select at most one non-stationary, motion-consistent candidate per frame.

    ``confidence_threshold`` is the evidence floor applied before stationary
    and motion analysis. When ``require_track_support`` is true, isolated
    candidates cannot become anchors. ``minimum_confidence`` is applied to the
    raw model confidence afterward. Motion consistency only ranks candidates
    that survive these gates.
    """
    candidates_by_frame = {
        frame: candidate_ball_predictions(predictions, confidence_threshold)
        for frame, predictions in predictions_by_frame.items()
    }
    selected = {}
    for frame, candidates in candidates_by_frame.items():
        if not candidates:
            selected[frame] = None
            continue

        scored_candidates = []
        for candidate in candidates:
            stats = candidate_motion_stats(frame, candidate, candidates_by_frame, window_frames)
            stationary_stats = candidate_stationary_stats(
                frame,
                candidate,
                candidates_by_frame,
                window_frames,
            )
            if is_stationary_candidate(stationary_stats):
                continue

            if require_track_support and stats["detected"] < MOTION_TRACK_MIN_DETECTIONS:
                continue

            raw_confidence = float(candidate.get("confidence", 1.0))
            if minimum_confidence is not None and raw_confidence < minimum_confidence:
                continue

            combined_score = raw_confidence + motion_consistency_score(stats)
            scored_candidates.append(
                (
                    combined_score,
                    candidate,
                )
            )

        if not scored_candidates:
            selected[frame] = None
            continue

        def score(candidate):
            return candidate[0]

        selected[frame] = max(scored_candidates, key=score)[1]
    return selected


def prediction_box(prediction):
    if prediction is None:
        return None
    try:
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return x, y, width, height


def prediction_inside_frame(prediction, frame_width, frame_height, edge_margin):
    box = prediction_box(prediction)
    if box is None or frame_width <= 0 or frame_height <= 0:
        return False

    x, y, width, height = box
    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x + width / 2
    y2 = y + height / 2

    return (
        x1 >= edge_margin
        and y1 >= edge_margin
        and x2 <= frame_width - edge_margin
        and y2 <= frame_height - edge_margin
    )


def interpolate_prediction(previous_prediction, next_prediction, alpha):
    previous_box = prediction_box(previous_prediction)
    next_box = prediction_box(next_prediction)
    if previous_box is None or next_box is None:
        return None

    interpolated = dict(previous_prediction)
    for key, previous_value, next_value in zip(
        ("x", "y", "width", "height"),
        previous_box,
        next_box,
    ):
        interpolated[key] = previous_value + (next_value - previous_value) * alpha

    previous_confidence = float(previous_prediction.get("confidence", 1.0))
    next_confidence = float(next_prediction.get("confidence", 1.0))
    interpolated["confidence"] = min(previous_confidence, next_confidence)
    interpolated["class"] = "trajectory-estimate"
    interpolated["class_name"] = "trajectory-estimate"
    return interpolated


def fill_short_trajectory_gaps(
    selected_predictions,
    frame_width,
    frame_height,
    max_gap_frames=TRAJECTORY_FILL_MAX_GAP_FRAMES,
    edge_margin_px=TRAJECTORY_FILL_EDGE_MARGIN_PX,
):
    """Interpolate short missing runs between real tracked detections.

    This only fills source frames that are present in `selected_predictions`
    and whose missing run is fully contiguous. It never extrapolates before
    the first detection, after the last detection, across sparse/untracked
    frame ranges, or near the video edge.
    """
    max_gap_frames = max(0, int(max_gap_frames))
    edge_margin_px = max(0.0, float(edge_margin_px))
    filled = dict(selected_predictions)
    previous_anchor = None
    pending_missing = []

    def maybe_fill(next_frame, next_prediction):
        nonlocal pending_missing
        if not pending_missing or previous_anchor is None or max_gap_frames <= 0:
            pending_missing = []
            return 0

        previous_frame, previous_prediction = previous_anchor
        gap_size = next_frame - previous_frame - 1
        expected_missing = list(range(previous_frame + 1, next_frame))
        if pending_missing != expected_missing or gap_size > max_gap_frames:
            pending_missing = []
            return 0

        if not prediction_inside_frame(
            previous_prediction,
            frame_width,
            frame_height,
            edge_margin_px,
        ):
            pending_missing = []
            return 0
        if not prediction_inside_frame(
            next_prediction,
            frame_width,
            frame_height,
            edge_margin_px,
        ):
            pending_missing = []
            return 0

        estimates = []
        for frame in pending_missing:
            alpha = (frame - previous_frame) / (next_frame - previous_frame)
            estimated = interpolate_prediction(previous_prediction, next_prediction, alpha)
            if estimated is None or not prediction_inside_frame(
                estimated,
                frame_width,
                frame_height,
                edge_margin_px,
            ):
                pending_missing = []
                return 0
            estimates.append((frame, estimated))

        for frame, estimated in estimates:
            filled[frame] = estimated
        filled_count = len(estimates)
        pending_missing = []
        return filled_count

    filled_count = 0
    for frame in sorted(selected_predictions):
        prediction = selected_predictions[frame]
        if prediction is None:
            pending_missing.append(frame)
            continue

        filled_count += maybe_fill(frame, prediction)
        previous_anchor = (frame, prediction)

    return filled, filled_count


def ball_csv_row(source_frame, source_fps, prediction):
    row = {
        "source_frame": source_frame,
        "timestamp_seconds": f"{source_frame / source_fps:.6f}",
        "detected": False,
        "class_name": "",
        "confidence": "",
        "x_center": "",
        "y_center": "",
        "width": "",
        "height": "",
        "x_min": "",
        "y_min": "",
        "x_max": "",
        "y_max": "",
    }

    if prediction is None:
        return row

    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])

    row.update(
        {
            "detected": True,
            "class_name": prediction_class_name(prediction),
            "confidence": f"{prediction.get('confidence', 1.0):.6f}",
            "x_center": f"{x:.3f}",
            "y_center": f"{y:.3f}",
            "width": f"{width:.3f}",
            "height": f"{height:.3f}",
            "x_min": f"{x - width / 2:.3f}",
            "y_min": f"{y - height / 2:.3f}",
            "x_max": f"{x + width / 2:.3f}",
            "y_max": f"{y + height / 2:.3f}",
        }
    )
    return row


def draw_predictions(frame, predictions):
    output_frame = frame.copy()
    frame_height, frame_width = output_frame.shape[:2]

    for prediction in predictions:
        confidence = prediction.get("confidence", 1.0)
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x = prediction["x"]
        y = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]

        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        class_name = prediction_class_name(prediction)
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output_frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output_frame


def draw_selected_prediction(frame, prediction, label_prefix="selected ball"):
    output_frame = frame.copy()
    if prediction is None:
        cv2.putText(
            output_frame,
            "no selected ball",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return output_frame

    frame_height, frame_width = output_frame.shape[:2]
    confidence = prediction.get("confidence", 1.0)
    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])

    x1 = int(x - width / 2)
    y1 = int(y - height / 2)
    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))
    center = (int(max(0, min(x, frame_width - 1))), int(max(0, min(y, frame_height - 1))))

    label = f"{label_prefix} {confidence:.2f}"
    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.circle(output_frame, center, 7, (0, 255, 255), -1)
    cv2.putText(
        output_frame,
        label,
        (x1, max(y1 - 10, 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return output_frame
