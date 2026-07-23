"""Shared ball-tracking constants and pure helpers.

Importing this module has no side effects (no network clients, no model
loads); anything here is safe for the Flask app, the CLIs, and tests.
"""

import cv2


CONFIDENCE_THRESHOLD = 0.25
BALL_CLASS_NAMES = {"ball", "squash ball", "squash-ball", "squash_ball"}
MOTION_TRACK_WINDOW_FRAMES = 5
MOTION_TRACK_MIN_DETECTIONS = 3
MOTION_TRACK_MIN_SPAN_PX = 8.0
MOTION_TRACK_MIN_PATH_PX = 10.0
MOTION_TRACK_MAX_STEP_PX = 80.0
MOTION_TRACK_STATIONARY_RADIUS_PX = 8.0
MOTION_TRACK_BONUS = 0.30
STATIONARY_TRACK_PENALTY = 0.40
TRAJECTORY_FILL_MAX_GAP_FRAMES = 4
TRAJECTORY_FILL_EDGE_MARGIN_PX = 24.0
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


def select_motion_consistent_ball_predictions(
    predictions_by_frame,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    window_frames=MOTION_TRACK_WINDOW_FRAMES,
):
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

            scored_candidates.append(
                (
                    float(candidate.get("confidence", 1.0))
                    + motion_consistency_score(stats),
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
