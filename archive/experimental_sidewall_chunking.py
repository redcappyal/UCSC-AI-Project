"""Archived sidewall-through front-wall chunking experiment.

This module is intentionally not imported by the app or training pipeline.
It preserves the July 2026 experiment where horizontal sidewall excursions
were skipped without ending the current front-wall chunk.
"""

from judge_call import Point

from bounce_gb_model_detector import (
    DEFAULT_FRONT_WALL_CHUNK_PAD_FRACTION,
    DEFAULT_FRONT_WALL_CHUNK_PAD_PX,
    DEFAULT_WALL_VISIT_GAP_FRAMES,
    collapse_wall_area_duplicates,
    env_float,
    finite_point,
    inside_wall_corners_gate,
    inside_x_range,
    is_plausible_ball_step,
    normalize_x_range,
    point_progress_on_line,
    wall_corners_y_bounds,
)


def experimental_front_wall_chunk_region(x, y, wall_gate, wall_x_range):
    """Classify a point as front wall, sidewall, or below the wall."""
    pad_px = env_float(
        "BOUNCE_GB_FRONT_WALL_CHUNK_PAD_PX",
        DEFAULT_FRONT_WALL_CHUNK_PAD_PX,
    )
    pad_fraction = env_float(
        "BOUNCE_GB_FRONT_WALL_CHUNK_PAD_FRACTION",
        DEFAULT_FRONT_WALL_CHUNK_PAD_FRACTION,
    )

    if wall_gate is None:
        normalized = normalize_x_range(wall_x_range)
        line_width = (
            normalized[1] - normalized[0]
            if normalized is not None
            else 0.0
        )
        return (
            "front_wall"
            if inside_x_range(
                x,
                wall_x_range,
                max(pad_px, pad_fraction * line_width),
            )
            else "sidewall"
        )

    point = Point(float(x), float(y))
    if wall_gate[0] == "wall_corners":
        horizontally_inside_wall = inside_wall_corners_gate(
            x,
            y,
            wall_gate[1],
            horizontal_only=True,
            pad_px=pad_px,
            pad_fraction=pad_fraction,
        )
        if not horizontally_inside_wall:
            return "sidewall"

        bottom_line = wall_gate[3] if len(wall_gate) > 3 else None
        if bottom_line is not None:
            bottom_margin = -bottom_line.signed_distance_below(point)
            return "front_wall" if bottom_margin >= -pad_px else "below_wall"

        _, bottom_y = wall_corners_y_bounds(wall_gate[1])
        return "front_wall" if float(y) <= bottom_y + pad_px else "below_wall"

    _, _, bottom_line = wall_gate
    bottom_margin = -bottom_line.signed_distance_below(point)
    if bottom_margin < -pad_px:
        return "below_wall"

    progress = point_progress_on_line(point, bottom_line)
    return (
        "front_wall"
        if -pad_fraction <= progress <= 1.0 + pad_fraction
        else "sidewall"
    )


def collapse_front_wall_chunks_through_sidewalls(
    candidates,
    parsed_rows,
    wall_gate,
    wall_x_range,
    fallback_gap=DEFAULT_WALL_VISIT_GAP_FRAMES,
):
    """Keep one chunk open across sidewall frames but exclude those frames."""
    if not candidates:
        return []
    if wall_gate is None:
        return collapse_wall_area_duplicates(candidates, fallback_gap)

    candidates_by_frame = {
        int(candidate["hit_frame"]): candidate
        for candidate in candidates
    }
    chunks = []
    current_candidates = []
    current_frames = []
    last_accepted_frame = None
    last_accepted_x = None
    last_accepted_y = None

    for frame in sorted(parsed_rows):
        row = parsed_rows[frame]
        if not finite_point(row):
            continue
        if not is_plausible_ball_step(
            last_accepted_frame,
            last_accepted_x,
            last_accepted_y,
            frame,
            row["x"],
            row["y"],
        ):
            continue

        region = experimental_front_wall_chunk_region(
            row["x"],
            row["y"],
            wall_gate,
            wall_x_range,
        )
        if region == "sidewall":
            continue
        if region != "front_wall":
            if current_frames:
                chunks.append((current_frames, current_candidates))
            current_candidates = []
            current_frames = []
            last_accepted_frame = None
            last_accepted_x = None
            last_accepted_y = None
            continue

        current_frames.append(int(frame))
        last_accepted_frame = int(frame)
        last_accepted_x = float(row["x"])
        last_accepted_y = float(row["y"])
        candidate = candidates_by_frame.get(int(frame))
        if candidate is not None:
            current_candidates.append(candidate)

    if current_frames:
        chunks.append((current_frames, current_candidates))

    picked = []
    for chunk_frames, chunk_candidates in chunks:
        if not chunk_candidates:
            continue
        best = dict(max(chunk_candidates, key=lambda item: item["score"]))
        best["wall_visit_candidate_count"] = len(chunk_candidates)
        best["wall_visit_frames"] = [
            int(item["hit_frame"])
            for item in chunk_candidates
        ]
        best["front_wall_chunk_start_frame"] = int(chunk_frames[0])
        best["front_wall_chunk_end_frame"] = int(chunk_frames[-1])
        best["front_wall_chunk_frame_count"] = len(chunk_frames)
        picked.append(best)

    return sorted(picked, key=lambda item: item["hit_frame"])
