"""Fusion event engine: three detection methods + squash sequence grammar.

Methods, run independently and merged:
1. audio  — repeating-waveform impact windows (audio_events.repeating_impact_windows):
            short windows when the ball plausibly struck something. Compressed
            match audio does not separate racket from wall sounds, so audio is
            an impact-presence gate, not a wall identifier.
2. derivative — trajectory samples with the highest velocity-derivative
            changes (compute_candidates), at a relaxed significance gate.
3. parabola — wide parabolic arcs always fitted to the centroid path; every
            boundary between arcs is a trajectory-change event.

Fusion labels each merged event wall / racket / floor from geometry measured
on real footage (ui_runs/1784236711057):
- apparent ball size is the depth proxy: the ball is ~2-5x the clip median at
  a racket strike (near the player and camera) and <=~1.2x at the wall/floor.
- a wall hit must happen on the wall's image region: above the tin line
  (plus a pad, since tin hits are wall hits called OUT) and within the
  calibrated lines' x-span. Position outside vetoes the wall label — it
  cannot prove one, because players stand in front of the wall region too.
- floor bounces flip vertical velocity from downward to upward (image y
  grows downward); front-wall hits do NOT reliably reverse image y, because
  a wall bounce reverses depth and preserves vertical velocity.

A Viterbi pass then enforces squash sequence logic over the event order:
    racket -> wall -> (floor)? -> racket -> wall ...
Floor bounces only follow wall hits and precede racket hits (never
racket -> floor -> wall); volleys skip the floor (wall -> racket).
Transitions that break the grammar carry a steep penalty rather than a hard
ban so a missed detection cannot force every later label wrong.
"""

import numpy as np

from classify_events import ball_size_px, clip, clip_median_ball_size
from detect_wall_hits import (
    EVENT_WINDOW_HALF_WIDTH,
    MAX_GAP_FRAMES,
    MAX_JUMP_PX_PER_FRAME,
    MIN_DV_PX_PER_SECOND,
    MIN_TURN_DEGREES,
    SMOOTH_WINDOW,
    compute_candidates,
    detect_bounce_two_stage,
    load_detected_positions_from_rows,
    split_into_tracks,
    turn_angle_degrees,
)
from judge_call import load_calibration_lines

FUSION_DEFAULTS = {
    "arc_rms_px": 8.0,           # "wide" parabola tolerance before an arc breaks
    "arc_min_points": 4,
    "arc3d_rms_px": 3.0,         # 3D reprojection tolerance before an arc breaks
    "derivative_relax": 0.6,     # fraction of the strict significance gates
    "merge_gap_s": 0.20,         # trajectory events closer than this are one event
    "audio_window_s": 0.12,      # event-to-audio-window match tolerance
    # audio = impact presence: a matched window supports wall AND racket
    # (both are loud); silence argues against them, floor thuds are quiet.
    "audio_impact_bonus": 0.5,
    "audio_floor_penalty": 0.25,
    "silent_wall_penalty": 1.0,
    "silent_racket_penalty": 0.5,
    "audio_only_scale": 0.5,     # audio window with no trajectory event: weaker
    "audio_only_full_db": 8.0,   # prominence at which an audio-only event earns full weight
    # wall-region gate: y above tin + pad, x within the lines' span + pad.
    # The below-tin pad covers the tin face: perspective puts a ball that is
    # on the wall but below the tin top edge under the calibrated line
    # (~60 px at 1080p on the Bay Club footage).
    "wall_band_bonus": 0.4,
    "wall_band_veto": 3.0,
    "band_pad_below_px": 60.0,
    "band_pad_x_px": 60.0,
    # apparent-size depth proxy (event size / clip median). Asymmetric: only
    # rackets measure large (near the player and camera), so a big ball is
    # strong racket evidence — but far-court rackets measure small, so a
    # small ball only weakly argues against one (Bay Club GT frame 220).
    "racket_size_mid": 1.6,      # near_vote crosses 0 here
    "racket_size_half": 0.6,     # ...and saturates one half-range away
    "racket_size_weight": 1.25,
    "racket_size_far_scale": 0.25,
    "far_size_bonus": 0.5,       # small ball mildly supports wall and floor
    "floor_flip_bonus": 1.5,
    "floor_no_flip_penalty": 0.75,
    "floor_gain_penalty": 1.0,   # a floor bounce cannot speed the ball up
    "racket_gain_bonus": 1.0,
    "racket_speed_gain": 1.15,   # |v_out|/|v_in| above this means energy was added
    # side wall: lateral (x) velocity flip outside the front wall's x-span
    "side_x_flip_bonus": 0.75,
    "side_no_flip_penalty": 0.75,
    "side_out_of_span_bonus": 0.75,
    "side_in_span_penalty": 1.0,
    "silent_side_penalty": 0.5,
    # 3D evidence mode (camera present). Distances in feet; sigma scales
    # with per-event depth resolution so far contacts get honest tolerance.
    "plane_sigma_px": 3.0,        # pixel noise driving positional sigma
    "plane_sigma_min_ft": 0.4,
    "surface_near_bonus": 1.5,
    "surface_far_penalty": 1.0,
    "reflection_bonus": 0.75,     # velocity mirrors about the surface normal
    "reflection_penalty": 0.5,
    "interior_clearance_ft": 2.5, # farther than this from every surface
    "interior_racket_bonus": 1.25,
    # skip-state: any event may be labeled noise, leaving the grammar state
    # unchanged. Set above the baseline emission of an unsupported audio-only
    # event so phantom sounds get absorbed instead of phase-shifting the
    # sequence — unless the grammar needs an event there (e.g. two rackets
    # with no wall between them pull the audio window in as the wall).
    "none_score": 0.6,
    "invalid_transition_penalty": 4.0,
}

STATES = ("wall", "floor", "side", "racket")
NONE_LABEL = "none"
# racket -> (side)? -> wall -> (side)? -> (floor)? -> racket; volleys skip
# the floor, and the ball may clip a side wall on the way to or from the
# front wall (Bay Club GT frame 135: racket -> side -> front wall).
# floor -> side is deliberately absent: with it, floor -> side -> floor
# chains through pairwise-legal steps into an impossible rally (a floor
# bounce cannot follow a floor bounce without a racket, side wall or not).
ALLOWED_TRANSITIONS = {
    ("racket", "wall"),
    ("wall", "floor"),
    ("wall", "racket"),
    ("floor", "racket"),
    ("racket", "side"),
    ("side", "wall"),
    ("wall", "side"),
    ("side", "floor"),
    ("side", "racket"),
}


def merge_fusion_config(config):
    merged = dict(FUSION_DEFAULTS)
    if config:
        merged.update(config)
    return merged


# --- Method 3: wide parabolic arcs ---------------------------------------


def _arc_fit(times, points):
    """Per-axis quadratic fit (degree drops when the arc is tiny); returns
    (coeff_x, coeff_y, rms_residual_px)."""
    degree = 2 if len(times) >= 4 else 1
    t0 = times[0]
    t = times - t0
    coeff_x = np.polyfit(t, points[:, 0], degree)
    coeff_y = np.polyfit(t, points[:, 1], degree)
    dx = points[:, 0] - np.polyval(coeff_x, t)
    dy = points[:, 1] - np.polyval(coeff_y, t)
    return coeff_x, coeff_y, float(np.sqrt(np.mean(dx * dx + dy * dy)))


def _arc_velocity(times, points, start, end, at):
    """Fitted velocity vector of arc [start:end) evaluated at sample `at`."""
    coeff_x, coeff_y, _ = _arc_fit(times[start:end], points[start:end])
    t = times[at] - times[start]
    return np.array(
        [np.polyval(np.polyder(coeff_x), t), np.polyval(np.polyder(coeff_y), t)]
    )


def segment_into_arcs(times, points, rms_px, min_points):
    """Greedy maximal-arc segmentation: grow each arc until adding the next
    sample pushes the quadratic fit's rms residual past rms_px."""
    arcs = []
    start = 0
    count = len(times)
    while start < count:
        end = min(start + min_points, count)
        while end < count:
            _, _, rms = _arc_fit(times[start : end + 1], points[start : end + 1])
            if rms > rms_px:
                break
            end += 1
        arcs.append((start, end))
        start = end
    return arcs


def parabolic_arc_events(frames, timestamps, positions, tracks, cfg):
    """Every boundary between adjacent fitted arcs, with fitted velocities."""
    events = []
    min_points = cfg["arc_min_points"]
    for track_start, track_end in tracks:
        if track_end - track_start < 2 * min_points:
            continue
        times = timestamps[track_start:track_end]
        points = positions[track_start:track_end]
        arcs = segment_into_arcs(times, points, cfg["arc_rms_px"], min_points)
        for k in range(1, len(arcs)):
            a_start, a_end = arcs[k - 1]
            b_start, b_end = arcs[k]
            if a_end - a_start < min_points or b_end - b_start < min_points:
                continue
            v_in = _arc_velocity(times, points, a_start, a_end, a_end - 1)
            v_out = _arc_velocity(times, points, b_start, b_end, b_start)
            events.append(
                _make_event(
                    track_start + b_start, frames, timestamps, positions, v_in, v_out, "parabola"
                )
            )
    return events


# --- Method 2: highest derivative changes ---------------------------------


def _local_velocities(timestamps, positions, index, track_start, track_end, span=2):
    lo = max(track_start, index - span)
    hi = min(track_end - 1, index + span)
    v_in = v_out = None
    if index > lo:
        v_in = (positions[index] - positions[lo]) / (timestamps[index] - timestamps[lo])
    if hi > index:
        v_out = (positions[hi] - positions[index]) / (timestamps[hi] - timestamps[index])
    return v_in, v_out


def derivative_events(frames, timestamps, positions, tracks, cfg):
    """Velocity-change candidates above a relaxed significance gate — the
    fusion and grammar passes do the filtering the strict gate used to."""
    relax = cfg["derivative_relax"]
    events = []
    for candidate in compute_candidates(frames, timestamps, positions, tracks, SMOOTH_WINDOW):
        if (
            candidate["dv_magnitude"] < relax * MIN_DV_PX_PER_SECOND
            or candidate["turn_degrees"] < relax * MIN_TURN_DEGREES
        ):
            continue
        index = candidate["sample_global_index"]
        track_start, track_end = tracks[candidate["track_index"]]
        v_in, v_out = _local_velocities(timestamps, positions, index, track_start, track_end)
        if v_in is None or v_out is None:
            continue
        events.append(_make_event(index, frames, timestamps, positions, v_in, v_out, "derivative"))
    return events


def _make_event(index, frames, timestamps, positions, v_in, v_out, method):
    v_in = np.asarray(v_in, dtype=np.float64)
    v_out = np.asarray(v_out, dtype=np.float64)
    return {
        "index": int(index),
        "frame": int(frames[index]),
        "time": float(timestamps[index]),
        "x": float(positions[index][0]),
        "y": float(positions[index][1]),
        "v_in": v_in,
        "v_out": v_out,
        "speed_before": float(np.linalg.norm(v_in)),
        "speed_after": float(np.linalg.norm(v_out)),
        "dv_magnitude": float(np.linalg.norm(v_out - v_in)),
        "turn_degrees": turn_angle_degrees(v_in, v_out),
        "methods": {method},
    }


def merge_trajectory_events(parabola, derivative, merge_gap_s):
    """Events from both methods within merge_gap_s collapse into one; the
    parabola representative wins (fitted velocities are stabler) and the
    merged event remembers which methods corroborated it."""
    fitted = {"parabola", "ballistic"}
    merged = []
    for event in sorted(parabola + derivative, key=lambda item: item["time"]):
        if merged and event["time"] - merged[-1]["time"] <= merge_gap_s:
            keeper = merged[-1]
            if fitted & event["methods"] and not fitted & keeper["methods"]:
                event["methods"] |= keeper["methods"]
                merged[-1] = event
            else:
                keeper["methods"] |= event["methods"]
        else:
            merged.append(event)
    return merged


# --- Fusion: audio evidence + physics emissions + sequence grammar --------


def _match_window(time, audio_windows, tolerance_s):
    best = None
    for window in audio_windows or []:
        offset = abs(window["time_seconds"] - time)
        if offset <= tolerance_s and (best is None or offset < abs(best["time_seconds"] - time)):
            best = window
    return best


class WallRegion:
    """Where in the image a ball could be touching the front wall.

    `wall(x, y)` — above the tin line (+ pad, so tin-face hits stay wall
    hits) and within the calibrated lines' x-span. Necessary, never
    sufficient: players stand in front of the wall region, so inside earns
    only a small bonus while outside is a strong veto.
    `in_x_span(x)` — within the lines' horizontal span; side walls live
    outside it.
    """

    def __init__(self, top_line, bottom_line, cfg):
        xs = [top_line.left.x, top_line.right.x, bottom_line.left.x, bottom_line.right.x]
        self._x_lo = min(xs) - cfg["band_pad_x_px"]
        self._x_hi = max(xs) + cfg["band_pad_x_px"]
        self._pad_below = cfg["band_pad_below_px"]
        self._tin = (
            (bottom_line.left.x, bottom_line.left.y),
            (bottom_line.right.x, bottom_line.right.y),
        )

    def in_x_span(self, x):
        return self._x_lo <= x <= self._x_hi

    def wall(self, x, y):
        if not self.in_x_span(x):
            return False
        (bx1, by1), (bx2, by2) = self._tin
        if bx2 == bx1:
            tin_y = max(by1, by2)
        else:
            fraction = min(1.0, max(0.0, (x - bx1) / (bx2 - bx1)))
            tin_y = by1 + fraction * (by2 - by1)
        return y <= tin_y + self._pad_below


def make_wall_region(calibration, cfg):
    if not calibration:
        return None
    try:
        top_line, bottom_line = load_calibration_lines(calibration)
    except (ValueError, KeyError, TypeError):
        return None
    return WallRegion(top_line, bottom_line, cfg)


def _audio_scores(event, audio_available, cfg):
    scores = {"wall": 0.0, "floor": 0.0, "side": 0.0, "racket": 0.0}
    window = event.get("audio_window")

    if audio_available:
        if window is not None:
            bonus = cfg["audio_impact_bonus"]
            if event.get("audio_only"):
                # Prominence-weighted so that when the grammar must pull one
                # of several unclaimed windows into a missing slot, the
                # loudest impact wins rather than an arbitrary tie-break.
                bonus *= cfg["audio_only_scale"] * min(
                    1.0, float(window["score"]) / cfg["audio_only_full_db"]
                )
            scores["wall"] += bonus
            scores["racket"] += bonus
            scores["side"] += bonus
            scores["floor"] -= cfg["audio_floor_penalty"]
        else:
            scores["wall"] -= cfg["silent_wall_penalty"]
            scores["racket"] -= cfg["silent_racket_penalty"]
            scores["side"] -= cfg["silent_side_penalty"]
    return scores


def _emission_scores(event, audio_available, wall_region, cfg):
    scores = _audio_scores(event, audio_available, cfg)

    if wall_region is not None and event.get("x") is not None:
        if wall_region.wall(event["x"], event["y"]):
            scores["wall"] += cfg["wall_band_bonus"]
        else:
            scores["wall"] -= cfg["wall_band_veto"]
        if wall_region.in_x_span(event["x"]):
            scores["side"] -= cfg["side_in_span_penalty"]
        else:
            scores["side"] += cfg["side_out_of_span_bonus"]

    ratio = event.get("size_ratio")
    if ratio is not None:
        near_vote = clip((ratio - cfg["racket_size_mid"]) / cfg["racket_size_half"])
        weight = cfg["racket_size_weight"]
        if near_vote < 0:
            weight *= cfg["racket_size_far_scale"]
        scores["racket"] += weight * near_vote
        if near_vote < 0:
            scores["wall"] += cfg["far_size_bonus"] * -near_vote
            scores["floor"] += cfg["far_size_bonus"] * -near_vote

    v_in, v_out = event.get("v_in"), event.get("v_out")
    if v_in is not None and v_out is not None:
        if v_in[1] > 0 and v_out[1] < 0:  # image y grows downward: down -> up
            scores["floor"] += cfg["floor_flip_bonus"]
        else:
            scores["floor"] -= cfg["floor_no_flip_penalty"]
        if v_in[0] * v_out[0] < 0:  # lateral reversal
            scores["side"] += cfg["side_x_flip_bonus"]
        else:
            scores["side"] -= cfg["side_no_flip_penalty"]
        gain = event["speed_after"] / (event["speed_before"] + 1e-9)
        if gain >= cfg["racket_speed_gain"]:
            # Only the racket adds energy: a bounce off any passive surface
            # (dead squash ball, restitution ~0.5) always loses speed.
            scores["racket"] += cfg["racket_gain_bonus"]
            scores["floor"] -= cfg["floor_gain_penalty"]
    return scores


def _surface_geometry(point_ft):
    x, y, z = (float(c) for c in point_ft)
    side_normal = np.array([1.0, 0.0, 0.0]) if x <= 10.5 else np.array([-1.0, 0.0, 0.0])
    return [
        ("floor", np.array([0.0, 0.0, 1.0]), z),
        ("wall", np.array([0.0, 1.0, 0.0]), y),
        ("side", side_normal, min(x, 21.0 - x)),
    ]


def _positional_sigma_ft(camera, point_ft, normal, cfg):
    """Pixel noise mapped to feet at this point, inflated when the surface
    normal is nearly parallel to the viewing ray (poorly observed axis)."""
    point = np.asarray(point_ft, dtype=float)
    transverse = cfg["plane_sigma_px"] * camera.depth_ft(point) / camera.focal_px
    _, direction = camera.ray(camera.project(point))
    perpendicular = normal - (normal @ direction) * direction
    observability = max(0.2, float(np.linalg.norm(perpendicular)))
    return max(cfg["plane_sigma_min_ft"], transverse / observability)


def _emission_scores_3d(event, audio_available, camera, cfg):
    scores = _audio_scores(event, audio_available, cfg)
    contact = event["contact_3d"]
    point = np.asarray(contact["point_ft"], dtype=float)
    v_in = np.asarray(contact["v_in_ft_s"], dtype=float)
    v_out = np.asarray(contact["v_out_ft_s"], dtype=float)
    speed_in = float(np.linalg.norm(v_in))
    speed_out = float(np.linalg.norm(v_out))

    min_distance = None
    try:
        surfaces = [
            (state, normal, distance, _positional_sigma_ft(camera, point, normal, cfg))
            for state, normal, distance in _surface_geometry(point)
        ]
    except ValueError:  # contact point projected behind the camera: no 3D vote
        return scores
    for state, normal, distance, sigma in surfaces:
        near = float(np.exp(-0.5 * (distance / sigma) ** 2))
        scores[state] += cfg["surface_near_bonus"] * near
        scores[state] -= cfg["surface_far_penalty"] * (1.0 - near)
        if speed_in > 1e-6 and speed_out > 1e-6:
            reflected = v_in - 2.0 * float(v_in @ normal) * normal
            alignment = float(reflected @ v_out) / (
                np.linalg.norm(reflected) * speed_out)
            restitution = speed_out / speed_in
            if alignment > 0 and restitution <= 1.05:
                scores[state] += cfg["reflection_bonus"] * near * alignment
            else:
                scores[state] -= cfg["reflection_penalty"] * near
        min_distance = distance if min_distance is None else min(min_distance, distance)

    if min_distance is not None and min_distance > cfg["interior_clearance_ft"]:
        scores["racket"] += cfg["interior_racket_bonus"]
    if speed_in > 1e-6 and speed_out / speed_in >= cfg["racket_speed_gain"]:
        scores["racket"] += cfg["racket_gain_bonus"]
        scores["floor"] -= cfg["floor_gain_penalty"]

    # Debuggability contract from the spec: the evidence that scored this
    # event must survive into the hit's signals (see step below).
    event["evidence_3d"] = {
        "mode": "3d",
        "plane_distance_ft": {state: round(distance, 3)
                              for state, _, distance, _ in surfaces},
        "sigma_ft": {state: round(sigma, 3)
                     for state, _, _, sigma in surfaces},
        "restitution": (round(speed_out / speed_in, 3)
                        if speed_in > 1e-6 else None),
    }
    return scores


def decode_sequence(emissions, cfg):
    """Skip-state Viterbi over the squash grammar.

    Each event is either labeled with a state or with NONE_LABEL (noise): a
    none event scores cfg["none_score"] and leaves the grammar state
    unchanged, so a phantom event cannot phase-shift the sequence, while a
    grammar gap (racket ... racket with no wall) still pulls the best
    in-between event into the missing slot. The DP state is the last real
    label assigned so far ("__start__" before any).
    """
    if not emissions:
        return []
    penalty = cfg["invalid_transition_penalty"]
    none_score = cfg["none_score"]
    start = "__start__"

    previous_scores = {start: 0.0}
    backpointers = []
    for emission in emissions:
        row = {}
        back = {}
        for prev_state, prev_score in previous_scores.items():
            candidate = prev_score + none_score
            if prev_state not in row or candidate > row[prev_state]:
                row[prev_state] = candidate
                back[prev_state] = (prev_state, NONE_LABEL)
            for state in STATES:
                transition = 0.0
                if prev_state != start and (prev_state, state) not in ALLOWED_TRANSITIONS:
                    transition = -penalty
                candidate = prev_score + transition + emission[state]
                if state not in row or candidate > row[state]:
                    row[state] = candidate
                    back[state] = (prev_state, state)
        previous_scores = row
        backpointers.append(back)

    state = max(previous_scores, key=previous_scores.get)
    labels = []
    for back in reversed(backpointers):
        state, label = back[state]
        labels.append(label)
    labels.reverse()
    return labels


def detect_events_fused(
    rows,
    audio_windows=None,
    calibration=None,
    wall_x_range=None,
    config=None,
    max_gap=MAX_GAP_FRAMES,
    camera=None,
):
    """Full fusion pass over one clip's tracking rows.

    Returns judge-ready hits sorted by frame, each labeled with event_type
    wall / floor / racket (wall hits carry two-stage impact fits when a
    calibration exists). Unmatched audio windows with no trajectory nearby
    still surface as audio-only events so nothing audible is dropped.

    camera: optional court_model.CameraModel; when present the parabola
    trajectory source is replaced by 3D ballistic segmentation and events
    gain contact_3d.
    """
    cfg = merge_fusion_config(config)
    frames, timestamps, positions, _ = load_detected_positions_from_rows(rows)
    wall_region = make_wall_region(calibration, cfg)
    audio_available = audio_windows is not None
    results = {int(row["source_frame"]): row for row in rows}
    median_size = clip_median_ball_size(results)

    events = []
    if len(frames) >= 3:
        tracks = split_into_tracks(frames, positions, max_gap, MAX_JUMP_PX_PER_FRAME)
        if camera is not None:
            from ballistic import arc_boundary_events

            trajectory_events = arc_boundary_events(
                frames, timestamps, positions, tracks, camera, cfg)
        else:
            trajectory_events = parabolic_arc_events(
                frames, timestamps, positions, tracks, cfg)
        events = merge_trajectory_events(
            trajectory_events,
            derivative_events(frames, timestamps, positions, tracks, cfg),
            cfg["merge_gap_s"],
        )

    for event in events:
        event["audio_window"] = _match_window(event["time"], audio_windows, cfg["audio_window_s"])
        size = ball_size_px(results, event["frame"], radius=2) if median_size else None
        event["size_ratio"] = size / median_size if size is not None else None

    # Audio windows nobody claimed: surface them anyway (recall-first). They
    # have no trajectory, so only audio evidence votes on their label.
    if audio_windows:
        span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        fps = (frames[-1] - frames[0]) / span if span > 0 else 30.0
        claimed_times = [event["time"] for event in events]
        for window in audio_windows:
            if any(abs(t - window["time_seconds"]) <= cfg["audio_window_s"] for t in claimed_times):
                continue
            events.append(
                {
                    "index": None,
                    "frame": int(round(window["time_seconds"] * fps)),
                    "time": float(window["time_seconds"]),
                    "x": None,
                    "y": None,
                    "v_in": None,
                    "v_out": None,
                    "speed_before": None,
                    "speed_after": None,
                    "dv_magnitude": None,
                    "turn_degrees": None,
                    "methods": {"audio"},
                    "audio_window": window,
                    "audio_only": True,
                }
            )
        events.sort(key=lambda item: item["time"])

    emissions = [
        _emission_scores_3d(event, audio_available, camera, cfg)
        if camera is not None and event.get("contact_3d")
        else _emission_scores(event, audio_available, wall_region, cfg)
        for event in events
    ]
    labels = decode_sequence(emissions, cfg)

    hits = []
    for event, label, emission in zip(events, labels, emissions):
        if label == NONE_LABEL:
            continue  # noise: absorbed by the skip state, not an event
        hit = {
            "hit_frame": event["frame"],
            "timestamp_seconds": event["time"],
            "event_type": "side_wall" if label == "side" else label,
            "method": "fusion",
            "methods": sorted(event["methods"]),
            "score": float(emission[label]),
            "dv_magnitude": event["dv_magnitude"],
            "speed_before": event["speed_before"],
            "speed_after": event["speed_after"],
            "turn_degrees": event["turn_degrees"],
            "after_gap": False,
        }
        if event["x"] is not None:
            hit["candidate_x"] = event["x"]
            hit["candidate_y"] = event["y"]
        signals = {
            "size_ratio": event.get("size_ratio"),
            "audio_score": None,
            "audio_offset_s": None,
        }
        window = event.get("audio_window")
        if window is not None:
            signals.update(
                audio_score=float(window["score"]),
                audio_offset_s=float(window["time_seconds"] - event["time"]),
                audio_cluster=int(window["cluster_id"]),
                audio_cluster_size=int(window["cluster_size"]),
            )
        hit["signals"] = signals
        if event.get("contact_3d"):
            hit["contact_3d"] = event["contact_3d"]
            hit["signals"]["evidence_3d"] = event.get("evidence_3d")
        if event.get("audio_only"):
            hit["source"] = "audio"
            hit["audio_assisted"] = True
            for key in ("dv_magnitude", "speed_before", "speed_after", "turn_degrees"):
                hit.pop(key)

        if label == "floor" and event.get("contact_3d"):
            point = event["contact_3d"]["point_ft"]
            hit["court_position_ft"] = {"x": float(point[0]), "y": float(point[1])}

        if label == "wall" and event.get("contact_3d") and camera is not None:
            contact = event["contact_3d"]
            point = np.asarray(contact["point_ft"], dtype=float)
            wall_point = point.copy()
            wall_point[1] = 0.0  # snap onto the front-wall plane for judging
            try:
                pixel = camera.project(wall_point)
                from court_model import distort_point
                hit["impact_x"], hit["impact_y"] = distort_point(
                    pixel, camera.distortion)
                hit["impact_time"] = contact["time"]
                hit["impact_height_ft"] = float(point[2])
            except ValueError:
                pass  # fall through to the 2D impact fit below
        if label == "wall" and calibration and event["index"] is not None \
                and "impact_x" not in hit:
            lo = max(0, event["index"] - EVENT_WINDOW_HALF_WIDTH)
            hi = min(len(frames), event["index"] + EVENT_WINDOW_HALF_WIDTH + 1)
            diagnostics = {}
            result = detect_bounce_two_stage(
                (frames[lo:hi], timestamps[lo:hi], positions[lo:hi]),
                calibration,
                diagnostics_out=diagnostics,
            )
            hit["diagnostics"] = diagnostics
            if result is not None:
                hit["impact_x"], hit["impact_y"] = result.impact_xy
                hit["impact_time"] = result.impact_t
                hit["impact_frame"] = float(frames[lo + result.impact_index])
        hits.append(hit)

    return sorted(hits, key=lambda hit: hit["hit_frame"])
