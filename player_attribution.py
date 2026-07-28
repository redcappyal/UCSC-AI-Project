"""Serve-anchored player attribution (spec §4.4).

Within a rally, front-wall hits strictly alternate (squash rule), so per-hit
identity is unnecessary: the only question is which track served each rally.
The resolver answers it from raw pixels — the track whose live sample is
nearest the last detected ball position shortly before the rally's first
front-wall hit. No homography, no calibration.
"""

import math

SERVE_LOOKBACK_S = 2.0        # ball row search window before a rally's first hit
MAX_SERVE_TRACK_GAP_S = 0.75  # a track vote needs a live sample this fresh


def _row_time(row):
    return float(row.get("timestamp_seconds", 0.0))


def _row_detected(row):
    detected = row.get("detected")
    if isinstance(detected, str):
        return detected.strip().lower() == "true"
    return bool(detected)


def build_serve_resolver(samples_by_track, ball_rows):
    """-> resolver(rally_hits) -> "A" | "B" | None.

    Stateless with respect to rallies: matches by the first hit's timestamp,
    so it can be called from any assign_front_wall_hit_players invocation.
    """
    detected_rows = sorted(
        (row for row in ball_rows if _row_detected(row)), key=_row_time
    )
    live = {
        track: [s for s in samples if not s.coasted]
        for track, samples in samples_by_track.items()
    }

    def resolver(rally_hits):
        if not rally_hits:
            return None
        first_hit_t = float(rally_hits[0].get("timestamp_seconds", 0.0))
        window = [row for row in detected_rows
                  if first_hit_t - SERVE_LOOKBACK_S <= _row_time(row) < first_hit_t]
        if not window:
            return None
        ball = window[-1]
        ball_t = _row_time(ball)
        ball_xy = (float(ball["x_center"]), float(ball["y_center"]))

        best_track, best_distance = None, None
        for track in ("A", "B"):
            candidates = [s for s in live.get(track, [])
                          if abs(s.t_s - ball_t) <= MAX_SERVE_TRACK_GAP_S]
            if not candidates:
                return None  # a missing vote makes the comparison meaningless
            nearest = min(candidates, key=lambda s: abs(s.t_s - ball_t))
            distance = math.hypot(nearest.foot_px[0] - ball_xy[0],
                                  nearest.foot_px[1] - ball_xy[1])
            if best_distance is None or distance < best_distance:
                best_track, best_distance = track, distance
        return best_track

    return resolver


def rally_identity_confidences(ambiguity_times, rallies):
    """Confidence that identity survived the break BEFORE each rally.

    confidence = 1 - min(1, ambiguous_events_in_break); rally 1 has no
    preceding break -> None. Deliberately blunt for v1 (spec: collected
    silently, debugged later)."""
    confidences = {}
    previous_end = None
    for rally in rallies:
        number = rally["rally_number"]
        start = float(rally.get("start_time_seconds", 0.0))
        if previous_end is None:
            confidences[number] = None
        else:
            count = sum(1 for t in ambiguity_times if previous_end < t <= start)
            confidences[number] = max(0.0, 1.0 - float(count))
        previous_end = float(rally.get("end_time_seconds", start))
    return confidences


def build_players_v1(assignment, tracker_stats, detector_backend,
                     serve_crop_relpath=None, player_names=None):
    observed = any(
        rally.get("server_source") == "observed"
        for rally in assignment.get("rallies", [])
    )
    score = {"1": 0, "2": 0}
    rallies = []
    for rally in assignment.get("rallies", []):
        winner = rally.get("winner_player_number")
        if winner in (1, 2):
            score[str(winner)] += 1
        rallies.append({
            "rally_number": rally.get("rally_number"),
            "server_player_number": rally.get("server_player_number"),
            "server_track": rally.get("server_track"),
            "server_source": rally.get("server_source"),
            "winner_player_number": winner,
            "winner_source": rally.get("winner_source"),
            "winner_crosscheck_agrees": rally.get("winner_crosscheck_agrees"),
            "identity_confidence": rally.get("identity_confidence"),
            "score_after": dict(score),
        })
    return {
        "attribution_backend": "observed" if observed else "assumed",
        "detector_backend": detector_backend,
        "tracker": tracker_stats,
        "rallies": rallies,
        "serve_crop": serve_crop_relpath,
        "player_names": dict(player_names) if player_names else {"A": None, "B": None},
    }


def serve_crop_target(assignment, samples_by_track):
    """-> (frame_idx, TrackSample) for the first observed server, or None.

    Anchors on the first rally with server_source == "observed" and a valid
    server_track — same anchor rule as index.html's attributionAnchor() —
    not unconditionally rally 1, since rally 1's serve can go unobserved
    while a later rally's is."""
    rallies = assignment.get("rallies") or []
    anchor = next(
        (rally for rally in rallies
         if rally.get("server_source") == "observed"
         and rally.get("server_track") in ("A", "B")),
        None,
    )
    if anchor is None:
        return None
    track = anchor.get("server_track")
    serve_t = float(anchor.get("start_time_seconds", 0.0))
    live = [s for s in samples_by_track.get(track, []) if not s.coasted]
    if not live:
        return None
    nearest = min(live, key=lambda s: abs(s.t_s - serve_t))
    return nearest.frame_idx, nearest
