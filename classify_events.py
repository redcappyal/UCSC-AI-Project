"""Classify detected impact events as wall bounces vs racket hits.

Runs after detection (trajectory detectors plus audio-rescued events); this
layer only labels each event, never adds or moves one. Three verification
signals, each voting in [-1, +1] (+1 = wall, -1 = racket) or abstaining
(None) when unavailable:

- audio: wall hits are much louder than racket hits
- size: the ball is farther from camera (smaller bbox) at the front wall
- timing: the racket->wall gap is shorter than the wall->next-racket gap

The combined wall_score is the weight-normalized sum over available votes, so
classification degrades gracefully: a clip with no audio track and a single
event still classifies on size alone, and with no signals at all the event is
"unknown" and judged exactly as before.
"""

import math


CLASSIFY_DEFAULTS = {
    # 0.15s absorbs display-frame quantization when no sub-frame impact fit
    # exists (a real wall peak landed 0.111s from its event's timestamp).
    "audio_match_tolerance_s": 0.15,
    # Real match recordings are compressed: a genuine wall impact measured
    # only +9.5 dB above the clip median, so the vote crosses 0 at 8 dB.
    "audio_mid_db": 8.0,            # dB-above-median where the audio vote crosses 0
    "audio_half_range_db": 4.0,
    "size_frame_radius": 2,         # detected frames around event used for robust size
    "size_mid_ratio": 1.0,          # event_size / clip_median_size where vote crosses 0
    "size_half_range": 0.35,
    "timing_mid_s": 0.9,            # gap-to-previous-event where vote crosses 0
    "timing_half_range_s": 0.5,
    "weights": {"audio": 0.4, "size": 0.35, "timing": 0.25},
    "wall_threshold": 0.15,         # score >= -> "wall"
    "racket_threshold": -0.15,      # score <= -> "racket"; between -> "unknown"
}

EVENT_WALL = "wall"
EVENT_RACKET = "racket"
EVENT_UNKNOWN = "unknown"


def merge_config(config):
    merged = dict(CLASSIFY_DEFAULTS)
    merged["weights"] = dict(CLASSIFY_DEFAULTS["weights"])
    if config:
        for key, value in config.items():
            if key == "weights" and isinstance(value, dict):
                merged["weights"].update(value)
            else:
                merged[key] = value
    return merged


def clip(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def _row_size(row):
    """Apparent ball size sqrt(width*height) for a results row, or None."""
    if not row:
        return None
    detected = row.get("detected")
    if isinstance(detected, str):
        detected = detected.strip().lower() in {"true", "1", "yes", "y"}
    if not detected:
        return None
    try:
        width = float(row["width"])
        height = float(row["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or not (math.isfinite(width) and math.isfinite(height)):
        return None
    return math.sqrt(width * height)


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def ball_size_px(results, frame, radius):
    """Median size over detected rows within +/-radius frames of frame, or None."""
    sizes = []
    for offset in range(-radius, radius + 1):
        size = _row_size(results.get(frame + offset))
        if size is not None:
            sizes.append(size)
    return _median(sizes)


def clip_median_ball_size(results):
    """Median size over every detected row in the clip, or None."""
    return _median([size for size in (_row_size(row) for row in results.values()) if size is not None])


def compute_event_signals(hits, results, audio_candidates, fps, config):
    """Per-hit signal dicts; any field is None when that signal is unavailable."""
    from audio_events import match_audio_peak

    median_size = clip_median_ball_size(results)
    ordered = sorted(range(len(hits)), key=lambda i: hits[i]["hit_frame"])
    event_times = {
        i: float(hits[i].get("impact_time", hits[i]["timestamp_seconds"])) for i in ordered
    }

    signals = [None] * len(hits)
    for position, index in enumerate(ordered):
        hit = hits[index]
        event_time = event_times[index]

        audio_score = audio_rms = audio_offset = None
        if audio_candidates is not None:
            peak = match_audio_peak(
                event_time, audio_candidates, config["audio_match_tolerance_s"]
            )
            if peak is not None:
                audio_score = float(peak["score"])
                audio_rms = float(peak["rms"])
                audio_offset = float(peak["time_seconds"] - event_time)

        size = ball_size_px(results, int(hit["hit_frame"]), config["size_frame_radius"])
        size_ratio = None
        if size is not None and median_size:
            size_ratio = size / median_size

        gap_prev = gap_next = None
        if position > 0:
            gap_prev = event_time - event_times[ordered[position - 1]]
        if position < len(ordered) - 1:
            gap_next = event_times[ordered[position + 1]] - event_time

        signals[index] = {
            "audio_score": audio_score,
            "audio_rms": audio_rms,
            "audio_offset_s": audio_offset,
            "ball_size_px": size,
            "size_ratio": size_ratio,
            "gap_prev_s": gap_prev,
            "gap_next_s": gap_next,
        }
    return signals


def _votes(signal, audio_available, config):
    votes = {}

    if audio_available:
        if signal["audio_score"] is not None:
            votes["audio"] = clip(
                (signal["audio_score"] - config["audio_mid_db"]) / config["audio_half_range_db"]
            )
        else:
            # Audio exists but no peak matched this event: wall hits are loud
            # enough that a missing peak is evidence, though not conclusive.
            votes["audio"] = -0.5

    if signal["size_ratio"] is not None:
        votes["size"] = clip(
            (config["size_mid_ratio"] - signal["size_ratio"]) / config["size_half_range"]
        )

    if signal["gap_prev_s"] is not None:
        votes["timing"] = clip(
            (config["timing_mid_s"] - signal["gap_prev_s"]) / config["timing_half_range_s"]
        )

    return votes


def classify_events(hits, results, audio_candidates, fps, config=None):
    """Returns copies of hits, each with event_type ('wall'|'racket'|'unknown'),
    wall_score (float|None), and signals. Never mutates the input hits."""
    config = merge_config(config)
    audio_available = audio_candidates is not None
    signals = compute_event_signals(hits, results, audio_candidates, fps, config)

    classified = []
    for hit, signal in zip(hits, signals):
        votes = _votes(signal, audio_available, config)
        total_weight = sum(config["weights"][name] for name in votes)

        if total_weight > 0:
            score = sum(config["weights"][name] * vote for name, vote in votes.items())
            score /= total_weight
            if score >= config["wall_threshold"]:
                event_type = EVENT_WALL
            elif score <= config["racket_threshold"]:
                event_type = EVENT_RACKET
            else:
                event_type = EVENT_UNKNOWN
        else:
            score = None
            event_type = EVENT_UNKNOWN

        entry = dict(hit)
        entry["event_type"] = event_type
        entry["wall_score"] = score
        entry["signals"] = signal
        classified.append(entry)

    return classified
