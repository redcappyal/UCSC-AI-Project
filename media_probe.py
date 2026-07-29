"""What a clip can support, measured before anything tries to analyse it.

The pipeline was built around one capture format -- our own locked 4K60 -- and
its thresholds still assume it. Analysis has to run on whatever a player
actually filmed, which means the first question about a clip is no longer "what
did we find in it" but "what could we have found". This module answers the
second: frame rate, size, how sharp the frames are, and whether there is an
audio channel at all.

Nothing here decides anything. `capabilities.py` turns these numbers into
enabled/disabled tiers with stated reasons; keeping the measurement separate is
what lets a gate be retuned without re-deriving what was measured.
"""

import math

import cv2
import numpy as np

# A 40-minute match is ~144k frames. The probe runs at ingest, in the request
# path, so it seeks to a fixed handful of frames rather than decoding through.
# 16 is enough to survive a few undecodable samples and still take a median.
SHARPNESS_SAMPLES = 16

# Containers regularly report 0 fps. app.video_info already falls back to 30 for
# exactly this reason; the same number is used here so a clip cannot probe as
# one frame rate and track as another.
FALLBACK_FPS = 30.0


def _sample_indices(frame_count, samples=SHARPNESS_SAMPLES):
    """Evenly spaced frame indices, avoiding the first and last frames.

    The endpoints are skipped deliberately: the first frame of a phone
    recording is often the auto-exposure ramp and the last is often a partial
    write, and both read as blur that the rest of the clip does not have.
    """
    if frame_count <= 0:
        return []
    if frame_count <= samples:
        return list(range(frame_count))
    step = frame_count / (samples + 1.0)
    return [int(round(step * (index + 1))) for index in range(samples)]


def _centre_crop(frame):
    """The middle half of the frame, where the court is.

    Squash footage from a back-wall mount puts the glass, the ceiling and the
    gallery around the edges. Letterboxing and vignetting both read as flat,
    which drags a whole-frame sharpness measure toward "blurred" on footage
    that is fine where it matters.
    """
    height, width = frame.shape[:2]
    top, left = height // 4, width // 4
    return frame[top:top + max(1, height // 2), left:left + max(1, width // 2)]


def _sharpness_of(frame):
    """Variance of the Laplacian -- the standard focus measure.

    A sharp edge has a large second derivative; blur spreads it out and the
    variance collapses. It is scale-dependent, so it compares clips of the same
    resolution and is only ever used against a threshold, never across sizes.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(_centre_crop(grey), cv2.CV_64F).var())


def _measure_sharpness(capture, frame_count):
    """Median sharpness over the sampled frames, or None if none decoded.

    Median, not mean: one frame caught mid-swing is genuinely blurred without
    saying anything about the clip, and a mean lets that single frame drag the
    whole measure under a gate.

    None means "could not measure", which is not the same answer as a low
    number and must not be collapsed into one -- a capability card that says
    "too blurred" has to have actually looked.
    """
    values = []
    for index in _sample_indices(frame_count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        values.append(_sharpness_of(frame))

    if not values:
        return None
    return float(np.median(values))


def _has_audio(video_path):
    """Whether the container carries an audio stream.

    False also covers "PyAV could not open it", which is deliberate: audio is
    never a capability gate (rally structure stays enabled either way), and
    whether audio actually yielded usable transients is reported at extraction
    time instead. Widening this to a tri-state would push a distinction into
    every consumer that none of them act on.
    """
    try:
        import av
    except ImportError:
        return False

    try:
        with av.open(str(video_path)) as container:
            return any(stream.type == "audio" for stream in container.streams)
    except Exception:
        return False


def probe_video(video_path):
    """Measure what `video_path` offers an analysis run.

    Returns fps, width, height, frame_count, duration_s, sharpness (None if no
    frame decoded) and has_audio.

    Raises ValueError when the file cannot be opened at all. That is louder
    than returning zeroes on purpose: zeroes are indistinguishable from a real
    unqualified clip, and every tier would then be disabled for a reason nobody
    measured.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or not math.isfinite(fps) or fps <= 0:
            fps = FALLBACK_FPS
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        sharpness = _measure_sharpness(capture, frame_count)
    finally:
        capture.release()

    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_s": frame_count / fps if fps else 0.0,
        "sharpness": sharpness,
        "has_audio": _has_audio(video_path),
    }
