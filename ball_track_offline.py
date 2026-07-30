"""Offline ball tracking of a single clip -> timestamped pixel samples.

Runs a ball detector -- by default the hosted RF-DETR the Flask pipeline runs
on (BALL_DETECTOR=rfdetr, via inference_engine) -- over one squash clip and
produces `TrackSample`s. Set BALL_DETECTOR=local (alias "yolox") to use the
locally trained detector (ball_model + ball_detector) instead. A local
manifest with frames_per_input > 1 (a temporal/WASB-style model) routes
through centred 3-frame windows (_centered_windows) instead of one-frame-at-a-
time detection -- see detections_to_track_samples. Heavy deps (cv2 video I/O,
the `inference` package, torch, and tracking_common -- which imports cv2
unconditionally at its own top) import lazily inside functions -- mirroring
inference_engine.py / train_yolo_ball.py's convention -- so importing this
module, or exercising its test injection seams, never touches them.

This module was the single-camera half of the former `stereo_offline.py`. The
two-clip fusion half was archived on 2026-07-27 (see archive/stereo/README.md);
this half stayed, because it is the runner that exercises the local ball
detector -- the thing the project is actually investing in. It still is; it
just no longer reaches for it *by default*, which is a statement about which
detector is trusted today, not about which one is being built.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackSample:
    """One accepted ball detection: a clip-relative time and a raw pixel centre.

    Previously imported from `stereo_engine`; declared here so the kept
    detector path owns its own vocabulary and does not reach into the archive.
    """
    t_s: float
    px: tuple


def _video_fps(video_path):
    """FPS from cv2 video metadata. Lazy cv2 import; tests monkeypatch this
    function entirely, so real video I/O is never exercised at test time."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        return cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()


def _iter_frames(video_path):
    """Yield (frame_index, frame_bgr) for every decoded frame, in order,
    starting at index 0. Lazy cv2 import (a generator body only starts
    running on first `next()`, so even calling this without iterating stays
    cv2-free); tests monkeypatch this function entirely to bypass real
    video I/O."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame_index, frame
            frame_index += 1
    finally:
        cap.release()


# The one place that decides which detector runs when nobody says otherwise.
# "rfdetr" as of 2026-07-30: the committed WASB artifact is wired but still
# unmeasured against eval_set/BASELINE-2026-07-23.md, so the hosted RF-DETR --
# the detector every existing number in this repo came from -- stays the
# default until that eval says otherwise. WASB is one env var away
# (BALL_DETECTOR=local) and nothing about it was removed.
BALL_DETECTOR_DEFAULT = "rfdetr"


def _import_rfdetr():
    """Seam for tests; keeps the heavy import lazy."""
    from inference_engine import get_tracking_model, infer_frame_predictions
    return get_tracking_model, infer_frame_predictions


def _load_ball_detector():
    """Seam for tests; keeps torch out of import time."""
    import ball_model
    return ball_model.load_detector()


def _detect_frame(runner, frame, manifest):
    """Seam for tests."""
    import ball_detector
    return ball_detector.detect_frame(runner, frame, manifest)


def _detect_frame_stack(runner, frames, manifest):
    """Seam for tests."""
    import ball_detector
    return ball_detector.detect_frame_stack(runner, frames, manifest)


def selected_detector():
    """Read + validate BALL_DETECTOR. The single source of truth for the
    backend name: every caller goes through this, so an unknown value raises
    everywhere instead of being silently reported as "local" on paths that
    never reach _build_infer (e.g. a clip that decodes zero frames).

    "local" is canonical; "yolox" is accepted as a compatible alias (it
    predates the local backend growing a temporal mode, and MODEL.md still
    documents it) and normalises to "local".

    Unset resolves to BALL_DETECTOR_DEFAULT ("rfdetr"). This is the only read
    of that constant, so the Flask app, this offline runner, and the eval
    scripts cannot drift onto different defaults."""
    import os
    backend = os.environ.get("BALL_DETECTOR", BALL_DETECTOR_DEFAULT).strip().lower()
    if backend == "yolox":
        backend = "local"
    if backend not in ("local", "rfdetr"):
        raise ValueError(
            f"Unknown BALL_DETECTOR {backend!r}; expected 'local' (or its "
            f"alias 'yolox') or 'rfdetr'")
    return backend


def _build_infer(model, confidence):
    """Build the per-frame detection callable.

    Defaults to the hosted RF-DETR the Flask pipeline runs
    (BALL_DETECTOR=rfdetr); BALL_DETECTOR=local (alias "yolox") selects the
    locally trained detector. Imported lazily so a clip with zero decoded
    frames never loads a model at all.

    Never falls back between detectors: a missing model raises, because a
    silent swap would make the local/rfdetr split invisible.
    """
    backend = selected_detector()

    if backend == "rfdetr":
        get_tracking_model, infer_frame_predictions = _import_rfdetr()
        if model is None:
            model = get_tracking_model()

        def _infer(frame):
            return infer_frame_predictions(model, frame, confidence)

        return _infer

    # backend == "local" here: selected_detector() already validated the
    # value (local or its yolox alias), so this is the only other
    # possibility.
    if model is not None and not hasattr(model, "manifest"):
        raise TypeError(
            f"_build_infer got a `model` with no `.manifest` attribute while "
            f"BALL_DETECTOR={backend!r} selects the local ball detector. "
            f"`model` means a ball_model runner (from ball_model.load_detector()) "
            f"here; it only means an RF-DETR tracking-model object when "
            f"BALL_DETECTOR=rfdetr. Pass model=None to load the default local "
            f"runner, or set BALL_DETECTOR=rfdetr if you meant to pass an "
            f"RF-DETR model.")

    runner = model if model is not None else _load_ball_detector()

    # `confidence` is intentionally unused below: the manifest's
    # conf_threshold is the detector's own floor, baked in at export time,
    # and owns filtering for the yolox branch. We only check that it does
    # not silently swallow the caller's requested threshold (see the guard
    # below).
    if runner.manifest.conf_threshold > confidence:
        raise ValueError(
            f"manifest conf_threshold ({runner.manifest.conf_threshold!r}) is "
            f"above the requested confidence ({confidence!r}); the detector "
            f"already drops everything below its own conf_threshold before "
            f"this call ever sees it, so the requested confidence would be "
            f"silently unreachable.")

    def _infer(frame):
        return _detect_frame(runner, frame, runner.manifest)

    return _infer


def _centered_windows(indexed_frames):
    """(frame_index, [prev, cur, nxt]) for every frame of an (index, frame)
    iterator -- the centred 3-frame window the temporal detector consumes.

    Emission lags decode by one frame: frame t's window needs t+1, which is
    fine offline. Clip edges pad by repeating the first/last frame, mirroring
    the dataset builder's edge padding so serving matches training. Supports
    exactly 3-frame windows; detections_to_track_samples rejects other
    frames_per_input values loudly.
    """
    previous = None       # frame t-1 (None at the left edge)
    current = None        # (index, frame t) awaiting its right neighbour
    for index, frame in indexed_frames:
        if current is not None:
            cur_index, cur_frame = current
            yield cur_index, [previous if previous is not None else cur_frame,
                              cur_frame, frame]
            previous = cur_frame
        current = (index, frame)
    if current is not None:
        cur_index, cur_frame = current
        yield cur_index, [previous if previous is not None else cur_frame,
                          cur_frame, cur_frame]


def detections_to_track_samples(video_path, model=None, *, confidence=0.4,
                                stride=1, offset_s=0.0, infer=None):
    """Run the ball detector over a clip -> [TrackSample].

    t_s = frame_index / fps + offset_s (fps from cv2 metadata; raises
    ValueError unless fps is finite and positive, and unless stride >= 1).
    Bad fps is fatal here on purpose: unlike app.py's `or 30.0` fallback for
    playback metadata, inventing a frame rate would silently shift every
    timestamp, and every downstream position with it. `infer` is an
    injection seam for tests:
    callable(frame_bgr) -> list of prediction dicts in
    inference_engine.infer_frame_predictions' normalized shape
    ({"x","y","width","height","confidence","class"}); default None means
    build it via _build_infer(model, confidence), lazily on first frame so a
    clip that decodes zero frames never touches the heavy inference stack.
    _build_infer's backend is selected_detector() (BALL_DETECTOR, default
    "local"/"yolox"): by default it loads ball_model.load_detector() (when
    `model` is None) and calls ball_detector.detect_frame, in which case
    `model`, when given, must be a ball_model runner (something with a
    `.manifest` attribute). BALL_DETECTOR=rfdetr instead calls
    get_tracking_model() (when `model` is None) +
    inference_engine.infer_frame_predictions, in which case `model` means an
    RF-DETR tracking-model object. Passing a `model` of the wrong shape for
    the selected backend raises TypeError rather than failing later,
    mid-frame, with an opaque AttributeError. Selects the ball prediction per
    frame via tracking_common's select_ball_prediction; frames with no
    accepted ball produce no sample. px = (x, y) center in RAW pixels.

    Temporal manifests: when `infer` is not supplied and the local backend's
    runner has runner.manifest.frames_per_input > 1 (a temporal/WASB-style
    model), this function skips the per-frame loop above and routes through
    _centered_windows instead. Frame t's detection uses [t-1, t, t+1] -- one
    frame of lookahead, double-sided context -- which is free here because
    this runs offline over an already-recorded clip, not live. Timestamps
    stay per-frame-correct (t's sample is always t / fps + offset_s) even
    though decoding t+1 happens before t's window is emitted. Exactly
    frames_per_input == 3 is supported today; any other value raises loudly
    rather than guessing how to widen the window. stride must be 1 for a
    temporal manifest -- the model was trained on consecutive frames, so
    skipping frames would feed it a context it never saw in training. The
    same "manifest conf_threshold must not exceed the requested confidence"
    guard _build_infer applies to the single-frame local path also applies
    here. This routing only ever applies to the local backend (rfdetr has no
    manifest / frames_per_input concept) and only when `infer` is not
    supplied -- an explicitly-passed `infer` always keeps the per-frame loop,
    regardless of manifest. Because frames_per_input lives on the manifest,
    and the manifest lives on the runner, deciding which loop to take
    requires resolving the runner before iterating any frames: unlike the
    rest of this function, a local-backend clip with zero decoded frames
    still loads the model when `model=None` (rfdetr and an explicitly-passed
    `infer` remain fully lazy).
    """
    from tracking_common import select_ball_prediction  # lazy: cv2 at its top

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride!r}")

    fps = _video_fps(video_path)
    # math.isfinite first: NaN is truthy and every NaN comparison is False,
    # so `not fps or fps <= 0` alone lets NaN through and every t_s becomes
    # NaN, surfacing much later as an opaque numpy error.
    if fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid fps ({fps!r}) reading {video_path}")

    samples = []

    # A caller-supplied `infer` always takes the per-frame loop below,
    # regardless of manifest -- it bypasses _build_infer entirely, so there
    # is no manifest to route on.
    if infer is None and selected_detector() == "local":
        runner = model if model is not None else _load_ball_detector()
        if not hasattr(runner, "manifest"):
            raise TypeError(
                f"detections_to_track_samples got a `model` with no `.manifest` "
                f"attribute while BALL_DETECTOR selects the local ball detector. "
                f"`model` means a ball_model runner (from "
                f"ball_model.load_detector()) here; pass model=None to load the "
                f"default local runner, or set BALL_DETECTOR=rfdetr if you meant "
                f"to pass an RF-DETR model.")
        frames_needed = int(getattr(runner.manifest, "frames_per_input", 1))

        if frames_needed > 1:
            # temporal manifests: centred windows, consecutive frames only
            if frames_needed != 3:
                raise ValueError(
                    f"frames_per_input={frames_needed} not supported; the centred "
                    f"window path implements exactly 3 (WASB). Widen _centered_windows "
                    f"deliberately if a future model needs 5.")
            if stride != 1:
                raise ValueError(
                    f"stride={stride} is invalid with a temporal manifest: the model "
                    f"consumes consecutive frames; non-adjacent frames were never in "
                    f"its training distribution.")
            if runner.manifest.conf_threshold > confidence:
                raise ValueError(
                    f"manifest conf_threshold ({runner.manifest.conf_threshold!r}) is "
                    f"above the requested confidence ({confidence!r}); the detector "
                    f"already drops everything below its own conf_threshold before "
                    f"this call ever sees it, so the requested confidence would be "
                    f"silently unreachable.")
            for frame_index, window in _centered_windows(_iter_frames(video_path)):
                predictions = _detect_frame_stack(runner, window, runner.manifest)
                ball = select_ball_prediction(predictions, confidence)
                if ball is None:
                    continue
                t_s = frame_index / fps + offset_s
                samples.append(TrackSample(t_s=t_s, px=(float(ball["x"]), float(ball["y"]))))
            return samples

    for frame_index, frame in _iter_frames(video_path):
        if frame_index % stride != 0:
            continue
        if infer is None:
            infer = _build_infer(model, confidence)
        predictions = infer(frame)
        ball = select_ball_prediction(predictions, confidence)
        if ball is None:
            continue
        t_s = frame_index / fps + offset_s
        samples.append(TrackSample(t_s=t_s, px=(float(ball["x"]), float(ball["y"]))))
    return samples
