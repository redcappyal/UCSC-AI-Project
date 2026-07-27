"""Offline ball tracking of a single clip -> timestamped pixel samples.

Runs a ball detector -- by default the locally trained YOLOX detector
(ball_model + ball_detector); set BALL_DETECTOR=rfdetr to use the hosted
RF-DETR the Flask pipeline still runs on (via inference_engine) -- over one
squash clip and produces `TrackSample`s. Heavy deps (cv2 video I/O, the
`inference` package, torch, and tracking_common -- which imports cv2
unconditionally at its own top) import lazily inside functions -- mirroring
inference_engine.py / train_yolo_ball.py's convention -- so importing this
module, or exercising its test injection seams, never touches them.

This module was the single-camera half of the former `stereo_offline.py`. The
two-clip fusion half was archived on 2026-07-27 (see archive/stereo/README.md);
this half stayed, because it is the runner that exercises the local YOLOX
detector -- the thing the project is actually investing in.
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


BALL_DETECTOR_DEFAULT = "yolox"


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


def selected_detector():
    """Read + validate BALL_DETECTOR. The single source of truth for the
    backend name: every caller goes through this, so an unknown value raises
    everywhere instead of being silently reported as "yolox" on paths that
    never reach _build_infer (e.g. a clip that decodes zero frames)."""
    import os
    backend = os.environ.get("BALL_DETECTOR", BALL_DETECTOR_DEFAULT).strip().lower()
    if backend not in ("yolox", "rfdetr"):
        raise ValueError(
            f"Unknown BALL_DETECTOR {backend!r}; expected 'yolox' or 'rfdetr'")
    return backend


def _build_infer(model, confidence):
    """Build the stereo path's per-frame callable.

    Defaults to the locally trained YOLOX detector (BALL_DETECTOR=yolox);
    BALL_DETECTOR=rfdetr restores the hosted RF-DETR the single-camera
    pipeline still uses. Imported lazily so a clip with zero decoded frames
    never loads a model -- the CLI smoke test depends on that.

    Never falls back between detectors: a missing model raises, because a
    silent swap would make the stereo/single-camera split invisible.
    """
    backend = selected_detector()

    if backend == "rfdetr":
        get_tracking_model, infer_frame_predictions = _import_rfdetr()
        if model is None:
            model = get_tracking_model()

        def _infer(frame):
            return infer_frame_predictions(model, frame, confidence)

        return _infer

    # backend == "yolox" here: selected_detector() already validated the
    # value, so this is the only other possibility.
    if model is not None and not hasattr(model, "manifest"):
        raise TypeError(
            f"_build_infer got a `model` with no `.manifest` attribute while "
            f"BALL_DETECTOR={backend!r} selects the local YOLOX detector. "
            f"`model` means a ball_model runner (from ball_model.load_detector()) "
            f"here; it only means an RF-DETR tracking-model object when "
            f"BALL_DETECTOR=rfdetr. Pass model=None to load the default YOLOX "
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


def detections_to_track_samples(video_path, model=None, *, confidence=0.4,
                                stride=1, offset_s=0.0, infer=None):
    """Run the ball detector over a clip -> [stereo_engine.TrackSample].

    t_s = frame_index / fps + offset_s (fps from cv2 metadata; raises
    ValueError unless fps is finite and positive, and unless stride >= 1).
    Bad fps is fatal here on purpose: unlike app.py's `or 30.0` fallback for
    playback metadata, inventing a frame rate would silently shift every
    timestamp and hence every triangulated 3D position. `infer` is an
    injection seam for tests:
    callable(frame_bgr) -> list of prediction dicts in
    inference_engine.infer_frame_predictions' normalized shape
    ({"x","y","width","height","confidence","class"}); default None means
    build it via _build_infer(model, confidence), lazily on first frame so a
    clip that decodes zero frames never touches the heavy inference stack.
    _build_infer's backend is selected_detector() (BALL_DETECTOR, default
    "yolox"): by default it loads ball_model.load_detector() (when `model`
    is None) and calls ball_detector.detect_frame, in which case `model`,
    when given, must be a ball_model runner (something with a `.manifest`
    attribute). BALL_DETECTOR=rfdetr instead calls get_tracking_model()
    (when `model` is None) + inference_engine.infer_frame_predictions, in
    which case `model` means an RF-DETR tracking-model object. Passing a
    `model` of the wrong shape for the selected backend raises TypeError
    rather than failing later, mid-frame, with an opaque AttributeError.
    Selects the ball prediction per frame via tracking_common's
    select_ball_prediction; frames with no accepted ball produce no
    sample. px = (x, y) center in RAW pixels.
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
