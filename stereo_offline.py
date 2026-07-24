"""Offline two-clip stereo fusion adapter.

Runs the production RF-DETR ball detector (via inference_engine) over two
calibrated, clock-offset squash clips and produces 3D impact calls through
stereo_engine. Heavy deps (cv2 video I/O, the `inference` package, and
tracking_common -- which imports cv2 unconditionally at its own top) import
lazily inside functions -- mirroring inference_engine.py / train_yolo_ball.py's
convention -- so importing this module, or exercising its test injection
seams, never touches them.
"""

import argparse
import json
from pathlib import Path

import court_model
import stereo_engine
from stereo_engine import TrackSample


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


def _build_infer(model, confidence):
    """Build the real detector's per-frame callable: get_tracking_model()
    (when `model` is None) + inference_engine.infer_frame_predictions.
    Imported lazily so this is only ever touched when a clip actually has a
    frame to run inference on -- a clip with zero decoded frames (as in the
    CLI smoke test) never loads the model."""
    from inference_engine import get_tracking_model, infer_frame_predictions

    if model is None:
        model = get_tracking_model()

    def _infer(frame):
        return infer_frame_predictions(model, frame, confidence)

    return _infer


def detections_to_track_samples(video_path, model=None, *, confidence=0.4,
                                stride=1, offset_s=0.0, infer=None):
    """Run the ball detector over a clip -> [stereo_engine.TrackSample].

    t_s = frame_index / fps + offset_s (fps from cv2 metadata; raises
    ValueError if fps <= 0). `infer` is an injection seam for tests:
    callable(frame_bgr) -> list of prediction dicts in
    inference_engine.infer_frame_predictions' normalized shape
    ({"x","y","width","height","confidence","class"}); default None means
    build it from the real model (get_tracking_model() when model is None)
    + infer_frame_predictions, built lazily on first frame so a clip that
    decodes zero frames never touches the heavy inference stack. Selects
    the ball prediction per frame via tracking_common's
    select_ball_prediction; frames with no accepted ball produce no
    sample. px = (x, y) center in RAW pixels.
    """
    from tracking_common import select_ball_prediction  # lazy: cv2 at its top

    fps = _video_fps(video_path)
    if not fps or fps <= 0:
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


def _impact_to_dict(impact):
    return {
        "t_s": impact.t_s,
        "surface": impact.surface,
        "point_ft": [float(value) for value in impact.point_ft],
        "call": impact.call,
        "margin_ft": impact.margin_ft,
        "confidence": impact.confidence,
        "snap_disagreement_ft": impact.snap_disagreement_ft,
    }


def fuse_clips(video_a, calibration_a, video_b, calibration_b, *,
               offset_s_b=0.0, confidence=0.4, stride=1, infer_a=None, infer_b=None):
    """Fuse two calibrated, clock-offset clips into 3D impact calls.

    Solves both calibrations (court_model.solve_camera_model), raising
    ValueError naming the failing side and its solve status if either does
    not solve. Builds ball-track samples for each clip via
    detections_to_track_samples (offset_s_b applied to clip B only, so both
    tracks share clip A's clock), runs stereo_engine.detect_impacts, and
    computes pair_agreement from the two calibrations' own observed
    correspondences -- the same court_model._camera_correspondences ->
    (court_ft, px) list construction /api/camera-pair-check uses in app.py,
    so this exercises the identical non-tautological cross-check.

    Returns a JSON-safe dict:
    {"impacts": [{"t_s","surface","point_ft","call","margin_ft",
                   "confidence","snap_disagreement_ft"}],
     "pair_agreement": {...},
     "sample_counts": {"a": n, "b": m}}
    """
    model_a, info_a = court_model.solve_camera_model(calibration_a)
    if model_a is None:
        raise ValueError(
            f"stereo_offline.fuse_clips: side 'a' calibration failed to "
            f"solve (status={info_a.get('status')!r})")
    model_b, info_b = court_model.solve_camera_model(calibration_b)
    if model_b is None:
        raise ValueError(
            f"stereo_offline.fuse_clips: side 'b' calibration failed to "
            f"solve (status={info_b.get('status')!r})")

    samples_a = detections_to_track_samples(
        video_a, confidence=confidence, stride=stride, infer=infer_a)
    samples_b = detections_to_track_samples(
        video_b, confidence=confidence, stride=stride,
        offset_s=offset_s_b, infer=infer_b)

    impacts = stereo_engine.detect_impacts(model_a, samples_a, model_b, samples_b)

    image_px_a, court_xyz_a, _labels_a = court_model._camera_correspondences(calibration_a)
    image_px_b, court_xyz_b, _labels_b = court_model._camera_correspondences(calibration_b)
    obs_a = list(zip(court_xyz_a, image_px_a))
    obs_b = list(zip(court_xyz_b, image_px_b))
    agreement = stereo_engine.pair_agreement(model_a, obs_a, model_b, obs_b)

    return {
        "impacts": [_impact_to_dict(impact) for impact in impacts],
        "pair_agreement": agreement,
        "sample_counts": {"a": len(samples_a), "b": len(samples_b)},
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-a", required=True)
    parser.add_argument("--calib-a", required=True)
    parser.add_argument("--video-b", required=True)
    parser.add_argument("--calib-b", required=True)
    parser.add_argument("--offset-s-b", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    calibration_a = json.loads(Path(args.calib_a).read_text(encoding="utf-8"))
    calibration_b = json.loads(Path(args.calib_b).read_text(encoding="utf-8"))

    result = fuse_clips(
        args.video_a, calibration_a, args.video_b, calibration_b,
        offset_s_b=args.offset_s_b, confidence=args.confidence, stride=args.stride,
    )

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
