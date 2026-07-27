"""ARCHIVED 2026-07-27 -- see archive/stereo/README.md. Not importable at runtime.

Offline two-clip stereo fusion adapter (the archived half of the former
`stereo_offline.py`).

Fuses two calibrated, clock-offset squash clips into 3D impact calls through
`stereo_engine`. The per-clip detector half stayed in the live tree as
`ball_track_offline.py` -- it is the runner that exercises the local YOLOX
detector, which is where the project's effort went instead.

RESTORING: this module imports `detections_to_track_samples` from
`ball_track_offline` (which still exists and still works); at the archive tag
both halves lived in one file. The env var `STEREO_DETECTOR` was renamed
`BALL_DETECTOR` when the halves split -- `selected_detector()` now lives in
`ball_track_offline`.
"""

import argparse
import json
from pathlib import Path

import court_model
import stereo_engine
from ball_track_offline import detections_to_track_samples, selected_detector


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
     "sample_counts": {"a": n, "b": m},
     "detector": {"backend": ..., "model": {...}}}
    """
    model_a, info_a = court_model.solve_camera_model(calibration_a)
    if model_a is None:
        raise ValueError(
            f"stereo_offline_fuse.fuse_clips: side 'a' calibration failed to "
            f"solve (status={info_a.get('status')!r})")
    model_b, info_b = court_model.solve_camera_model(calibration_b)
    if model_b is None:
        raise ValueError(
            f"stereo_offline_fuse.fuse_clips: side 'b' calibration failed to "
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

    if infer_a is not None and infer_b is not None:
        detector_info = {"backend": "injected"}
    else:
        backend = selected_detector()
        if backend == "rfdetr":
            detector_info = {"backend": "rfdetr"}
        else:
            import ball_model
            detector_info = {"backend": "yolox", "model": ball_model.describe()}

    return {
        "impacts": [_impact_to_dict(impact) for impact in impacts],
        "pair_agreement": agreement,
        "sample_counts": {"a": len(samples_a), "b": len(samples_b)},
        "detector": detector_info,
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
