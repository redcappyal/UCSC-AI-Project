import pytest

import ball_detector


def _box(x, y, w=10.0, h=10.0, confidence=0.9):
    return {"x": x, "y": y, "width": w, "height": h,
            "confidence": confidence, "class": "ball", "class_name": "ball"}


def test_tile_windows_4k_grid_is_11_by_6():
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    xs = sorted({x for x, _ in windows})
    ys = sorted({y for _, y in windows})
    assert len(xs) == 11
    assert len(ys) == 6
    assert len(windows) == 66


def test_tile_windows_last_tile_is_clamped_in_bounds():
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    assert max(x for x, _ in windows) + 416 == 3840
    assert max(y for _, y in windows) + 416 == 2160


def test_tile_windows_covers_every_pixel_column():
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    covered = set()
    for x, _ in windows:
        covered.update(range(x, x + 416))
    assert covered == set(range(3840))


def test_tile_windows_exact_tile_size_frame_is_single_window():
    assert ball_detector.tile_windows(416, 416, 416, 64) == [(0, 0)]


def test_tile_windows_frame_smaller_than_tile_is_single_origin():
    assert ball_detector.tile_windows(300, 200, 416, 64) == [(0, 0)]


def test_tile_windows_rejects_overlap_at_or_above_tile():
    with pytest.raises(ValueError, match="overlap"):
        ball_detector.tile_windows(3840, 2160, 416, 416)


def test_tile_windows_stride_exceeds_max_ball_width():
    # Overlap must exceed the p90 ball width (24 px) so a ball is wholly
    # contained in at least one tile rather than clipped across a seam.
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    xs = sorted({x for x, _ in windows})
    assert 416 - (xs[1] - xs[0]) == 64


def test_iou_identical_boxes_is_one():
    assert ball_detector.iou(_box(100, 100), _box(100, 100)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert ball_detector.iou(_box(100, 100), _box(500, 500)) == 0.0


def test_merge_detections_drops_duplicate_across_tile_seam():
    kept = ball_detector.merge_detections(
        [_box(100, 100, confidence=0.7), _box(101, 100, confidence=0.9)], 0.65)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.9


def test_merge_detections_keeps_two_distinct_balls():
    kept = ball_detector.merge_detections([_box(100, 100), _box(900, 900)], 0.65)
    assert len(kept) == 2


def test_merge_detections_empty_input():
    assert ball_detector.merge_detections([], 0.65) == []


import numpy as np

import ball_model


def _manifest(tmp_path, **overrides):
    fields = {
        "schema_version": "ball-model-v1", "name": "t", "version": 1,
        "input_size": (416, 416), "decode": "in_graph", "conf_threshold": 0.25,
        "nms_iou": 0.65, "class_names": ("ball",), "tile_overlap_px": 64,
        "max_batch_tiles": 32, "artifact_sha256": "x", "source_checkpoint": "",
        "trained_commit": "", "val_ap50_95": 0.0, "notes": "",
        "model_dir": tmp_path,
    }
    fields.update(overrides)
    return ball_model.ModelManifest(**fields)


class _FakeRunner:
    """Emits one box at a fixed FULL-FRAME point, expressed tile-locally."""

    def __init__(self, windows, target_xy, score=0.9):
        self.windows = windows
        self.target_xy = target_xy
        self.score = score
        self.batch_sizes = []

    def run_batch(self, crops):
        self.batch_sizes.append(len(crops))
        out = []
        for crop in crops:
            index = len(out) + sum(self.batch_sizes[:-1])
            x0, y0 = self.windows[index]
            tx, ty = self.target_xy
            local_x, local_y = tx - x0, ty - y0
            h, w = crop.shape[:2]
            if 0 <= local_x < w and 0 <= local_y < h:
                out.append([(float(local_x), float(local_y), 10.0, 10.0,
                             self.score, 0)])
            else:
                out.append([])
        return out


def test_detect_frame_maps_tile_local_box_to_full_frame():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    manifest = _manifest(None)
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    runner = _FakeRunner(windows, target_xy=(1500, 900))

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(1500.0)
    assert detections[0]["y"] == pytest.approx(900.0)


def test_detect_frame_labels_class_ball_so_tracking_common_accepts_it():
    from tracking_common import is_ball_prediction

    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    manifest = _manifest(None)
    runner = _FakeRunner([(0, 0)], target_xy=(200, 200))

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert detections[0]["class"] == "ball"
    assert detections[0]["class_name"] == "ball"
    # candidate_ball_predictions silently falls back to ALL predictions when
    # nothing matches BALL_CLASS_NAMES, so a mislabel would not fail loudly.
    assert is_ball_prediction(detections[0])


def test_detect_frame_drops_boxes_below_manifest_conf_threshold():
    frame = np.zeros((416, 416, 3), dtype=np.uint8)
    manifest = _manifest(None, conf_threshold=0.5)
    runner = _FakeRunner([(0, 0)], target_xy=(200, 200), score=0.4)

    assert ball_detector.detect_frame(runner, frame, manifest) == []


def test_detect_frame_respects_max_batch_tiles():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    manifest = _manifest(None, max_batch_tiles=16)
    windows = ball_detector.tile_windows(3840, 2160, 416, 64)
    runner = _FakeRunner(windows, target_xy=(1500, 900))

    ball_detector.detect_frame(runner, frame, manifest)

    assert max(runner.batch_sizes) <= 16
    assert sum(runner.batch_sizes) == 66


def test_detect_frame_pads_frame_smaller_than_tile():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    manifest = _manifest(None)
    runner = _FakeRunner([(0, 0)], target_xy=(100, 100))

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(100.0)
