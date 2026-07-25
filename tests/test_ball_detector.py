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
