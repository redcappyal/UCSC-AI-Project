import numpy as np
import pytest

import ball_detector
import ball_model


@pytest.fixture(autouse=True)
def _no_ambient_batch_cap(monkeypatch):
    """A BALL_MAX_BATCH_TILES in the developer's shell must not silently
    re-tune the batch that every assertion in this file makes."""
    monkeypatch.delenv("BALL_MAX_BATCH_TILES", raising=False)


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


_MARKER = 255


def _marker_frame(width, height, x, y):
    """Black frame with a single lit pixel at FULL-FRAME (x, y)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[y, x] = _MARKER
    return frame


def _tiles_seeing(x, y, frame_w=3840, frame_h=2160, tile=416, overlap=64):
    """Indices into tile_windows() of every tile whose crop contains (x, y)."""
    windows = ball_detector.tile_windows(frame_w, frame_h, tile, overlap)
    return [i for i, (x0, y0) in enumerate(windows)
            if x0 <= x < x0 + tile and y0 <= y < y0 + tile]


class _MarkerRunner:
    """Reports, tile-locally, the marker pixel it FINDS in each crop.

    It is never told where the target is -- it reads the image it is handed. A
    crop taken with the axes swapped, or padded anywhere but the top-left,
    hands this runner a different picture, so the full-frame coordinate the
    tests assert moves. A runner keyed on crop.shape alone cannot tell.
    """

    def __init__(self, score=0.9, class_index=0):
        self.score = score
        self.class_index = class_index
        self.batch_sizes = []
        self.boxes_emitted = 0

    def run_batch(self, crops):
        self.batch_sizes.append(len(crops))
        out = []
        for crop in crops:
            boxes = [(float(col), float(row), 10.0, 10.0,
                      self.score, self.class_index)
                     for row, col in np.argwhere(crop[:, :, 0] == _MARKER)]
            self.boxes_emitted += len(boxes)
            out.append(boxes)
        return out


class _ShortRunner:
    """Returns one result fewer than it was handed.

    The realistic shape of this bug is a runner that omits tiles with no
    detections; the consequence is identical either way.
    """

    def run_batch(self, crops):
        return [[] for _ in crops[:-1]]


def test_detect_frame_maps_tile_local_box_to_full_frame():
    frame = _marker_frame(3840, 2160, 1500, 900)
    manifest = _manifest(None)
    runner = _MarkerRunner()

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(1500.0)
    assert detections[0]["y"] == pytest.approx(900.0)
    # One tile sees this point, so the merge had nothing to hide behind.
    assert runner.boxes_emitted == 1


def test_detect_frame_maps_a_target_only_the_clamped_edge_tiles_can_see():
    # The last row and column of tiles are clamped flush to the frame edge, so
    # their origins are NOT stride multiples: 3424 and 1744, not 3520/2112.
    assert _tiles_seeing(3820, 2140) == [65]
    frame = _marker_frame(3840, 2160, 3820, 2140)
    manifest = _manifest(None)
    runner = _MarkerRunner()

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(3820.0)
    assert detections[0]["y"] == pytest.approx(2140.0)


def test_detect_frame_maps_a_target_whose_tile_lands_in_a_later_batch():
    # Tile 37 of 66: with max_batch_tiles=16 it is in the third batch, so the
    # origin it is paired with comes from a chunk, not from the flat window
    # list. An off-by-a-chunk error only shows up here.
    seen = _tiles_seeing(1500, 1200)
    assert seen == [37] and min(seen) >= 16
    frame = _marker_frame(3840, 2160, 1500, 1200)
    manifest = _manifest(None, max_batch_tiles=16)
    runner = _MarkerRunner()

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert len(runner.batch_sizes) > 1
    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(1500.0)
    assert detections[0]["y"] == pytest.approx(1200.0)


def test_detect_frame_collapses_one_ball_seen_by_four_overlapping_tiles():
    # (352, 352) is in both an x and a y overlap band.
    assert len(_tiles_seeing(352, 352)) == 4
    frame = _marker_frame(3840, 2160, 352, 352)
    manifest = _manifest(None)
    runner = _MarkerRunner()

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert runner.boxes_emitted == 4
    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(352.0)
    assert detections[0]["y"] == pytest.approx(352.0)


def test_detect_frame_rejects_a_runner_returning_fewer_results_than_crops():
    # Without strict zip this truncates in silence, and every box after the
    # gap is added to the wrong tile origin.
    frame = _marker_frame(3840, 2160, 1500, 900)
    manifest = _manifest(None)

    with pytest.raises(ValueError, match="shorter"):
        ball_detector.detect_frame(_ShortRunner(), frame, manifest)


def test_detect_frame_labels_class_ball_so_tracking_common_accepts_it():
    from tracking_common import is_ball_prediction

    frame = _marker_frame(416, 416, 200, 200)
    manifest = _manifest(None)
    runner = _MarkerRunner()

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert detections[0]["class"] == "ball"
    assert detections[0]["class_name"] == "ball"
    # candidate_ball_predictions silently falls back to ALL predictions when
    # nothing matches BALL_CLASS_NAMES, so a mislabel would not fail loudly.
    assert is_ball_prediction(detections[0])


def test_detect_frame_rejects_negative_class_index_instead_of_wrapping():
    frame = _marker_frame(416, 416, 200, 200)
    manifest = _manifest(None, class_names=("ball", "player"))
    runner = _MarkerRunner(class_index=-1)

    # -1 indexes the LAST class name in Python; a mislabelled "player" would
    # be indistinguishable from a real one downstream.
    with pytest.raises(ValueError, match="-1"):
        ball_detector.detect_frame(runner, frame, manifest)


def test_detect_frame_rejects_class_index_past_the_manifest_names():
    frame = _marker_frame(416, 416, 200, 200)
    manifest = _manifest(None)
    runner = _MarkerRunner(class_index=7)

    with pytest.raises(ValueError, match="7"):
        ball_detector.detect_frame(runner, frame, manifest)


def test_detect_frame_rejects_non_square_input_size():
    frame = _marker_frame(416, 416, 200, 200)
    manifest = _manifest(None, input_size=(416, 320))

    with pytest.raises(ValueError, match=r"416, 320"):
        ball_detector.detect_frame(_MarkerRunner(), frame, manifest)


def test_detect_frame_drops_boxes_below_manifest_conf_threshold():
    frame = _marker_frame(416, 416, 200, 200)
    manifest = _manifest(None, conf_threshold=0.5)
    runner = _MarkerRunner(score=0.4)

    assert ball_detector.detect_frame(runner, frame, manifest) == []


def test_detect_frame_respects_max_batch_tiles():
    frame = _marker_frame(3840, 2160, 1500, 900)
    manifest = _manifest(None, max_batch_tiles=16)
    runner = _MarkerRunner()

    ball_detector.detect_frame(runner, frame, manifest)

    assert max(runner.batch_sizes) <= 16
    assert sum(runner.batch_sizes) == 66


def test_detect_frame_pads_frame_smaller_than_tile():
    # Top-left padding, so full-frame coordinates need no compensation.
    frame = _marker_frame(300, 200, 100, 100)
    manifest = _manifest(None)
    runner = _MarkerRunner()

    detections = ball_detector.detect_frame(runner, frame, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(100.0)
    assert detections[0]["y"] == pytest.approx(100.0)


# -- detect_frame_stack ------------------------------------------------------
#
# Same tile-local -> full-frame mapping and cross-tile merge as detect_frame,
# but the runner is handed co-located crops from every frame in the stack,
# concatenated oldest-first along the channel axis. Detections locate the
# ball in the MIDDLE frame -- the runner decodes the centre MIMO channel.


def _stack_manifest(tmp_path=None, **overrides):
    values = dict(frames_per_input=3, decode="heatmap_peak",
                  heatmap_stride=2, nominal_ball_px=12.0)
    values.update(overrides)
    return _manifest(tmp_path, **values)


class _StackShortRunner:
    """Returns one result fewer than it was handed -- the stack-path analogue
    of _ShortRunner."""

    def run_batch(self, stacks):
        return [[] for _ in stacks[:-1]]


def test_detect_frame_stack_concatenates_frames_oldest_first():
    manifest = _stack_manifest()
    frames = [np.full((416, 416, 3), fill, dtype=np.uint8) for fill in (10, 20, 30)]

    class Probe:
        def run_batch(self, stacks):
            (stack,) = stacks
            assert stack.shape[2] == 9, stack.shape
            assert stack[0, 0, 0] == 10 and stack[0, 0, 3] == 20 and stack[0, 0, 6] == 30
            return [[(100.0, 50.0, 12.0, 12.0, 0.9, 0)]]

    detections = ball_detector.detect_frame_stack(Probe(), frames, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(100.0)
    assert detections[0]["y"] == pytest.approx(50.0)
    assert detections[0]["class"] == "ball"


def test_detect_frame_stack_rejects_wrong_frame_count():
    manifest = _stack_manifest()
    frames = [np.zeros((416, 416, 3), dtype=np.uint8)] * 2

    with pytest.raises(ValueError, match="frames_per_input"):
        ball_detector.detect_frame_stack(_StackShortRunner(), frames, manifest)


def test_detect_frame_stack_rejects_mismatched_frame_shapes():
    manifest = _stack_manifest()
    frames = [np.zeros((416, 416, 3), dtype=np.uint8),
              np.zeros((416, 416, 3), dtype=np.uint8),
              np.zeros((832, 416, 3), dtype=np.uint8)]

    with pytest.raises(ValueError, match="shape"):
        ball_detector.detect_frame_stack(_StackShortRunner(), frames, manifest)


def test_detect_frame_stack_maps_tile_local_peak_to_full_frame(monkeypatch):
    # 4K stack, peak in a non-first tile -- port of the v1 mapping test. Batch
    # covers all 66 tiles so the fake stays simple (index 0 == first window),
    # which the default cap of 8 would otherwise break: every chunk would have
    # an index 0 and OneHit would fire nine times.
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "66")
    manifest = _stack_manifest(max_batch_tiles=66)
    frames = [np.zeros((2160, 3840, 3), dtype=np.uint8)] * 3

    class OneHit:
        def run_batch(self, stacks):
            out = [[] for _ in stacks]
            out[0] = [(10.0, 20.0, 12.0, 12.0, 0.9, 0)]
            return out

    detections = ball_detector.detect_frame_stack(OneHit(), frames, manifest)

    assert len(detections) == 1
    assert detections[0]["x"] == pytest.approx(10.0)
    assert detections[0]["y"] == pytest.approx(20.0)


def test_detect_frame_stack_rejects_a_runner_returning_fewer_results_than_crops():
    # Without strict zip this truncates in silence, and every box after the
    # gap is added to the wrong tile origin.
    manifest = _stack_manifest()
    frames = [np.zeros((2160, 3840, 3), dtype=np.uint8)] * 3

    with pytest.raises(ValueError, match="shorter"):
        ball_detector.detect_frame_stack(_StackShortRunner(), frames, manifest)


def test_detect_frame_stack_drops_boxes_below_manifest_conf_threshold():
    manifest = _stack_manifest(conf_threshold=0.5)
    frames = [np.zeros((416, 416, 3), dtype=np.uint8)] * 3

    class LowScore:
        def run_batch(self, stacks):
            return [[(200.0, 200.0, 12.0, 12.0, 0.4, 0)] for _ in stacks]

    assert ball_detector.detect_frame_stack(LowScore(), frames, manifest) == []


# -- effective_batch_tiles ---------------------------------------------------
#
# manifest.max_batch_tiles is the exported model's ceiling, not a mandate, and
# nothing measured benefits from a batch as big as its 32. bench_wasb_device.py
# has CUDA FLAT across the whole range (Lambda A10: 16.9 / 17.3 / 16.8 / 18.1 /
# 18.2 ms/tile at batch 1 / 8 / 16 / 32 / 64 -- the model saturates the card on
# one tile), while MPS falls off a ~40x cliff between 8 and 10 (8 GB M2 Air:
# 81.8 ms/tile at batch 8, 3082 at batch 10) and can fail outright with a Metal
# OOM at 4K tile counts. A batch above 8 therefore helps nobody and ruins one
# device, so the cap is one default rather than a per-device branch.
#
# Batching is a throughput knob only: nothing here may change WHICH tiles are
# swept or what comes back.


def test_effective_batch_tiles_defaults_to_the_measured_cap():
    assert ball_detector.effective_batch_tiles(_manifest(None)) == 8
    assert ball_detector.DEFAULT_MAX_BATCH_TILES == 8


def test_effective_batch_tiles_never_raises_a_smaller_manifest():
    # A future model that declares a smaller batch keeps it: the manifest is
    # the ceiling, and the cap only ever lowers.
    assert ball_detector.effective_batch_tiles(_manifest(None, max_batch_tiles=4)) == 4


def test_effective_batch_tiles_env_lowers_the_batch(monkeypatch):
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "4")
    assert ball_detector.effective_batch_tiles(_manifest(None)) == 4


def test_effective_batch_tiles_env_raises_above_the_default(monkeypatch):
    # The default is caution, not a lock: an operator who has measured their
    # own machine can go higher, up to the manifest.
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "16")
    assert ball_detector.effective_batch_tiles(_manifest(None)) == 16


def test_effective_batch_tiles_env_cannot_exceed_the_manifest_ceiling(monkeypatch):
    # BALL_MAX_BATCH_TILES caps; it does not license a batch the exported
    # model never declared. A bigger ceiling is a manifest edit (or
    # BALL_MODEL_DIR), so the value stays attributable to the artifact.
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "64")
    assert ball_detector.effective_batch_tiles(_manifest(None)) == 32


def test_effective_batch_tiles_blank_env_reads_as_unset(monkeypatch):
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "   ")
    assert ball_detector.effective_batch_tiles(_manifest(None)) == 8


def test_effective_batch_tiles_rejects_non_integer_env(monkeypatch):
    # Silently ignoring a typo would leave the batch where it was on the
    # machine the operator was trying to rescue, and look like a no-op.
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "eight")
    with pytest.raises(ValueError, match="BALL_MAX_BATCH_TILES"):
        ball_detector.effective_batch_tiles(_manifest(None))


def test_effective_batch_tiles_rejects_zero_env(monkeypatch):
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "0")
    with pytest.raises(ValueError, match="BALL_MAX_BATCH_TILES"):
        ball_detector.effective_batch_tiles(_manifest(None))


def test_effective_batch_tiles_rejects_negative_env(monkeypatch):
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "-4")
    with pytest.raises(ValueError, match="BALL_MAX_BATCH_TILES"):
        ball_detector.effective_batch_tiles(_manifest(None))


def test_effective_batch_tiles_floors_a_zero_manifest_at_one():
    assert ball_detector.effective_batch_tiles(_manifest(None, max_batch_tiles=0)) == 1


def test_effective_batch_tiles_takes_no_device():
    # The cap deliberately does NOT branch on device -- CUDA gains nothing
    # from a bigger batch either (flat 16.8-18.2 ms/tile from batch 1 to 64),
    # so there is no device to consult and no per-device branch to drift.
    import inspect
    assert list(inspect.signature(
        ball_detector.effective_batch_tiles).parameters) == ["manifest"]


def test_detect_frame_caps_the_batch_below_the_manifest():
    frame = _marker_frame(3840, 2160, 1500, 900)
    runner = _MarkerRunner()

    ball_detector.detect_frame(runner, frame, _manifest(None))

    assert max(runner.batch_sizes) <= 8
    assert sum(runner.batch_sizes) == 66      # every tile still swept


def test_detect_frame_cap_leaves_detections_unchanged(monkeypatch):
    # The cap is a throughput knob. Same frame, same detections, only the
    # chunking differs -- this is the property the real-model A/B checks.
    frame = _marker_frame(3840, 2160, 352, 352)
    manifest = _manifest(None)

    capped = ball_detector.detect_frame(_MarkerRunner(), frame, manifest)
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "32")
    uncapped = ball_detector.detect_frame(_MarkerRunner(), frame, manifest)

    assert capped == uncapped
    assert len(capped) == 1


def test_detect_frame_honours_the_env_cap(monkeypatch):
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "5")
    frame = _marker_frame(3840, 2160, 1500, 900)
    runner = _MarkerRunner()

    detections = ball_detector.detect_frame(runner, frame, _manifest(None))

    assert max(runner.batch_sizes) <= 5
    assert sum(runner.batch_sizes) == 66
    assert detections[0]["x"] == pytest.approx(1500.0)


class _StackBatchRunner:
    """Records the batch sizes it is handed, and finds nothing."""

    def __init__(self):
        self.batch_sizes = []

    def run_batch(self, stacks):
        self.batch_sizes.append(len(stacks))
        return [[] for _ in stacks]


def test_detect_frame_stack_caps_the_batch_below_the_manifest():
    # The stack path is the one the pipeline actually runs (frames_per_input
    # 3), and it is the memory-hungrier of the two: 9 channels per tile. 66
    # tiles is a 4K frame, where the uncapped batch measured a Metal OOM.
    manifest = _stack_manifest()
    frames = [np.zeros((2160, 3840, 3), dtype=np.uint8)] * 3
    runner = _StackBatchRunner()

    ball_detector.detect_frame_stack(runner, frames, manifest)

    assert max(runner.batch_sizes) <= 8
    assert sum(runner.batch_sizes) == 66


def test_detect_frame_stack_env_can_restore_the_manifest_batch(monkeypatch):
    monkeypatch.setenv("BALL_MAX_BATCH_TILES", "32")
    manifest = _stack_manifest()
    frames = [np.zeros((2160, 3840, 3), dtype=np.uint8)] * 3
    runner = _StackBatchRunner()

    ball_detector.detect_frame_stack(runner, frames, manifest)

    assert max(runner.batch_sizes) == 32
