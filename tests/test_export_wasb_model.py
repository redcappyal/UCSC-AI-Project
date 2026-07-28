"""Tests for export_wasb_model.py's pure parts.

This module must stay torch-free at import time (see export_wasb_model.py's
docstring) so this test file can run in the app venv, which has no torch.
Only build_manifest() and derive_heatmap_stride() are under test -- the
actual trace/export path (main()) needs torch + a WASB-SBDT checkout and
runs only in the training environment (see docs/WASB-TRAIN.md).
"""
import pytest

import export_wasb_model


def test_build_manifest_dict_v2_fields():
    manifest = export_wasb_model.build_manifest(
        name="crosscourt-wasb-416", version=1, input_size=416,
        frames_per_input=3, heatmap_stride=2, nominal_ball_px=12.0,
        conf_threshold=0.1, nms_iou=0.45, tile_overlap_px=64,
        max_batch_tiles=32, artifact_sha256="deadbeef",
        source_checkpoint="wasb_best.pth", trained_commit="abc1234",
        val_metric=0.0, notes="")
    assert manifest["schema_version"] == "ball-model-v2"
    assert manifest["decode"] == "heatmap_peak"
    assert manifest["frames_per_input"] == 3
    assert manifest["class_names"] == ["ball"]


def test_build_manifest_carries_every_v2_field_from_task1():
    manifest = export_wasb_model.build_manifest(
        name="crosscourt-wasb-416", version=1, input_size=416,
        frames_per_input=3, heatmap_stride=2, nominal_ball_px=12.0,
        conf_threshold=0.1, nms_iou=0.45, tile_overlap_px=64,
        max_batch_tiles=32, artifact_sha256="deadbeef",
        source_checkpoint="wasb_best.pth", trained_commit="abc1234",
        val_metric=0.4034, notes="diagnostic only")
    # Full v2 field list ball_model.load_manifest requires (Task 1) --
    # KeyError on anything missing, v1 or v2.
    for field in (
        "schema_version", "name", "version", "input_size", "decode",
        "conf_threshold", "nms_iou", "class_names", "tile_overlap_px",
        "max_batch_tiles", "artifact_sha256", "frames_per_input",
        "heatmap_stride", "nominal_ball_px",
    ):
        assert field in manifest, f"missing required manifest field {field!r}"
    assert manifest["input_size"] == [416, 416]
    assert manifest["heatmap_stride"] == 2
    assert manifest["nominal_ball_px"] == 12.0
    assert manifest["source_checkpoint"] == "wasb_best.pth"
    assert manifest["trained_commit"] == "abc1234"
    assert manifest["notes"] == "diagnostic only"


def test_build_manifest_frames_per_input_passthrough_not_hardcoded():
    # frames_per_input must flow through unchanged -- nothing in
    # build_manifest should assume 3.
    manifest = export_wasb_model.build_manifest(
        name="crosscourt-wasb-416", version=1, input_size=416,
        frames_per_input=5, heatmap_stride=1, nominal_ball_px=12.0,
        conf_threshold=0.1, nms_iou=0.45, tile_overlap_px=64,
        max_batch_tiles=32, artifact_sha256="deadbeef",
        source_checkpoint="wasb_best.pth", trained_commit="abc1234",
        val_metric=0.0, notes="")
    assert manifest["frames_per_input"] == 5


# -- derive_heatmap_stride: the measure-don't-assume rule (docs/WASB-TRAIN.md
# §6) -- WASB's default config yields stride 1, not the stride-2/4 a
# conventional heatmap detector would use, so this must be derived from the
# real traced output shape, never hardcoded, and cross-checked against an
# operator-supplied --heatmap-stride if given.

def test_derive_heatmap_stride_measures_from_output_shape():
    # 416 input, 208x208 heatmap -> stride 2.
    stride = export_wasb_model.derive_heatmap_stride(
        (1, 3, 208, 208), frames_per_input=3, input_size=416)
    assert stride == 2


def test_derive_heatmap_stride_handles_wasb_default_stride_one():
    # WASB's actual default config (docs/WASB-TRAIN.md §3): stride 1, output
    # resolution equals input resolution. Must not be assumed away.
    stride = export_wasb_model.derive_heatmap_stride(
        (1, 3, 416, 416), frames_per_input=3, input_size=416)
    assert stride == 1


def test_derive_heatmap_stride_rejects_frame_count_mismatch():
    with pytest.raises(
            ValueError,
            match=r"traced output has 1 frame channels but --frames-per-input=3"):
        export_wasb_model.derive_heatmap_stride(
            (1, 1, 208, 208), frames_per_input=3, input_size=416)


def test_derive_heatmap_stride_names_actual_shape_on_frame_mismatch():
    with pytest.raises(ValueError) as excinfo:
        export_wasb_model.derive_heatmap_stride(
            (1, 1, 208, 208), frames_per_input=3, input_size=416)
    assert "(1, 1, 208, 208)" in str(excinfo.value)


def test_derive_heatmap_stride_rejects_non_square_heatmap():
    with pytest.raises(ValueError, match=r"(?i)square"):
        export_wasb_model.derive_heatmap_stride(
            (1, 3, 208, 209), frames_per_input=3, input_size=416)


def test_derive_heatmap_stride_rejects_non_divisible_size():
    with pytest.raises(ValueError, match=r"(?i)divis"):
        export_wasb_model.derive_heatmap_stride(
            (1, 3, 300, 300), frames_per_input=3, input_size=416)


def test_derive_heatmap_stride_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"(?i)shape"):
        export_wasb_model.derive_heatmap_stride(
            (1, 208, 208), frames_per_input=3, input_size=416)


def test_derive_heatmap_stride_cross_checks_cli_value_matching():
    # Operator supplied --heatmap-stride matching the measured value: fine.
    stride = export_wasb_model.derive_heatmap_stride(
        (1, 3, 208, 208), frames_per_input=3, input_size=416,
        cli_heatmap_stride=2)
    assert stride == 2


def test_derive_heatmap_stride_cross_check_mismatch_is_fatal_naming_both():
    with pytest.raises(ValueError) as excinfo:
        export_wasb_model.derive_heatmap_stride(
            (1, 3, 208, 208), frames_per_input=3, input_size=416,
            cli_heatmap_stride=4)
    message = str(excinfo.value)
    assert "2" in message  # measured
    assert "4" in message  # operator-supplied


def test_derive_heatmap_stride_rejects_batch_dim_not_one():
    # Export always traces a single-example input; a batch dim other than 1
    # means the traced graph itself is wrong, not something to shrug past.
    with pytest.raises(ValueError, match=r"(?i)batch"):
        export_wasb_model.derive_heatmap_stride(
            (2, 3, 208, 208), frames_per_input=3, input_size=416)


def test_derive_heatmap_stride_names_actual_shape_on_batch_mismatch():
    with pytest.raises(ValueError) as excinfo:
        export_wasb_model.derive_heatmap_stride(
            (2, 3, 208, 208), frames_per_input=3, input_size=416)
    assert "(2, 3, 208, 208)" in str(excinfo.value)


# -- validate_export_inputs: pre-write fatal checks needing no torch --------


def test_validate_export_inputs_rejects_even_frames_per_input():
    with pytest.raises(ValueError, match=r"(?i)odd"):
        export_wasb_model.validate_export_inputs(4, 12.0)


def test_validate_export_inputs_rejects_non_positive_nominal_ball_px():
    with pytest.raises(ValueError, match=r"(?i)nominal-ball-px"):
        export_wasb_model.validate_export_inputs(3, 0.0)


def test_validate_export_inputs_rejects_negative_nominal_ball_px():
    with pytest.raises(ValueError, match=r"(?i)nominal-ball-px"):
        export_wasb_model.validate_export_inputs(3, -1.0)


def test_validate_export_inputs_accepts_valid_values():
    export_wasb_model.validate_export_inputs(3, 12.0)   # must not raise


# -- validate_traced_output_is_probabilities: the logits-vs-probabilities ---
# export boundary (docs/WASB-TRAIN.md §6) -- pure over a numpy array so it's
# testable without torch, same as derive_heatmap_stride.


def test_validate_traced_output_is_probabilities_passes_in_range():
    np = pytest.importorskip("numpy")
    # Must not raise.
    export_wasb_model.validate_traced_output_is_probabilities(
        np.array([0.0, 0.1, 0.5, 1.0], dtype="float32"))


def test_validate_traced_output_rejects_out_of_range_naming_min_max():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError) as excinfo:
        export_wasb_model.validate_traced_output_is_probabilities(
            np.array([-2.5, 0.5, 3.7], dtype="float32"))
    message = str(excinfo.value)
    assert "-2.5" in message
    assert "3.7" in message


def test_validate_traced_output_rejects_bare_logits_shape():
    # A raw (un-sigmoided) HRNet output routinely lands outside [0, 1] --
    # this is the exact failure this check exists to catch before any file
    # is written.
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match=r"(?i)\[0, 1\]"):
        export_wasb_model.validate_traced_output_is_probabilities(
            np.array([-4.2, 0.0, 6.8], dtype="float32"))


def test_attrdict_bridges_item_and_attribute_access():
    # HRNet reads cfg['frames_in'] in __init__ but cfg.MODEL.EXTRA in
    # _make_deconv_layers; a config object must survive both styles, nested.
    cfg = export_wasb_model.AttrDict(
        {"frames_in": 3, "MODEL": {"EXTRA": {"STEM": {"STRIDES": [1, 1]}}}})
    assert cfg["frames_in"] == 3
    assert cfg.MODEL.EXTRA.STEM.STRIDES == [1, 1]
    assert cfg["MODEL"]["EXTRA"]["STEM"]["STRIDES"] == [1, 1]
    with pytest.raises(AttributeError):
        _ = cfg.missing_key
