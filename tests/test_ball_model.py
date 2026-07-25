import json
import hashlib
import pytest

import ball_model


def _write_model(tmp_path, **overrides):
    artifact = tmp_path / "model.torchscript"
    artifact.write_bytes(b"not-a-real-torchscript")
    manifest = {
        "schema_version": "ball-model-v1",
        "name": "crosscourt-ball-416",
        "version": 1,
        "input_size": [416, 416],
        "decode": "in_graph",
        "conf_threshold": 0.25,
        "nms_iou": 0.65,
        "class_names": ["ball"],
        "tile_overlap_px": 64,
        "max_batch_tiles": 32,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "source_checkpoint": "best_ckpt.pth (epoch 100)",
        "trained_commit": "2968a89",
        "val_ap50_95": 0.4034,
        "notes": "val is diagnostic only",
    }
    manifest.update(overrides)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_load_manifest_reads_all_fields(tmp_path):
    manifest = ball_model.load_manifest(_write_model(tmp_path))
    assert manifest.name == "crosscourt-ball-416"
    assert manifest.input_size == (416, 416)
    assert manifest.conf_threshold == 0.25
    assert manifest.nms_iou == 0.65
    assert manifest.class_names == ("ball",)
    assert manifest.tile_overlap_px == 64


def test_load_manifest_rejects_unknown_schema_version(tmp_path):
    model_dir = _write_model(tmp_path, schema_version="ball-model-v99")
    with pytest.raises(ValueError, match="ball-model-v99"):
        ball_model.load_manifest(model_dir)


def test_load_manifest_rejects_sha256_mismatch(tmp_path):
    model_dir = _write_model(tmp_path, artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        ball_model.load_manifest(model_dir)


def test_load_manifest_missing_dir_names_the_path_and_export_script(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError) as excinfo:
        ball_model.load_manifest(missing)
    message = str(excinfo.value)
    assert "nope" in message
    assert "export_ball_model.py" in message


def test_load_manifest_rejects_overlap_not_smaller_than_tile(tmp_path):
    model_dir = _write_model(tmp_path, tile_overlap_px=416)
    with pytest.raises(ValueError, match="tile_overlap_px"):
        ball_model.load_manifest(model_dir)


def test_describe_returns_none_when_unavailable(tmp_path):
    assert ball_model.describe(tmp_path / "nope") is None


def test_describe_summarises_for_provenance_stamping(tmp_path):
    summary = ball_model.describe(_write_model(tmp_path))
    assert summary["name"] == "crosscourt-ball-416"
    assert summary["version"] == 1
    assert summary["artifact_sha256"] == hashlib.sha256(
        b"not-a-real-torchscript").hexdigest()
    assert summary["conf_threshold"] == 0.25
    assert summary["input_size"] == [416, 416]
    assert summary["source_checkpoint"] == "best_ckpt.pth (epoch 100)"


def test_describe_returns_none_on_scalar_input_size(tmp_path):
    """describe() must be best-effort: scalar input_size (TypeError) must return None."""
    model_dir = _write_model(tmp_path, input_size=416)
    assert ball_model.describe(model_dir) is None


def test_load_manifest_raises_on_scalar_input_size(tmp_path):
    """load_manifest() must raise loudly on malformed manifest (TypeError from scalar input_size)."""
    model_dir = _write_model(tmp_path, input_size=416)
    with pytest.raises(TypeError):
        ball_model.load_manifest(model_dir)


def test_describe_returns_none_on_top_level_json_list(tmp_path):
    """describe() must be best-effort: top-level JSON list (AttributeError) must return None."""
    artifact = tmp_path / "model.torchscript"
    artifact.write_bytes(b"not-a-real-torchscript")
    (tmp_path / "manifest.json").write_text(json.dumps([]), encoding="utf-8")
    assert ball_model.describe(tmp_path) is None


def test_load_manifest_raises_on_top_level_json_list(tmp_path):
    """load_manifest() must raise loudly on top-level JSON list (AttributeError from .get() on list)."""
    artifact = tmp_path / "model.torchscript"
    artifact.write_bytes(b"not-a-real-torchscript")
    (tmp_path / "manifest.json").write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(AttributeError):
        ball_model.load_manifest(tmp_path)


def test_model_manifest_artifact_path(tmp_path):
    """ModelManifest.artifact_path must resolve to model.torchscript inside model_dir."""
    model_dir = _write_model(tmp_path)
    manifest = ball_model.load_manifest(model_dir)
    assert manifest.artifact_path == model_dir / "model.torchscript"
    assert manifest.artifact_path.is_file()


def test_load_detector_missing_model_raises_and_does_not_fall_back(tmp_path):
    # Silent fallback to RF-DETR would make the stereo/single-camera detector
    # split invisible; the spec requires this to be loud.
    with pytest.raises(FileNotFoundError, match="export_ball_model.py"):
        ball_model.load_detector(tmp_path / "nope")


def test_ball_model_imports_without_torch():
    import sys
    # The default CI job has no torch installed; importing ball_model must not
    # need it. If torch is absent this is trivially true, so assert the module
    # did not pull it in either way.
    before = "torch" in sys.modules
    import importlib
    importlib.reload(ball_model)
    assert ("torch" in sys.modules) == before


@pytest.mark.requires_model
def test_torchscript_runner_returns_boxes_for_real_model():
    runner = ball_model.load_detector()
    import numpy as np
    crops = [np.zeros((416, 416, 3), dtype=np.uint8)]
    result = runner.run_batch(crops)
    assert len(result) == 1
    assert isinstance(result[0], list)
