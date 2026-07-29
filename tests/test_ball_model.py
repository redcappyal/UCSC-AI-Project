import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import ball_model


def _runner_with_manifest(**fields):
    """A HeatmapRunner with no real module/torch -- only _decode_output is under test."""
    manifest = SimpleNamespace(**fields)
    return ball_model.HeatmapRunner(module=None, manifest=manifest, torch_module=None)


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


def test_v2_manifest_loads_temporal_fields(tmp_path):
    model_dir = _write_model(
        tmp_path,
        schema_version="ball-model-v2",
        decode="heatmap_peak",
        frames_per_input=3,
        heatmap_stride=2,
        nominal_ball_px=12.0,
    )
    manifest = ball_model.load_manifest(model_dir)
    assert manifest.frames_per_input == 3
    assert manifest.decode == "heatmap_peak"
    assert manifest.heatmap_stride == 2
    assert manifest.nominal_ball_px == 12.0


def test_v1_manifest_reports_single_frame_defaults(tmp_path):
    manifest = ball_model.load_manifest(_write_model(tmp_path))
    assert manifest.frames_per_input == 1
    assert manifest.decode == "in_graph"
    assert manifest.heatmap_stride == 0
    assert manifest.nominal_ball_px == 0.0


def test_v2_manifest_rejects_missing_frames_per_input(tmp_path):
    model_dir = _write_model(
        tmp_path,
        schema_version="ball-model-v2",
        decode="heatmap_peak",
        heatmap_stride=2,
        nominal_ball_px=12.0,
    )
    with pytest.raises(KeyError):
        ball_model.load_manifest(model_dir)


def test_v2_manifest_rejects_nonpositive_heatmap_stride(tmp_path):
    model_dir = _write_model(
        tmp_path,
        schema_version="ball-model-v2",
        decode="heatmap_peak",
        frames_per_input=3,
        heatmap_stride=0,
        nominal_ball_px=12.0,
    )
    with pytest.raises(ValueError, match="heatmap_stride"):
        ball_model.load_manifest(model_dir)


def test_v2_manifest_rejects_zero_frames_per_input(tmp_path):
    model_dir = _write_model(
        tmp_path,
        schema_version="ball-model-v2",
        decode="heatmap_peak",
        frames_per_input=0,
        heatmap_stride=2,
        nominal_ball_px=12.0,
    )
    with pytest.raises(ValueError, match="frames_per_input"):
        ball_model.load_manifest(model_dir)


def test_v2_manifest_rejects_even_frames_per_input(tmp_path):
    model_dir = _write_model(
        tmp_path,
        schema_version="ball-model-v2",
        decode="heatmap_peak",
        frames_per_input=2,
        heatmap_stride=2,
        nominal_ball_px=12.0,
    )
    with pytest.raises(ValueError, match="odd"):
        ball_model.load_manifest(model_dir)


def test_v2_manifest_rejects_nonpositive_nominal_ball_px(tmp_path):
    model_dir = _write_model(
        tmp_path,
        schema_version="ball-model-v2",
        decode="heatmap_peak",
        frames_per_input=3,
        heatmap_stride=2,
        nominal_ball_px=0.0,
    )
    with pytest.raises(ValueError, match="nominal_ball_px"):
        ball_model.load_manifest(model_dir)


def test_v2_manifest_rejects_non_heatmap_peak_decode(tmp_path):
    model_dir = _write_model(
        tmp_path,
        schema_version="ball-model-v2",
        decode="in_graph",
        frames_per_input=3,
        heatmap_stride=2,
        nominal_ball_px=12.0,
    )
    with pytest.raises(ValueError, match="heatmap_peak"):
        ball_model.load_manifest(model_dir)


def test_model_manifest_artifact_path(tmp_path):
    """ModelManifest.artifact_path must resolve to model.torchscript inside model_dir."""
    model_dir = _write_model(tmp_path)
    manifest = ball_model.load_manifest(model_dir)
    assert manifest.artifact_path == model_dir / "model.torchscript"
    assert manifest.artifact_path.is_file()


def test_load_detector_missing_model_raises_and_does_not_fall_back(tmp_path):
    # Silent fallback to RF-DETR would make the local/hosted detector split
    # invisible; the spec requires this to be loud.
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
    crops = [np.zeros((416, 416, 3), dtype=np.uint8)]
    result = runner.run_batch(crops)
    assert len(result) == 1
    assert isinstance(result[0], list)


def test_decode_heatmap_finds_subpixel_peak():
    hm = np.zeros((208, 208), dtype=np.float32)
    hm[50, 100] = 0.9
    hm[50, 101] = 0.6          # pulls the sub-pixel x positively
    peaks = ball_model.decode_heatmap(hm, threshold=0.1, stride=2, nominal_px=12.0)
    assert len(peaks) == 1
    cx, cy, score = peaks[0]
    assert score == pytest.approx(0.9)
    assert cy == pytest.approx(100.0, abs=1.0)          # 50 * stride
    assert 200.0 < cx < 204.0                           # 100 * stride, nudged right


def test_decode_heatmap_below_threshold_is_empty():
    hm = np.full((208, 208), 0.05, dtype=np.float32)
    assert ball_model.decode_heatmap(hm, 0.1, 2, 12.0) == []


def test_decode_heatmap_two_separated_peaks_both_found():
    hm = np.zeros((208, 208), dtype=np.float32)
    hm[20, 20] = 0.8
    hm[150, 150] = 0.5
    peaks = ball_model.decode_heatmap(hm, 0.1, 2, 12.0)
    assert len(peaks) == 2


def test_decode_heatmap_plateau_emits_single_peak():
    hm = np.zeros((208, 208), dtype=np.float32)
    hm[30, 30] = hm[30, 31] = 0.7    # two equal neighbours must not double-fire
    peaks = ball_model.decode_heatmap(hm, 0.1, 2, 12.0)
    assert len(peaks) == 1


def test_decode_heatmap_wide_plateau_emits_single_peak():
    # A 3px-wide flat run: the 3rd pixel's 3x3 window no longer overlaps the
    # winner's window directly, so suppression must chain through the
    # already-skipped 2nd pixel rather than only reaching one step.
    hm = np.zeros((208, 208), dtype=np.float32)
    hm[30, 30] = hm[30, 31] = hm[30, 32] = 0.7
    peaks = ball_model.decode_heatmap(hm, 0.1, 2, 12.0)
    assert len(peaks) == 1


def test_decode_heatmap_long_plateau_emits_single_peak():
    hm = np.zeros((208, 208), dtype=np.float32)
    hm[30, 30:35] = 0.7    # 5px-wide flat run
    peaks = ball_model.decode_heatmap(hm, 0.1, 2, 12.0)
    assert len(peaks) == 1


def test_heatmap_runner_decodes_only_the_middle_channel():
    # Output [B, frames, Hh, Wh]; runner must decode ONLY the middle
    # (centre-frame) channel and emit TorchScriptRunner-shaped tuples.
    out = np.zeros((1, 3, 208, 208), dtype=np.float32)
    out[:, 1, 50, 100] = 0.9   # middle frame -- the one being detected
    out[:, 0, 10, 10] = 0.9    # past-frame channel -- must be ignored
    out[:, 2, 90, 90] = 0.9    # future-frame channel -- must be ignored
    runner = _runner_with_manifest(frames_per_input=3, conf_threshold=0.1,
                                   heatmap_stride=2, nominal_ball_px=12.0)
    results = runner._decode_output(out)
    assert len(results) == 1 and len(results[0]) == 1
    cx, cy, w, h, score, class_index = results[0][0]
    assert (cy, score, class_index) == (pytest.approx(100.0, abs=1.0), pytest.approx(0.9), 0)
    assert w == h == 12.0


def test_load_detector_dispatches_to_heatmap_runner_for_v2_manifest(tmp_path, monkeypatch):
    """load_detector must route decode == 'heatmap_peak' to HeatmapRunner, not TorchScriptRunner."""
    import sys
    import types

    model_dir = _write_model(
        tmp_path,
        schema_version="ball-model-v2",
        decode="heatmap_peak",
        frames_per_input=3,
        heatmap_stride=2,
        nominal_ball_px=12.0,
    )

    fake_module = SimpleNamespace(eval=lambda: None)
    fake_torch = types.SimpleNamespace(
        jit=types.SimpleNamespace(load=lambda *a, **k: fake_module),
        cuda=_FakeCuda(False),
        backends=types.SimpleNamespace(mps=_FakeMps(False)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    runner = ball_model.load_detector(model_dir)
    assert isinstance(runner, ball_model.HeatmapRunner)


class _FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeMps:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


def _fake_torch(cuda=False, mps=False):
    import types
    return types.SimpleNamespace(
        cuda=_FakeCuda(cuda),
        backends=types.SimpleNamespace(mps=_FakeMps(mps)),
    )


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.delenv("BALL_DEVICE", raising=False)
    assert ball_model._resolve_device(_fake_torch(cuda=True)) == "cuda"


def test_resolve_device_auto_without_cuda_is_cpu_never_mps(monkeypatch):
    # MPS available but auto must not pick it (8 GB unified-memory machines).
    monkeypatch.delenv("BALL_DEVICE", raising=False)
    assert ball_model._resolve_device(_fake_torch(mps=True)) == "cpu"


def test_resolve_device_explicit_mps_honored(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "mps")
    assert ball_model._resolve_device(_fake_torch(mps=True)) == "mps"


def test_resolve_device_explicit_mps_unavailable_raises(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "mps")
    with pytest.raises(RuntimeError, match="MPS"):
        ball_model._resolve_device(_fake_torch(mps=False))


def test_resolve_device_explicit_cuda_unavailable_raises(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="CUDA|cuda"):
        ball_model._resolve_device(_fake_torch(cuda=False))


def test_resolve_device_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "tpu")
    with pytest.raises(ValueError, match="tpu"):
        ball_model._resolve_device(_fake_torch())


def test_resolve_device_explicit_cpu(monkeypatch):
    monkeypatch.setenv("BALL_DEVICE", "cpu")
    assert ball_model._resolve_device(_fake_torch(cuda=True)) == "cpu"
