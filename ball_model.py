"""Manifest + artifact loading for the on-device ball detector's server twin.

Everything that varies between model iterations lives in manifest.json, so a
future model is a directory drop-in and one env var -- never a code change.

torch is imported lazily inside load_detector(): the default test suite runs
without it (see requirements-test.txt), and nothing here needs it to read a
manifest.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "ball-model-v1"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "crosscourt-ball-416-v1"
_EXPORT_HINT = (
    "Produce it in the training environment with export_ball_model.py "
    "(see ios/MODEL.md §2b)."
)

_MANIFEST_CACHE = {}
_DETECTOR_CACHE = {}


@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    name: str
    version: int
    input_size: tuple
    decode: str
    conf_threshold: float
    nms_iou: float
    class_names: tuple
    tile_overlap_px: int
    max_batch_tiles: int
    artifact_sha256: str
    source_checkpoint: str
    trained_commit: str
    val_ap50_95: float
    notes: str
    model_dir: Path

    @property
    def artifact_path(self):
        return self.model_dir / "model.torchscript"


def model_dir_from_env():
    configured = os.environ.get("BALL_MODEL_DIR", "").strip()
    return Path(configured) if configured else DEFAULT_MODEL_DIR


def load_manifest(model_dir=None):
    model_dir = Path(model_dir) if model_dir is not None else model_dir_from_env()
    cached = _MANIFEST_CACHE.get(model_dir)
    if cached is not None:
        return cached

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No ball-model manifest at {manifest_path}. {_EXPORT_HINT}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed manifest at {manifest_path}: {exc}") from exc

    schema = raw.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version {schema!r} at {manifest_path}; "
            f"this build understands {SCHEMA_VERSION!r}. Fields are not guessed.")

    artifact = model_dir / "model.torchscript"
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Manifest at {manifest_path} but no model.torchscript beside it. "
            f"{_EXPORT_HINT}")

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != raw.get("artifact_sha256"):
        raise ValueError(
            f"artifact sha256 mismatch for {artifact}: manifest says "
            f"{raw.get('artifact_sha256')!r}, file is {digest!r}. A mismatched "
            f"artifact makes every result unattributable -- re-run "
            f"export_ball_model.py to regenerate both together.")

    input_size = tuple(int(v) for v in raw["input_size"])
    overlap = int(raw["tile_overlap_px"])
    if not 0 <= overlap < min(input_size):
        raise ValueError(
            f"tile_overlap_px must be in [0, {min(input_size)}), got {overlap}")

    manifest = ModelManifest(
        schema_version=schema,
        name=str(raw["name"]),
        version=int(raw["version"]),
        input_size=input_size,
        decode=str(raw["decode"]),
        conf_threshold=float(raw["conf_threshold"]),
        nms_iou=float(raw["nms_iou"]),
        class_names=tuple(str(c) for c in raw["class_names"]),
        tile_overlap_px=overlap,
        max_batch_tiles=int(raw["max_batch_tiles"]),
        artifact_sha256=digest,
        source_checkpoint=str(raw.get("source_checkpoint", "")),
        trained_commit=str(raw.get("trained_commit", "")),
        val_ap50_95=float(raw.get("val_ap50_95", 0.0)),
        notes=str(raw.get("notes", "")),
        model_dir=model_dir,
    )
    _MANIFEST_CACHE[model_dir] = manifest
    return manifest


def describe(model_dir=None):
    """Provenance summary for stamping into output, or None if unavailable.

    Best-effort on purpose: this is reporting, not inference. Inference still
    fails loudly (load_detector) when the model is missing.
    """
    try:
        manifest = load_manifest(model_dir)
    except Exception:
        return None
    return {
        "name": manifest.name,
        "version": manifest.version,
        "artifact_sha256": manifest.artifact_sha256,
        "conf_threshold": manifest.conf_threshold,
        "nms_iou": manifest.nms_iou,
        "input_size": list(manifest.input_size),
        "source_checkpoint": manifest.source_checkpoint,
        "trained_commit": manifest.trained_commit,
    }


class TorchScriptRunner:
    """Runs the traced ball model over a batch of tile crops.

    Traced with decode_in_inference=True, so the graph emits decoded
    [batch, n_anchors, 5 + n_classes] and this class only thresholds by
    objectness and reshapes. The Core ML artifact (ios/MODEL.md §4) is traced
    the other way -- raw head outputs, decode in Swift for ANE residency --
    which is what manifest.decode distinguishes.
    """

    def __init__(self, module, manifest, torch_module):
        self._module = module
        self._torch = torch_module
        self.manifest = manifest

    def run_batch(self, crops):
        import numpy as np

        torch = self._torch
        # HWC uint8 BGR -> NCHW float32. YOLOX trains on raw 0-255 BGR as
        # cv2 delivers it (ValTransform applies no mean/std and no channel
        # swap), so pass the channels through unchanged and do not rescale.
        stacked = np.stack(crops).astype("float32")
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            raw = self._module(tensor)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        output = raw.detach().cpu().numpy()

        # Vectorised threshold before touching Python. A 416 crop yields ~3549
        # anchors; at 66 tiles that is ~234k rows per frame, and a per-row
        # Python loop over all of them would dominate the runtime.
        results = []
        for row in output:
            objectness = row[:, 4]
            class_scores = row[:, 5:]
            if class_scores.shape[1]:
                class_index = class_scores.argmax(axis=1)
                score = objectness * class_scores[
                    np.arange(class_scores.shape[0]), class_index]
            else:
                class_index = np.zeros(row.shape[0], dtype=int)
                score = objectness
            keep = np.nonzero(score >= self.manifest.conf_threshold)[0]
            results.append([
                (float(row[i, 0]), float(row[i, 1]),
                 float(row[i, 2]), float(row[i, 3]),
                 float(score[i]), int(class_index[i]))
                for i in keep
            ])
        return results


def load_detector(model_dir=None):
    """Load the traced model. Raises loudly; never falls back to another detector."""
    model_dir = Path(model_dir) if model_dir is not None else model_dir_from_env()
    cached = _DETECTOR_CACHE.get(model_dir)
    if cached is not None:
        return cached

    manifest = load_manifest(model_dir)

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The local ball detector needs torch. The default test env is "
            "deliberately torch-free; install the full requirements.txt."
        ) from exc

    module = torch.jit.load(str(manifest.artifact_path), map_location="cpu")
    module.eval()
    runner = TorchScriptRunner(module, manifest, torch)
    _DETECTOR_CACHE[model_dir] = runner
    return runner
