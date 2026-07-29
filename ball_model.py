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
SCHEMA_VERSION_V2 = "ball-model-v2"
SCHEMA_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_V2)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "crosscourt-wasb-416-v1"
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
    frames_per_input: int = 1
    heatmap_stride: int = 0
    nominal_ball_px: float = 0.0

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
    if schema not in SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported manifest schema_version {schema!r} at {manifest_path}; "
            f"this build understands {SCHEMA_VERSIONS!r}. Fields are not guessed.")

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

    if schema == SCHEMA_VERSION_V2:
        frames_per_input = int(raw["frames_per_input"])   # KeyError on absence is the contract
        heatmap_stride = int(raw["heatmap_stride"])
        nominal_ball_px = float(raw["nominal_ball_px"])
        if frames_per_input < 1:
            raise ValueError(f"frames_per_input must be >= 1, got {frames_per_input}")
        if frames_per_input % 2 == 0:
            raise ValueError(
                f"frames_per_input must be odd for heatmap_peak decode (the "
                f"detected frame sits in the middle of a centered window), "
                f"got {frames_per_input}")
        if heatmap_stride < 1:
            raise ValueError(f"heatmap_stride must be >= 1, got {heatmap_stride}")
        if nominal_ball_px <= 0:
            raise ValueError(f"nominal_ball_px must be > 0, got {nominal_ball_px}")
        if str(raw["decode"]) != "heatmap_peak":
            raise ValueError(
                f"ball-model-v2 only supports decode='heatmap_peak', got {raw['decode']!r}")
    else:
        frames_per_input, heatmap_stride, nominal_ball_px = 1, 0, 0.0

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
        frames_per_input=frames_per_input,
        heatmap_stride=heatmap_stride,
        nominal_ball_px=nominal_ball_px,
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


def _resolve_device(torch_module):
    """BALL_DEVICE -> torch device string.

    Auto (unset or "auto") picks CUDA when available, else CPU. It never
    picks MPS: HRNet over 32-tile batches on an 8 GB unified-memory machine
    is a realistic memory-pressure panic, so MPS is opt-in only
    (BALL_DEVICE=mps). An explicitly requested device that is unavailable
    raises rather than falling back -- a silent CPU fallback would make a
    "fast" run quietly 50x slower.
    """
    configured = os.environ.get("BALL_DEVICE", "").strip().lower()
    if configured in ("", "auto"):
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if configured == "cpu":
        return "cpu"
    if configured == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("BALL_DEVICE=cuda but torch reports no CUDA device")
        return "cuda"
    if configured == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("BALL_DEVICE=mps but torch reports MPS unavailable")
        return "mps"
    raise ValueError(
        f"Unknown BALL_DEVICE {configured!r}; expected auto, cpu, cuda or mps")


class TorchScriptRunner:
    """Runs the traced ball model over a batch of tile crops.

    Traced with decode_in_inference=True, so the graph emits decoded
    [batch, n_anchors, 5 + n_classes] and this class only thresholds by
    objectness and reshapes. The Core ML artifact (ios/MODEL.md §4) is traced
    the other way -- raw head outputs, decode in Swift for ANE residency --
    which is what manifest.decode distinguishes.
    """

    def __init__(self, module, manifest, torch_module, device="cpu"):
        self._module = module
        self._torch = torch_module
        self.manifest = manifest
        self.device = device

    def run_batch(self, crops):
        import numpy as np

        torch = self._torch
        # HWC uint8 BGR -> NCHW float32. YOLOX trains on raw 0-255 BGR as
        # cv2 delivers it (ValTransform applies no mean/std and no channel
        # swap), so pass the channels through unchanged and do not rescale.
        stacked = np.stack(crops).astype("float32")
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
        if self.device != "cpu":
            tensor = tensor.to(self.device)

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


def decode_heatmap(heatmap, threshold, stride, nominal_px):
    """Sub-pixel peaks of one heatmap, in input-pixel coordinates.

    Local maxima (8-neighbourhood, strict against later-scanned equals so a
    flat 2-px plateau fires once) above `threshold`, refined by a 3x3
    centre-of-mass. Returns [(cx, cy, score)] sorted by score descending.
    `nominal_px` is not used here (it sizes the reported box in the runner);
    it is in the signature so callers hold the full decode contract in one
    place.
    """
    import numpy as np

    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"decode_heatmap expects a 2D heatmap, got shape {hm.shape}")
    padded = np.pad(hm, 1, mode="constant", constant_values=-np.inf)
    neighbourhoods = np.stack([
        padded[dy:dy + hm.shape[0], dx:dx + hm.shape[1]]
        for dy in range(3) for dx in range(3) if not (dy == 1 and dx == 1)
    ])
    is_peak = (hm >= neighbourhoods.max(axis=0)) & (hm >= threshold)
    # break plateau ties: keep only the first occurrence in scan order. A
    # skipped pixel is marked taken too (not just the winner), so suppression
    # propagates along an arbitrarily wide flat run instead of only reaching
    # one 3x3 window past the winner -- otherwise a plateau >=3px wide would
    # let its far end re-fire as a second "peak".
    ys, xs = np.nonzero(is_peak)
    peaks, taken = [], np.zeros_like(hm, dtype=bool)
    for y, x in zip(ys, xs):
        already_taken = taken[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].any()
        taken[y, x] = True
        if already_taken:
            continue
        window = hm[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        wy, wx = np.mgrid[max(y - 1, 0):min(y + 2, hm.shape[0]),
                          max(x - 1, 0):min(x + 2, hm.shape[1])]
        weight = np.clip(window, 0, None)
        total = float(weight.sum())
        cy = float((weight * wy).sum() / total) if total > 0 else float(y)
        cx = float((weight * wx).sum() / total) if total > 0 else float(x)
        peaks.append((cx * stride, cy * stride, float(hm[y, x])))
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks


class HeatmapRunner:
    """Runs the traced WASB-style model over co-located multi-frame tile stacks.

    The graph emits [batch, frames_per_input, Hh, Wh] MIMO heatmaps; only the
    MIDDLE (centre-frame) channel is decoded -- analysis is offline, so every
    frame is detected with both its past and future neighbour in view, which
    is what carries the model through direction reversals at wall and floor
    contacts. Training supervises all frames; serving answers "where is the
    ball in the CENTRE frame".

    Input stacks are BGR frames concatenated oldest-first along the channel
    axis, raw 0-255, matching the training adapter in docs/WASB-TRAIN.md. No
    mean/std, consistent with TorchScriptRunner's BGR-raw convention.

    run_batch expects the traced graph to emit sigmoid PROBABILITIES in
    [0, 1], not raw logits -- export_wasb_model.py wraps sigmoid in-graph at
    trace time (docs/WASB-TRAIN.md §6), so a checkpoint being logits-native
    is invisible here.
    """

    def __init__(self, module, manifest, torch_module, device="cpu"):
        self._module = module
        self._torch = torch_module
        self.manifest = manifest
        self.device = device

    def _decode_output(self, output):
        results = []
        size = float(self.manifest.nominal_ball_px)
        middle = int(self.manifest.frames_per_input) // 2
        for heatmaps in output:                      # [frames, Hh, Wh]
            peaks = decode_heatmap(
                heatmaps[middle], self.manifest.conf_threshold,
                self.manifest.heatmap_stride, size)
            results.append([(cx, cy, size, size, score, 0)
                            for cx, cy, score in peaks])
        return results

    def run_batch(self, stacks):
        import numpy as np

        torch = self._torch
        stacked = np.stack(stacks).astype("float32")           # [B, H, W, 3*F]
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        with torch.no_grad():
            raw = self._module(tensor)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        return self._decode_output(raw.detach().cpu().numpy())


def load_detector(model_dir=None):
    """Load the traced model. Raises loudly; never falls back to another detector.

    BALL_DEVICE picks the compute device (see _resolve_device: CUDA auto,
    MPS opt-in). It is read once per model_dir -- the runner is cached --
    so changing it mid-process has no effect.
    """
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

    device = _resolve_device(torch)
    module = torch.jit.load(str(manifest.artifact_path), map_location="cpu")
    module.eval()
    if device != "cpu":
        module = module.to(device)
    if manifest.decode == "heatmap_peak":
        runner = HeatmapRunner(module, manifest, torch, device=device)
    else:
        runner = TorchScriptRunner(module, manifest, torch, device=device)
    _DETECTOR_CACHE[model_dir] = runner
    return runner
