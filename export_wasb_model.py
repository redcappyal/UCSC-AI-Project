"""Trace a fine-tuned WASB (HRNet, MIMO heatmap) ball checkpoint to
TorchScript + a ball-model-v2 manifest.

Run this in the TRAINING environment (the box `docs/WASB-TRAIN.md` describes,
with WASB-SBDT cloned per its §1), never in the app env. That is the whole
point: WASB-SBDT's `HRNet` stays a training-only dependency -- imported only
inside main() below -- so the serving path (`ball_model.py`,
`ball_detector.py`) loads a self-contained traced artifact and torch alone,
the same boundary `export_ball_model.py` draws for YOLOX.

The one thing this script must never do is assume `heatmap_stride`.
WASB-SBDT's *default* config produces stride 1 (output resolution equals
input resolution), not the stride-2/4 a conventional heatmap detector uses --
see docs/WASB-TRAIN.md §3/§6. So the traced module is run once on the example
input here, the real output shape is measured, and `heatmap_stride` is
derived from that measurement (input_size // heatmap_height), cross-checked
against `--heatmap-stride` if the operator supplied one. A mismatch is
fatal -- a wrong stride silently breaks `ball_model.py`'s heatmap-peak
coordinate scaling on the Mac.

    python export_wasb_model.py \
        --ckpt wasb_runs/crosscourt-ball-wasb-v1/wasb_best.pth \
        --wasb-repo C:\\path\\to\\WASB-SBDT \
        --wasb-cfg configs/model/wasb-crosscourt-416.yaml \
        --out models/crosscourt-wasb-416-v1 \
        --version 1
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SCHEMA_VERSION = "ball-model-v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="path to wasb_best.pth")
    parser.add_argument("--wasb-repo", required=True,
                         help="path to the WASB-SBDT clone (docs/WASB-TRAIN.md "
                              "§1, pinned commit 923462c); its src/ is added "
                              "to sys.path to import HRNet")
    parser.add_argument("--wasb-cfg", required=True,
                         help="path to the HRNet model config (YAML/JSON) the "
                              "checkpoint was trained with -- must match "
                              "train_wasb.py's config, not WASB-SBDT's "
                              "unmodified configs/model/wasb.yaml (input size "
                              "changed to 416x416 per docs/WASB-TRAIN.md §3)")
    parser.add_argument("--out", required=True, help="output model directory")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--name", default="crosscourt-wasb-416")
    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument("--frames-per-input", type=int, default=3,
                         help="must be odd -- ball_model.py's v2 loader "
                              "rejects an even value")
    parser.add_argument("--heatmap-stride", type=int, default=None,
                         help="optional cross-check only: the manifest value "
                              "is always the one MEASURED from the traced "
                              "output shape, never this. If given, a mismatch "
                              "against the measured value is fatal")
    parser.add_argument("--nominal-ball-px", type=float, default=12.0)
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                         help="deliberately low (BYTE-style rescue): weak "
                              "heatmap peaks flow into tracking_common's "
                              "motion-consistency scorer, which promotes "
                              "moving candidates +0.30 and suppresses "
                              "stationary ones -0.40; the selection bar stays "
                              "at the caller's 0.4 (detections_to_track_samples)")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--tile-overlap-px", type=int, default=64)
    parser.add_argument("--max-batch-tiles", type=int, default=32)
    parser.add_argument("--val-metric", type=float, default=0.0,
                         help="val heatmap F1 at conf_threshold (docs/"
                              "WASB-TRAIN.md §3's early-stop metric), stored "
                              "in the manifest's val_ap50_95 field for "
                              "consistency with the v1 YOLOX manifest")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def derive_heatmap_stride(output_shape, *, frames_per_input, input_size,
                           cli_heatmap_stride=None):
    """Measure heatmap_stride from a traced model's real output shape.

    `output_shape` is the `[N, F, Hh, Wh]` shape of the traced module's
    output on the `(1, 3*frames_per_input, input_size, input_size)` example
    input. WASB-SBDT's default config yields stride 1 (Hh == input_size), not
    the conventional 2 or 4 -- so this never assumes a value, it only reads
    the shape that came back and dies loudly, naming the actual shape, if it
    doesn't fit the expected [N, F, Hh, Hh] pattern.

    If `cli_heatmap_stride` (the operator's `--heatmap-stride`) is given, the
    measured value must match it exactly, or this raises naming both.
    """
    if len(output_shape) != 4:
        raise ValueError(
            f"expected a rank-4 traced output shape [N, frames, Hh, Wh], got "
            f"{tuple(output_shape)} (rank {len(output_shape)})")

    _n, frames, hh, wh = output_shape
    if frames != frames_per_input:
        raise ValueError(
            f"traced output has {frames} frame channels but "
            f"--frames-per-input={frames_per_input}; full output shape was "
            f"{tuple(output_shape)}. The traced graph does not match the "
            f"requested temporal window -- refusing to guess a mapping.")

    if hh != wh:
        raise ValueError(
            f"traced heatmap output is not square: {hh}x{wh} (full shape "
            f"{tuple(output_shape)}); the tiled-sweep detector and "
            f"ball_model.py's heatmap-peak decode both assume a square "
            f"heatmap grid.")

    if hh <= 0 or input_size % hh != 0:
        raise ValueError(
            f"input_size={input_size} is not evenly divisible by the traced "
            f"heatmap dimension {hh} (full output shape {tuple(output_shape)}); "
            f"heatmap_stride cannot be derived from a non-integer ratio.")

    measured = input_size // hh

    if cli_heatmap_stride is not None and cli_heatmap_stride != measured:
        raise ValueError(
            f"heatmap_stride mismatch: measured {measured} from the traced "
            f"output shape {tuple(output_shape)}, but --heatmap-stride="
            f"{cli_heatmap_stride} was given. The manifest must reflect what "
            f"the traced graph actually does, not an assumption -- fix "
            f"whichever one is wrong before exporting.")

    return measured


def build_manifest(*, name, version, input_size, frames_per_input,
                    heatmap_stride, nominal_ball_px, conf_threshold, nms_iou,
                    tile_overlap_px, max_batch_tiles, artifact_sha256,
                    source_checkpoint, trained_commit, val_metric, notes):
    """Pure assembly of the ball-model-v2 manifest dict -- no I/O, no torch.

    Emits every field ball_model.load_manifest's v2 branch requires (Task 1),
    not just the WASB-specific delta -- the loader raises KeyError on any
    field missing regardless of schema version.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "version": version,
        "input_size": [input_size, input_size],
        "decode": "heatmap_peak",
        "frames_per_input": frames_per_input,
        "heatmap_stride": heatmap_stride,
        "nominal_ball_px": nominal_ball_px,
        "conf_threshold": conf_threshold,
        "nms_iou": nms_iou,
        "class_names": ["ball"],
        "tile_overlap_px": tile_overlap_px,
        "max_batch_tiles": max_batch_tiles,
        "artifact_sha256": artifact_sha256,
        "source_checkpoint": source_checkpoint,
        "trained_commit": trained_commit,
        "val_ap50_95": val_metric,
        "notes": notes,
    }


def main():
    args = parse_args()
    import sys

    import torch
    import yaml

    sys.path.insert(0, str(Path(args.wasb_repo) / "src"))
    from models.hrnet import HRNet  # WASB-SBDT, training-env only

    if args.frames_per_input % 2 == 0:
        raise ValueError(
            f"--frames-per-input must be odd (heatmap_peak decode centers "
            f"the detected frame in the window), got {args.frames_per_input}")

    cfg = yaml.safe_load(Path(args.wasb_cfg).read_text(encoding="utf-8"))
    model = HRNet(cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    # docs/WASB-TRAIN.md §1: verified checkpoint format is a plain
    # torch.save(state, path) dict keyed 'model_state_dict'; fall back to
    # treating the loaded object itself as the state dict for a checkpoint
    # saved without that wrapper.
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    size = args.input_size
    channels = 3 * args.frames_per_input
    example = torch.randn(1, channels, size, size)
    traced = torch.jit.trace(model, example)

    with torch.no_grad():
        output = traced(example)
    output_shape = tuple(output.shape)
    heatmap_stride = derive_heatmap_stride(
        output_shape, frames_per_input=args.frames_per_input,
        input_size=size, cli_heatmap_stride=args.heatmap_stride)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "model.torchscript"
    traced.save(str(artifact))

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = build_manifest(
        name=args.name,
        version=args.version,
        input_size=size,
        frames_per_input=args.frames_per_input,
        heatmap_stride=heatmap_stride,
        nominal_ball_px=args.nominal_ball_px,
        conf_threshold=args.conf_threshold,
        nms_iou=args.nms_iou,
        tile_overlap_px=args.tile_overlap_px,
        max_batch_tiles=args.max_batch_tiles,
        artifact_sha256=digest,
        source_checkpoint=str(Path(args.ckpt).name),
        trained_commit=_git_commit(),
        val_metric=args.val_metric,
        notes=args.notes,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {artifact} ({artifact.stat().st_size} bytes)")
    print(f"sha256 {digest}")
    print(f"measured heatmap_stride={heatmap_stride} from output shape {output_shape}")


if __name__ == "__main__":
    main()
