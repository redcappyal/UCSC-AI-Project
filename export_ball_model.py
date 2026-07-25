"""Trace the trained YOLOX ball checkpoint to TorchScript + manifest.

Run this in the TRAINING environment (ball-detector-train/.venv), never in the
app env. That is the whole point: yolox_ball_exp.py's docstring requires YOLOX
stay a training dependency, so the serving path gets a self-contained traced
artifact and never imports YOLOX.

    python export_ball_model.py \
        --exp yolox_ball_exp.py \
        --ckpt .../YOLOX_outputs/crosscourt-ball-416/best_ckpt.pth \
        --out models/crosscourt-ball-416-v1 \
        --version 1
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SCHEMA_VERSION = "ball-model-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="path to yolox_ball_exp.py")
    parser.add_argument("--ckpt", required=True, help="path to best_ckpt.pth")
    parser.add_argument("--out", required=True, help="output model directory")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--name", default="crosscourt-ball-416")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--tile-overlap-px", type=int, default=64)
    parser.add_argument("--max-batch-tiles", type=int, default=32)
    parser.add_argument("--val-ap50-95", type=float, default=0.0)
    parser.add_argument("--notes", default="val is diagnostic only -- shares a "
                                           "rig/session with train")
    return parser.parse_args()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    import torch
    from yolox.exp import get_exp

    exp = get_exp(args.exp, None)
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Decode inside the graph: the server then only thresholds and NMSes, so
    # the grid math lives in one place. The Core ML export (ios/MODEL.md §4)
    # needs the opposite -- raw outputs, decode in Swift for ANE residency.
    model.head.decode_in_inference = True

    size = exp.test_size[0]
    example = torch.randn(1, 3, size, size)
    traced = torch.jit.trace(model, example)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "model.torchscript"
    traced.save(str(artifact))

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": args.name,
        "version": args.version,
        "input_size": [size, size],
        "decode": "in_graph",
        "conf_threshold": args.conf_threshold,
        "nms_iou": float(exp.nmsthre),
        "class_names": ["ball"],
        "tile_overlap_px": args.tile_overlap_px,
        "max_batch_tiles": args.max_batch_tiles,
        "artifact_sha256": digest,
        "source_checkpoint": str(Path(args.ckpt).name),
        "trained_commit": _git_commit(),
        "val_ap50_95": args.val_ap50_95,
        "notes": args.notes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {artifact} ({artifact.stat().st_size} bytes)")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
