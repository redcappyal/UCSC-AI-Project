"""Measure WASB tiled-sweep throughput on this machine's compute device.

The hosting cost model turns entirely on one number: milliseconds per 416px
tile. Everything else (tiles per frame, frames per session) is arithmetic from
the manifest. Run this on a CUDA host to replace the estimate with a
measurement:

    .venv/bin/python bench_wasb_device.py
    .venv/bin/python bench_wasb_device.py --batch-sizes 8,16,32,64

Timing notes that matter on GPUs: CUDA and MPS launch kernels asynchronously,
so a timer stopped without an explicit synchronize measures launch overhead
rather than compute. Every timed region below syncs before stopping, and every
batch size gets warmup iterations first so lazy init and kernel autotune land
outside the measurement.
"""

import argparse
import statistics
import time

import numpy as np

import ball_model
from ball_detector import tile_windows

# Sessions the cost projection assumes, and the shapes they come in. The 1080p
# entry is the demo asset; the 4K60 entry is what ios/ capture settings record.
PROFILES = {
    "1080p30": (1920, 1080, 30.0, 331.6),
    "4K60": (3840, 2160, 60.0, 300.0),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,8,16,32",
                        help="Comma-separated tile batch sizes to sweep.")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Untimed iterations before each measurement.")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timed iterations per batch size.")
    parser.add_argument("--frame-stride", type=int, default=4,
                        help="Pipeline frame stride the projection assumes.")
    parser.add_argument("--sessions-per-day", type=int, default=100)
    parser.add_argument("--gpu-cost-per-hour", type=float, default=1.29,
                        help="USD/hr for the projection (Lambda A10 = 1.29).")
    return parser.parse_args()


def describe_device(torch, device):
    """One line naming what we are actually timing on."""
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / (1024 ** 3)
        return f"cuda: {props.name}, {vram:.0f} GB VRAM, torch {torch.__version__}"
    if device == "mps":
        return f"mps: Apple GPU, torch {torch.__version__}"
    return f"cpu: torch {torch.__version__}"


def sync(torch, device):
    """Block until queued GPU work is done, so perf_counter means something."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def make_stacks(batch, tile, channels, rng):
    """A batch of tile stacks shaped like the real serving path.

    Random uint8 rather than real frames: the convolution cost is independent
    of pixel values, and this keeps the benchmark runnable on a bare box with
    no video asset checked out.
    """
    return [rng.integers(0, 256, size=(tile, tile, channels), dtype=np.uint8)
            for _ in range(batch)]


def time_batch(runner, torch, device, stacks, warmup, iters):
    """Median and best seconds for one run_batch call at this batch size."""
    for _ in range(warmup):
        runner.run_batch(stacks)
    sync(torch, device)

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        runner.run_batch(stacks)
        sync(torch, device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples)


def tiles_per_frame(width, height, tile, overlap):
    return len(tile_windows(width, height, tile, overlap))


def project(ms_per_tile, tile, overlap, frame_stride, sessions_per_day, cost_per_hour):
    """Per-session GPU time and monthly spend at the given throughput."""
    rows = []
    for name, (width, height, fps, duration) in PROFILES.items():
        per_frame = tiles_per_frame(width, height, tile, overlap)
        frames = int(duration * fps) // frame_stride
        tiles = per_frame * frames
        session_s = tiles * ms_per_tile / 1000.0
        gpu_hours_day = session_s * sessions_per_day / 3600.0
        rows.append({
            "profile": name,
            "tiles_per_frame": per_frame,
            "frames": frames,
            "tiles": tiles,
            "session_min": session_s / 60.0,
            "gpu_hours_day": gpu_hours_day,
            "usd_month": gpu_hours_day * 30 * cost_per_hour,
        })
    return rows


def main():
    args = parse_args()
    import torch

    manifest = ball_model.load_manifest()
    runner = ball_model.load_detector()
    device = getattr(runner, "device", "cpu")
    tile = int(manifest.input_size[0])
    channels = 3 * int(manifest.frames_per_input)
    overlap = int(manifest.tile_overlap_px)

    print(describe_device(torch, device))
    print(f"model: {manifest.name} v{manifest.version}  tile {tile}px  "
          f"overlap {overlap}px  frames_per_input {manifest.frames_per_input} "
          f"({channels}ch)  manifest max_batch_tiles {manifest.max_batch_tiles}")
    print()

    rng = np.random.default_rng(0)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    print(f"{'batch':>6}  {'median ms':>10}  {'ms/tile':>9}  {'tiles/sec':>10}")
    best_ms_per_tile = None
    for batch in batch_sizes:
        stacks = make_stacks(batch, tile, channels, rng)
        try:
            median_s, best_s = time_batch(
                runner, torch, device, stacks, args.warmup, args.iters)
        except RuntimeError as exc:                      # OOM at large batch
            print(f"{batch:>6}  FAILED: {str(exc).splitlines()[0][:60]}")
            continue
        ms_per_tile = median_s * 1000.0 / batch
        print(f"{batch:>6}  {median_s * 1000:>10.1f}  {ms_per_tile:>9.2f}  "
              f"{batch / median_s:>10.1f}")
        if best_ms_per_tile is None or ms_per_tile < best_ms_per_tile:
            best_ms_per_tile = ms_per_tile

    if best_ms_per_tile is None:
        print("\nNo batch size completed; nothing to project.")
        return

    print(f"\nbest measured: {best_ms_per_tile:.2f} ms/tile")
    print(f"\nprojection at stride {args.frame_stride}, "
          f"{args.sessions_per_day} sessions/day, ${args.gpu_cost_per_hour:.2f}/GPU-hr:")
    print(f"{'profile':>9}  {'tiles/frm':>9}  {'tiles/session':>13}  "
          f"{'min/session':>11}  {'GPU-hr/day':>10}  {'$/month':>9}")
    for row in project(best_ms_per_tile, tile, overlap, args.frame_stride,
                       args.sessions_per_day, args.gpu_cost_per_hour):
        print(f"{row['profile']:>9}  {row['tiles_per_frame']:>9}  "
              f"{row['tiles']:>13,}  {row['session_min']:>11.1f}  "
              f"{row['gpu_hours_day']:>10.1f}  {row['usd_month']:>9,.0f}")


if __name__ == "__main__":
    main()
