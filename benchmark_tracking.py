"""Benchmark the ball-tracking inference hot path.

Run this before and after each optimization phase, on the same machine and
clip, to measure real speedups:

    .venv/bin/python benchmark_tracking.py
    .venv/bin/python benchmark_tracking.py --providers cpu
    .venv/bin/python benchmark_tracking.py --providers coreml
"""

import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROVIDER_PRESETS = {
    "default": None,
    "cpu": "[CPUExecutionProvider]",
    "coreml": "[CoreMLExecutionProvider,CPUExecutionProvider]",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "SquashAnalytics.mp4")
    parser.add_argument("--start-frame", type=int, default=300)
    parser.add_argument("--frames", type=int, default=120, help="Frames to benchmark per variant.")
    parser.add_argument("--inference-width", type=int, default=960)
    parser.add_argument("--batch-sizes", default="4,8")
    parser.add_argument(
        "--providers",
        choices=sorted(PROVIDER_PRESETS),
        default="default",
        help="ONNX execution provider preset (must be chosen before the model loads).",
    )
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def setup_environment(providers):
    os.environ.setdefault("MODEL_CACHE_DIR", str(ROOT / ".roboflow-cache"))
    os.environ.setdefault("METRICS_ENABLED", "False")
    os.environ.setdefault("OTEL_METRICS_ENABLED", "False")
    preset = PROVIDER_PRESETS[providers]
    if preset is not None:
        os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = preset

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(ROOT / ".env")


def load_frames(video_path, start_frame, count):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    decode_start = time.perf_counter()
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    decode_elapsed = time.perf_counter() - decode_start
    cap.release()

    if not frames:
        raise RuntimeError("No frames decoded; check --video and --start-frame.")
    return frames, decode_elapsed


def time_variant(name, frames, warmup, run_one):
    try:
        for frame in frames[: max(1, warmup)]:
            run_one(frame)

        start = time.perf_counter()
        for frame in frames:
            run_one(frame)
        elapsed = time.perf_counter() - start
    except Exception as error:
        return {"name": name, "frames": len(frames), "seconds": None, "error": error}
    return {"name": name, "frames": len(frames), "seconds": elapsed}


def time_batched(name, frames, warmup, batch_size, run_batch):
    try:
        run_batch(frames[: max(1, min(warmup, batch_size))])

        start = time.perf_counter()
        for i in range(0, len(frames), batch_size):
            run_batch(frames[i : i + batch_size])
        elapsed = time.perf_counter() - start
    except Exception as error:
        return {"name": name, "frames": len(frames), "seconds": None, "error": error}
    return {"name": name, "frames": len(frames), "seconds": elapsed}


def print_results(results):
    print()
    header = f"{'variant':<38} {'frames':>6} {'total s':>9} {'fps':>8} {'ms/frame':>9}"
    print(header)
    print("-" * len(header))
    for result in results:
        if result["seconds"] is None:
            error_text = str(result.get("error", "failed")).splitlines()[0][:60]
            print(f"{result['name']:<38} FAILED: {error_text}")
            continue
        fps = result["frames"] / result["seconds"] if result["seconds"] else float("inf")
        ms = result["seconds"] / result["frames"] * 1000 if result["frames"] else 0.0
        print(f"{result['name']:<38} {result['frames']:>6} {result['seconds']:>9.2f} {fps:>8.2f} {ms:>9.1f}")


def main():
    args = parse_args()
    setup_environment(args.providers)

    import cv2
    from PIL import Image

    from inference_engine import DEFAULT_MODEL_ID, resize_frame_for_inference
    from tracking_common import CONFIDENCE_THRESHOLD

    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("Set ROBOFLOW_API_KEY in .env before benchmarking.")

    from inference import get_model

    model_id = os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID)
    print(f"Loading model {model_id} (providers preset: {args.providers})")
    load_start = time.perf_counter()
    model = get_model(model_id=model_id, api_key=api_key, countinference=False)
    print(f"Model loaded in {time.perf_counter() - load_start:.1f}s")

    # inference >= 1.3 wraps the real model in an adapter; introspect both.
    candidates = [model, getattr(model, "_model", None)]
    for target in [c for c in candidates if c is not None]:
        for attr in ("img_size_w", "img_size_h", "batching_enabled", "batch_size", "input_shape"):
            value = getattr(target, attr, None)
            if value is not None:
                print(f"  {type(target).__name__}.{attr}: {value}")
        session = getattr(target, "onnx_session", None) or getattr(target, "_session", None)
        if session is not None:
            print(f"  active providers: {session.get_providers()}")

    frames, decode_elapsed = load_frames(args.video, args.start_frame, args.frames)
    height, width = frames[0].shape[:2]
    print(f"\nBenchmarking {len(frames)} frames of {width}x{height} at inference width {args.inference_width}")

    resized = [resize_frame_for_inference(frame, args.inference_width)[0] for frame in frames]

    results = [{"name": "decode only", "frames": len(frames), "seconds": decode_elapsed}]

    results.append(
        time_variant(
            "decode+resize (resize only)",
            frames,
            args.warmup,
            lambda frame: resize_frame_for_inference(frame, args.inference_width),
        )
    )

    def pil_path(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return model.infer(image=image, confidence=CONFIDENCE_THRESHOLD)

    results.append(time_variant("PIL round-trip (current path)", resized, args.warmup, pil_path))

    def numpy_path(frame):
        return model.infer(frame, confidence=CONFIDENCE_THRESHOLD)

    results.append(time_variant("numpy BGR direct", resized, args.warmup, numpy_path))

    for batch_size in (int(b) for b in args.batch_sizes.split(",") if b.strip()):
        results.append(
            time_batched(
                f"numpy batched (batch={batch_size})",
                resized,
                args.warmup,
                batch_size,
                lambda batch: model.infer(batch, confidence=CONFIDENCE_THRESHOLD),
            )
        )

    print_results(results)


if __name__ == "__main__":
    main()
