"""CLI: run a YOLO checkpoint over a video and write ball coordinates.

Writes the same CSV schema as the production tracker (tracking_common), so the
existing eval/label tooling scores YOLO checkpoints with zero changes. This is
the acceptance gate for the on-device model: compare its detection rate and
eval numbers against the RF-DETR CSVs before shipping the Core ML export.

ultralytics is imported lazily inside main(); the adapter stays importable in
the requirements-test.txt environment.
"""

import argparse
import csv
from pathlib import Path

import cv2

from tracking_common import (
    CONFIDENCE_THRESHOLD,
    CSV_FIELDNAMES,
    ball_csv_row,
    draw_predictions,
    select_ball_prediction,
)


def yolo_boxes_to_predictions(rows, names):
    """ultralytics Boxes.data rows -> prediction dicts for tracking_common.

    rows: iterable of (x1, y1, x2, y2, confidence, class_index); tensors or
    plain sequences both work. names: {class_index: class_name}.
    """
    predictions = []
    for x1, y1, x2, y2, confidence, class_index in rows:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        predictions.append({
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1,
            "confidence": float(confidence),
            "class": names.get(int(class_index), str(int(class_index))),
        })
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="YOLO .pt checkpoint")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--annotated", default=None,
                        help="Optional annotated .mp4 output path")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO  # lazy: heavy, and absent in the test env

    model = YOLO(args.weights)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.annotated:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(args.annotated), cv2.VideoWriter_fourcc(*"mp4v"),
            fps / max(1, args.stride), (width, height))

    detected = 0
    written = 0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        csv_writer.writeheader()
        frame_index = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % args.stride:
                continue
            if args.max_frames is not None and written >= args.max_frames:
                break

            result = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            rows = result.boxes.data.tolist() if result.boxes is not None else []
            predictions = yolo_boxes_to_predictions(rows, result.names)
            ball = select_ball_prediction(predictions)
            csv_writer.writerow(ball_csv_row(frame_index, fps, ball))
            written += 1
            if ball is not None:
                detected += 1
            if writer is not None:
                # draw_predictions returns an annotated copy (no mutation).
                writer.write(draw_predictions(frame, [ball] if ball else []))

    capture.release()
    if writer is not None:
        writer.release()
    rate = detected / written * 100 if written else 0.0
    print(f"frames={written} detected={detected} rate={rate:.1f}%")
    print(f"csv={args.output_csv}")


if __name__ == "__main__":
    main()
