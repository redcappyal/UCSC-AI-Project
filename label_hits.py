import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2


DEFAULT_VIDEO_PATH = Path(__file__).with_name("Bay Club Squash 5min+audio.mp4")
DEFAULT_LABELS_PATH = Path(__file__).with_name("wall_hits.csv")
WINDOW_NAME = "Squash Wall Hit Labeler"
SIDECAR_SCHEMA = "label-run-v1"


def video_sha256(video_path):
    """Same identity the upload endpoint assigns: sha256 of the whole file.
    Lets build_eval_set.py tie these labels to a tracking run of the same
    video without depending on filenames, which drift."""
    hasher = hashlib.sha256()
    with Path(video_path).open("rb") as video_file:
        for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sidecar_path(labels_path):
    return Path(labels_path).with_suffix(".meta.json")


def save_sidecar(labels_path, video_path, video_sha, fps, frame_count, labels):
    """A label CSV is only frame numbers; on its own it cannot say which video
    those frames index into. The sidecar carries that identity so the labels
    survive as evaluation data instead of dying as a loose column."""
    sidecar_path(labels_path).write_text(
        json.dumps(
            {
                "schema_version": SIDECAR_SCHEMA,
                "video_path": str(Path(video_path).resolve()),
                "video_sha": video_sha,
                "fps": fps,
                "frame_count": frame_count,
                "label_count": len(labels),
                "labeled_min_frame": min(labels) if labels else None,
                "labeled_max_frame": max(labels) if labels else None,
                "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_labels(labels_path):
    if not labels_path.exists():
        return set()

    labels = set()
    with labels_path.open(newline="") as labels_file:
        reader = csv.reader(labels_file)
        first_row = next(reader, None)
        if not first_row:
            return labels

        frame_column_names = ("hit_frame", "source_frame", "frame")
        normalized_first_row = [value.strip() for value in first_row]
        frame_column_index = next(
            (
                index
                for index, column in enumerate(normalized_first_row)
                if column in frame_column_names
            ),
            None,
        )

        if frame_column_index is None:
            rows = [first_row, *reader]
            frame_column_index = 0
        else:
            rows = reader

        for row in rows:
            if len(row) <= frame_column_index:
                continue
            value = row[frame_column_index].strip()
            if value:
                labels.add(int(value))

    return labels


def save_labels(labels_path, labels):
    with labels_path.open("w", newline="") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=["hit_frame"])
        writer.writeheader()
        for frame in sorted(labels):
            writer.writerow({"hit_frame": frame})


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}.")
    return frame


def scale_frame(frame, max_width, max_height):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale == 1.0:
        return frame

    return cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def draw_text_box(frame, lines, origin=(16, 28)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    line_height = 26
    padding = 10

    widths = [
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    ]
    box_width = max(widths) + padding * 2
    box_height = len(lines) * line_height + padding
    x, y = origin

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - 22),
        (x - padding + box_width, y - 22 + box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def render_frame(frame, frame_index, frame_count, fps, labels, playing):
    display = frame.copy()
    timestamp_seconds = frame_index / fps if fps else 0.0
    status = "HIT MARKED" if frame_index in labels else "not marked"
    play_status = "PLAYING" if playing else "PAUSED"

    lines = [
        f"Frame {frame_index} / {frame_count - 1}   Time {timestamp_seconds:.3f}s",
        f"{status}   Labels: {len(labels)}   {play_status}",
        "h mark/unmark | arrows/a/d prev/next | [/]/</> jump | g goto | space play | s save | q quit",
    ]
    draw_text_box(display, lines)

    if frame_index in labels:
        cv2.rectangle(display, (8, 8), (display.shape[1] - 8, display.shape[0] - 8), (0, 255, 0), 5)

    return display


def prompt_for_frame(current_frame, minimum_frame, maximum_frame):
    raw_value = input(f"Jump to frame [current {current_frame}]: ").strip()
    if not raw_value:
        return current_frame

    try:
        return clamp(int(raw_value), minimum_frame, maximum_frame)
    except ValueError:
        print("Invalid frame number; staying on the current frame.")
        return current_frame


def find_next_label(labels, current_frame, direction):
    if not labels:
        return current_frame

    ordered = sorted(labels)
    if direction > 0:
        for frame in ordered:
            if frame > current_frame:
                return frame
        return ordered[0]

    for frame in reversed(ordered):
        if frame < current_frame:
            return frame
    return ordered[-1]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Frame-by-frame video labeler for squash wall-hit frames."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=f"Video to label. Defaults to {DEFAULT_VIDEO_PATH.name}.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"Output label CSV. Defaults to {DEFAULT_LABELS_PATH.name}.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame to show first.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to label. Defaults to the end of the video.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Maximum display width.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Maximum display height.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    cap = cv2.VideoCapture(str(args.video))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    minimum_frame = clamp(args.start_frame, 0, frame_count - 1)
    maximum_frame = frame_count - 1
    if args.end_frame is not None:
        maximum_frame = clamp(args.end_frame, minimum_frame, frame_count - 1)

    labels = load_labels(args.labels)
    frame_index = minimum_frame
    playing = False
    dirty = False

    print(f"Hashing {args.video.name} for label identity...", flush=True)
    video_sha = video_sha256(args.video)

    def persist():
        save_labels(args.labels, labels)
        save_sidecar(args.labels, args.video, video_sha, fps, frame_count, labels)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("Controls:")
    print("  h: mark/unmark current frame as a wall hit")
    print("  Left/a: previous frame")
    print("  Right/d: next frame")
    print("  [: jump back 10 frames    ]: jump forward 10 frames")
    print("  <: jump back 100 frames   >: jump forward 100 frames")
    print("  space: play/pause")
    print("  g: type a frame number in the terminal and jump there")
    print("  n/N: jump to next/previous labeled hit")
    print("  s: save labels")
    print("  q or Esc: save and quit")

    try:
        while True:
            frame = read_frame(cap, frame_index)
            display = scale_frame(frame, args.max_width, args.max_height)
            display = render_frame(display, frame_index, frame_count, fps, labels, playing)
            cv2.imshow(WINDOW_NAME, display)

            delay_ms = max(1, int(1000 / fps)) if playing else 0
            key = cv2.waitKeyEx(delay_ms)

            if playing and key == -1:
                frame_index = clamp(frame_index + 1, minimum_frame, maximum_frame)
                if frame_index == maximum_frame:
                    playing = False
                continue

            if key in {-1, 255}:
                continue

            raw_key_char = chr(key & 0xFF) if 0 <= (key & 0xFF) < 128 else ""
            key_char = raw_key_char.lower()

            if key in {27} or key_char == "q":
                break

            if key_char == " ":
                playing = not playing
            elif key_char == "h":
                if frame_index in labels:
                    labels.remove(frame_index)
                    print(f"Removed hit label at frame {frame_index}")
                else:
                    labels.add(frame_index)
                    print(f"Added hit label at frame {frame_index}")
                dirty = True
            elif key_char == "s":
                persist()
                dirty = False
                print(f"Saved {len(labels)} label(s) to {args.labels}")
            elif key_char == "g":
                playing = False
                frame_index = prompt_for_frame(frame_index, minimum_frame, maximum_frame)
            elif raw_key_char == "n":
                frame_index = find_next_label(labels, frame_index, 1)
            elif raw_key_char == "N":
                frame_index = find_next_label(labels, frame_index, -1)
            elif key_char in {"d", "."} or key in {83, 65363, 63235, 2555904}:
                frame_index = clamp(frame_index + 1, minimum_frame, maximum_frame)
            elif key_char in {"a", ","} or key in {81, 65361, 63234, 2424832}:
                frame_index = clamp(frame_index - 1, minimum_frame, maximum_frame)
            elif key_char == "]":
                frame_index = clamp(frame_index + 10, minimum_frame, maximum_frame)
            elif key_char == "[":
                frame_index = clamp(frame_index - 10, minimum_frame, maximum_frame)
            elif key_char == ">":
                frame_index = clamp(frame_index + 100, minimum_frame, maximum_frame)
            elif key_char == "<":
                frame_index = clamp(frame_index - 100, minimum_frame, maximum_frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if dirty:
        persist()
        print(f"Saved {len(labels)} label(s) to {args.labels}")


if __name__ == "__main__":
    main()
