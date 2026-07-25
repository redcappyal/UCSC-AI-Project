"""CLI: turn a raw Roboflow COCO export into YOLOX-ready ball crops.

The phone model detects a ball that is 4-25 px wide in a 4K frame. Resizing a
whole frame down to the network's input (what `inference_engine.py` does today
at 960 px wide) throws away most of the evidence before the model sees it, so
this builds the training set the other way round: fixed-size windows cut at
*native* resolution around each annotation, which keeps every pixel of the ball
and makes the input tensor smaller rather than larger.

What it does, and why each step exists:

- **Drops low-resolution source clips.** Below ~960 px wide the ball is 3-4 px;
  those frames teach the model to fire on evidence that is not there.
- **Converts polygons to axis-aligned boxes.** Annotations are polygons; the
  detector head regresses an AABB. The polygon's major axis is preserved in a
  sidecar (`streak`) because for a motion-streaked ball its two endpoints are
  where the ball was at the *start* and *end* of the exposure — sub-frame
  timing the box discards.
- **Splits by source clip, never by frame.** Consecutive frames of one rally
  are near-duplicates, so a frame-level split leaks train into val and reports
  a score that has nothing to do with generalisation.
- **Reports burst structure.** Densely-labelled runs of consecutive frames
  carry far less information than their count suggests; the manifest prints
  both so the redundancy is visible rather than assumed.
- **Emits hard negatives**, including windows cut from ball-bearing frames.
  Squash walls are covered in ball marks — dark smudges the size and colour of
  the ball — and without negatives containing them the model fires on every one.

Usage: python prepare_ball_dataset.py --source "~/Desktop/Annotated Data/SquashAI.coco"
"""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = "ball-crops-v1"
CLASS_NAME = "ball"

# Roboflow names frames "<clip>_mov-0042_jpg.rf.<hash>.jpg"; the clip prefix is
# the split unit, and the frame number is what reveals burst structure.
FRAME_RE = re.compile(r"^(?P<clip>.+?)[-_](?:mov|mp4)-(?P<frame>\d+)", re.IGNORECASE)

# Below this the ball is 3-4 px even before the network resizes anything.
MIN_SOURCE_WIDTH = 960
# Per-clip rescaling is a dataset-level normalisation, not a per-frame one, so
# it can't peek at an individual label. Still clamped: a clip needing more than
# this to reach the target is too far out of domain to rescale into it.
SCALE_LIMITS = (0.5, 4.0)

# Anything outside this set breaks a crop filename somewhere in the chain: cv2
# on Windows, or a zip round-trip that loses the UTF-8 flag. See
# docs/superpowers/specs/2026-07-24-ascii-crop-filenames-design.md.
UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def clip_and_frame(file_name):
    """('BayClub', 42) for a Roboflow frame name; (stem, None) if unparseable."""
    match = FRAME_RE.match(file_name)
    if not match:
        return Path(file_name).stem.split(".rf.")[0], None
    return match.group("clip"), int(match.group("frame"))


def ascii_slug(stem):
    """ASCII, filesystem-safe form of a source frame stem.

    One source clip is a YouTube title carrying U+FF5C (｜, a sanitised "|"), and
    a filename built from it is unusable on Windows twice over: cv2.imread
    returns None for it, and cv2.imwrite reports success while writing a
    mojibake name. Emitting ASCII is what makes the crop names portable.

    NFKD runs first so accents transliterate (é -> e) instead of vanishing, but
    it also turns U+FF5C into a literal "|" — valid ASCII, invalid in a Windows
    filename — so the charset filter runs after it, never instead of it.

    The transform is lossy, so a name that changed carries an 8-hex digest of
    the original: two clips that collapse onto one base stay distinct, and the
    suffix is stable per source name. Only a name already in canonical form —
    safe charset *and* no leading or trailing "._-" — is returned untouched;
    a name that the charset filter leaves alone but the strip still shortens
    earns the digest too, because stripping is itself lossy: Win32 silently
    discards trailing dots, so "abc." and "abc" are the same file on Windows,
    and a leading dot hides a file on Unix. Collapsing either onto the bare
    form unchanged would let two source frames silently overwrite each
    other's crops, so the digest stays with every lossy transform, not just
    the charset one.
    """
    folded = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    slug = UNSAFE_CHARS.sub("_", folded).strip("._-") or "clip"
    if slug == stem:
        return slug
    return f"{slug}-{hashlib.sha1(stem.encode('utf-8')).hexdigest()[:8]}"


def polygon_points(segmentation):
    """COCO segmentation -> [(x, y), ...]. Roboflow emits one flat ring."""
    if not segmentation:
        return []
    flat = segmentation[0] if isinstance(segmentation[0], (list, tuple)) else segmentation
    return list(zip(flat[0::2], flat[1::2]))


def polygon_aabb(points):
    """Axis-aligned (x, y, w, h) enclosing a polygon."""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def streak_metrics(points):
    """Major axis of an annotation: the two most distant vertices.

    For a ball frozen by a fast shutter this is just the diameter and the angle
    is meaningless. For a streak it is the exposure path, and `endpoints` are
    the ball's position at either end of the shutter window — two timestamped
    positions per frame, where a round ball gives only one at an unknown
    instant. Consumed by the tracker and judge_call.py, never by the box head.
    """
    if len(points) < 2:
        return None
    best, best_d2 = None, -1.0
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d2 > best_d2:
                best, best_d2 = (a, b), d2
    (ax, ay), (bx, by) = best
    length = math.sqrt(best_d2)
    if length <= 0:
        return None
    # Minor axis: twice the largest perpendicular offset from the major axis.
    ux, uy = (bx - ax) / length, (by - ay) / length
    width = 2 * max(abs(-(py - ay) * ux + (px - ax) * uy) for px, py in points)
    return {
        "endpoints": [[ax, ay], [bx, by]],
        "length": length,
        "width": width,
        "angle_deg": math.degrees(math.atan2(by - ay, bx - ax)),
        "aspect": length / width if width > 0 else 1.0,
    }


def load_export(source):
    """Every annotated frame in a Roboflow COCO export, split labels discarded.

    The export's own train/valid/test split is rebuilt here: Roboflow splits by
    frame, which puts near-identical neighbours from one rally on both sides.
    """
    records, categories = [], set()
    for split_dir in sorted(p for p in Path(source).iterdir() if p.is_dir()):
        for json_path in sorted(split_dir.glob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            categories.update(c["name"] for c in data.get("categories", []))
            by_image = defaultdict(list)
            for ann in data.get("annotations", []):
                by_image[ann["image_id"]].append(ann)
            for image in data.get("images", []):
                clip, frame = clip_and_frame(image["file_name"])
                balls = []
                for ann in by_image.get(image["id"], []):
                    points = polygon_points(ann.get("segmentation"))
                    if points:
                        box = polygon_aabb(points)
                        streak = streak_metrics(points)
                    elif ann.get("bbox"):
                        box, streak = tuple(ann["bbox"]), None
                    else:
                        continue
                    if box[2] <= 0 or box[3] <= 0:
                        continue
                    balls.append({"bbox": list(box), "streak": streak})
                records.append({
                    "path": split_dir / image["file_name"],
                    "clip": clip,
                    "frame": frame,
                    "width": image["width"],
                    "height": image["height"],
                    "balls": balls,
                })
    records.sort(key=lambda r: (r["clip"], r["frame"] if r["frame"] is not None else -1,
                                r["path"].name))
    return records, sorted(categories)


def burst_count(records, gap=3):
    """Temporally separated groups of frames, as opposed to raw frame count.

    Labelling runs of consecutive frames is the common failure: at 60 fps
    neighbours are ~95% identical, so 300 such labels are worth closer to 10.
    """
    per_clip = defaultdict(list)
    for record in records:
        if record["frame"] is not None:
            per_clip[record["clip"]].append(record["frame"])
    total = 0
    for frames in per_clip.values():
        frames = sorted(set(frames))
        total += 1 + sum(1 for a, b in zip(frames, frames[1:]) if b - a > gap)
    return total


def thin_bursts(records, min_gap):
    """Keep at most one frame per `min_gap` within each clip.

    Redundant neighbours cost training time and bias the loss toward whichever
    rally happened to be labelled densely, without adding information.
    """
    if min_gap <= 1:
        return records, 0
    kept, last = [], {}
    for record in records:
        if record["frame"] is None:
            kept.append(record)
            continue
        previous = last.get(record["clip"])
        if previous is None or record["frame"] - previous >= min_gap:
            kept.append(record)
            last[record["clip"]] = record["frame"]
    return kept, len(records) - len(kept)


def split_by_clip(records, val_clips, test_clips):
    """Assign whole clips to splits; a clip never spans two of them.

    Clips not named explicitly go to train, so adding footage widens training
    without silently diluting a frozen eval set.
    """
    val, test = set(val_clips or ()), set(test_clips or ())
    overlap = val & test
    if overlap:
        raise SystemExit(f"clip(s) in both --val-clips and --test-clips: {sorted(overlap)}")
    splits = defaultdict(list)
    for record in records:
        name = "val" if record["clip"] in val else "test" if record["clip"] in test else "train"
        splits[name].append(record)
    return splits


def clip_scale_factors(records, target_ball_px):
    """Per-clip resize factor bringing the median ball to `target_ball_px`.

    Training footage is 1080p while the rig shoots 4K, so the same ball spans
    different pixel counts in each. Normalising by the clip's median annotation
    (never an individual one) puts the model's input distribution where
    deployment will be. 0 disables it and crops stay at native scale.
    """
    if not target_ball_px:
        return {}
    widths = defaultdict(list)
    for record in records:
        for ball in record["balls"]:
            widths[record["clip"]].append(ball["bbox"][2])
    factors = {}
    for clip, values in widths.items():
        median = statistics.median(values)
        if median > 0:
            factors[clip] = min(max(target_ball_px / median, SCALE_LIMITS[0]), SCALE_LIMITS[1])
    return factors


def _window(cx, cy, crop, width, height, rng, jitter):
    """A crop box around (cx, cy), offset by up to `jitter` of a half-window.

    Centring the ball every time teaches the head to predict the centre; the
    offset is what forces it to actually localise.
    """
    span = jitter * crop / 2
    x = cx - crop / 2 + rng.uniform(-span, span)
    y = cy - crop / 2 + rng.uniform(-span, span)
    x = int(round(min(max(x, 0), max(width - crop, 0))))
    y = int(round(min(max(y, 0), max(height - crop, 0))))
    return x, y


def _clip_box(bbox, ox, oy, crop, min_visible):
    """Re-express a source bbox inside a crop, or None if too little is left."""
    x, y, w, h = bbox
    x0, y0 = max(x - ox, 0), max(y - oy, 0)
    x1, y1 = min(x - ox + w, crop), min(y - oy + h, crop)
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) * (y1 - y0) < min_visible * w * h:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def plan_crops(record, crop, scale, rng, positives, negatives, jitter, min_visible):
    """Crop windows for one frame: `positives` around each ball, plus negatives.

    Negatives are cut from ball-bearing frames too, not only empty ones — that
    is where wall ball-marks, shoe soles and line junctions live, and they are
    what the model would otherwise fire on.
    """
    width, height = int(round(record["width"] * scale)), int(round(record["height"] * scale))
    if width < crop or height < crop:
        return []
    balls = [{"bbox": [v * scale for v in ball["bbox"]], "streak": ball["streak"]}
             for ball in record["balls"]]

    plans = []
    for ball in balls:
        bx, by, bw, bh = ball["bbox"]
        for _ in range(positives):
            ox, oy = _window(bx + bw / 2, by + bh / 2, crop, width, height, rng, jitter)
            boxes = []
            for other in balls:
                clipped = _clip_box(other["bbox"], ox, oy, crop, min_visible)
                if clipped:
                    boxes.append({"bbox": clipped, "streak": other["streak"]})
            if boxes:
                plans.append({"origin": (ox, oy), "scale": scale, "boxes": boxes})

    for _ in range(negatives):
        ox = rng.randint(0, max(width - crop, 0))
        oy = rng.randint(0, max(height - crop, 0))
        if any(_clip_box(ball["bbox"], ox, oy, crop, min_visible) for ball in balls):
            continue                      # a negative must contain no ball
        plans.append({"origin": (ox, oy), "scale": scale, "boxes": []})
    return plans


def render_split(records, plans_by_record, out_dir, split, crop, quality):
    """Write crop JPEGs and the split's COCO json. Returns the manifest slice."""
    import cv2                            # lazy: geometry is testable without it

    images_dir = out_dir / split
    images_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for record in records:
        plans = plans_by_record.get(record["path"])
        if not plans:
            continue
        frame = cv2.imread(str(record["path"]))
        if frame is None:
            continue
        scale = plans[0]["scale"]
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        for index, plan in enumerate(plans):
            ox, oy = plan["origin"]
            tile = frame[oy:oy + crop, ox:ox + crop]
            if tile.shape[0] != crop or tile.shape[1] != crop:
                continue
            name = f"{record['path'].stem}_c{index}.jpg"
            cv2.imwrite(str(images_dir / name), tile,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": name,
                           "width": crop, "height": crop,
                           "clip": record["clip"], "source_frame": record["frame"]})
            for box in plan["boxes"]:
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [round(v, 2) for v in box["bbox"]],
                    "area": round(box["bbox"][2] * box["bbox"][3], 2),
                    "iscrowd": 0,
                    # Sidecar, not a training target: see streak_metrics().
                    "streak": box["streak"],
                })

    annotations_dir = out_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"instances_{split}.json").write_text(
        json.dumps({"images": images, "annotations": annotations,
                    "categories": [{"id": 0, "name": CLASS_NAME,
                                    "supercategory": "none"}]},
                   sort_keys=True) + "\n", encoding="utf-8")
    return {
        "crops": len(images),
        "annotations": len(annotations),
        "negative_crops": sum(1 for i in images
                              if not any(a["image_id"] == i["id"] for a in annotations)),
        "clips": sorted({r["clip"] for r in records}),
        "source_frames": len(records),
        "source_bursts": burst_count(records),
    }


def build(source, out, crop, positives, negatives, jitter, min_visible,
          min_source_width, min_frame_gap, target_ball_px, val_clips, test_clips,
          seed, quality):
    records, categories = load_export(source)
    total_frames = len(records)

    kept = [r for r in records if r["width"] >= min_source_width]
    dropped_low_res = defaultdict(int)
    for record in records:
        if record["width"] < min_source_width:
            dropped_low_res[f"{record['clip']} ({record['width']}x{record['height']})"] += 1

    kept, thinned = thin_bursts(kept, min_frame_gap)
    scales = clip_scale_factors(kept, target_ball_px)
    splits = split_by_clip(kept, val_clips, test_clips)

    rng = random.Random(seed)
    plans_by_record, per_split = {}, {}
    for split in ("train", "val", "test"):
        split_records = splits.get(split, [])
        for record in split_records:
            # Eval splits get one centred, un-jittered window per ball: a frozen
            # set must measure the model, not the sampler.
            is_train = split == "train"
            plans = plan_crops(
                record, crop, scales.get(record["clip"], 1.0), rng,
                positives if is_train else 1,
                negatives if is_train else 0,
                jitter if is_train else 0.0, min_visible)
            if plans:
                plans_by_record[record["path"]] = plans
        per_split[split] = (split_records, )

    out = Path(out)
    manifest_splits = {}
    for split in ("train", "val", "test"):
        manifest_splits[split] = render_split(
            splits.get(split, []), plans_by_record, out, split, crop, quality)

    ball_widths = sorted(b["bbox"][2] for r in kept for b in r["balls"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_categories": categories,
        "crop": crop,
        "seed": seed,
        "params": {"positives_per_ball": positives, "negatives_per_frame": negatives,
                   "jitter": jitter, "min_visible": min_visible,
                   "min_source_width": min_source_width,
                   "min_frame_gap": min_frame_gap,
                   "target_ball_px": target_ball_px},
        "clip_scale_factors": {k: round(v, 3) for k, v in sorted(scales.items())},
        "source_frames_seen": total_frames,
        "source_frames_kept": len(kept),
        "dropped_low_resolution": dict(sorted(dropped_low_res.items())),
        "dropped_by_burst_thinning": thinned,
        # Raw annotation count overstates the data: these are the temporally
        # separated moments behind it.
        "independent_bursts_kept": burst_count(kept),
        "ball_width_px": {
            "p10": round(ball_widths[int(len(ball_widths) * 0.1)], 1),
            "p50": round(ball_widths[int(len(ball_widths) * 0.5)], 1),
            "p90": round(ball_widths[int(len(ball_widths) * 0.9)], 1),
        } if ball_widths else None,
        "splits": manifest_splits,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path,
                        help="Roboflow COCO export root (contains train/valid/test).")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("ball_crops"))
    parser.add_argument("--crop", type=int, default=416,
                        help="Window size in native source pixels (multiple of 32).")
    parser.add_argument("--positives-per-ball", type=int, default=4)
    parser.add_argument("--negatives-per-frame", type=int, default=2,
                        help="Ball-free windows per frame; these carry the wall "
                             "ball-marks the model would otherwise fire on.")
    parser.add_argument("--jitter", type=float, default=0.8,
                        help="Offset as a fraction of a half-window, so the ball "
                             "is not always centred. 0 centres every crop.")
    parser.add_argument("--min-visible", type=float, default=0.6,
                        help="Fraction of a ball that must survive the crop for "
                             "its box to be kept.")
    parser.add_argument("--min-source-width", type=int, default=MIN_SOURCE_WIDTH)
    parser.add_argument("--min-frame-gap", type=int, default=1,
                        help="Keep at most one frame per N within a clip. >1 "
                             "thins densely-labelled consecutive runs.")
    parser.add_argument("--target-ball-px", type=float, default=0,
                        help="Rescale each clip so its median ball is this wide, "
                             "matching training scale to the 4K rig. 0 = native.")
    parser.add_argument("--val-clips", nargs="*", default=[])
    parser.add_argument("--test-clips", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build(
        args.source.expanduser(), args.out, args.crop, args.positives_per_ball,
        args.negatives_per_frame, args.jitter, args.min_visible,
        args.min_source_width, args.min_frame_gap, args.target_ball_px,
        args.val_clips, args.test_clips, args.seed, args.jpeg_quality)

    print(f"{manifest['source_frames_kept']}/{manifest['source_frames_seen']} "
          f"source frames kept -> {args.out}")
    for name, count in manifest["dropped_low_resolution"].items():
        print(f"  dropped {count} frame(s) below {args.min_source_width}px wide: {name}")
    if manifest["dropped_by_burst_thinning"]:
        print(f"  thinned {manifest['dropped_by_burst_thinning']} near-duplicate frame(s)")
    print(f"  {manifest['independent_bursts_kept']} independent burst(s) behind "
          f"those frames — the real sample count")
    if manifest["ball_width_px"]:
        print(f"  ball width px: {manifest['ball_width_px']}")
    for split, stats in manifest["splits"].items():
        print(f"  {split:5}: {stats['crops']:5d} crops "
              f"({stats['negative_crops']} negative), "
              f"{stats['annotations']:5d} boxes, clips={stats['clips']}")
    if not manifest["splits"]["val"]["crops"]:
        print("  WARNING no val split — pass --val-clips, or every number you "
              "measure will be on data the model trained on")


if __name__ == "__main__":
    main()
