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
# Schema used once any image entry carries a "sequence" field (--seq-frames >
# 1): consumers that don't understand 3-frame sequences can key off this to
# fall back to the anchor-only crop rather than choking on the extra field.
SCHEMA_VERSION_SEQ = "ball-crops-v2"
CLASS_NAME = "ball"

# Mean |pixel diff| tolerance for verify_frame_alignment: JPEG recompression
# of the export image plus a decode of the same source frame should land well
# under this; a genuine off-by-N frame mismatch does not.
ALIGNMENT_MAX_ABS_DIFF = 12.0
VIDEO_EXTENSIONS = (".mp4", ".mov", ".MP4", ".MOV")

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

    A residual risk survives this: a stem already shaped like a slug's output
    (e.g. "Rally_One-4075aa34") is returned untouched, and a genuinely
    different stem can slug to that same string (e.g. "Rally ｜ One") —
    probability ~2⁻³² given the 8-hex digest. render_split's duplicate-
    filename guard is what makes that acceptable: the collision surfaces as
    a SystemExit naming both source paths, never a silent overwrite.
    """
    folded = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    slug = UNSAFE_CHARS.sub("_", folded).strip("._-") or "clip"
    if slug == stem:
        return slug
    return f"{slug}-{hashlib.sha1(stem.encode('utf-8')).hexdigest()[:8]}"


def crop_file_name(source_stem, index):
    """Filename for the `index`-th crop cut from a source frame."""
    return f"{ascii_slug(source_stem)}_c{index}.jpg"


def slugified_clips(records):
    """Clips whose crops carry a digest suffix in their filename.

    Crop filenames are slugged from `record["path"].stem`, not from `clip`
    itself, so this checks the stem for divergence and reports the clip it
    belongs to. Reported in the manifest so the digest suffix on those crops
    is self-explaining; the readable name itself stays in each image's `clip`.
    """
    return sorted({r["clip"] for r in records
                   if ascii_slug(r["path"].stem) != r["path"].stem})


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


def _imread_unicode(path):
    """cv2.imread that survives a non-ASCII path on Windows.

    cv2 gets the UTF-8 bytes of a Python str and hands them to the CRT's
    non-Unicode file API, so on a cp1252 box every non-ASCII path misses and
    imread returns None with the file sitting right there. Reading the bytes in
    Python and decoding them in memory sidesteps the path entirely. Returns None
    for a missing or unreadable file, as imread did.
    """
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path, image, params=None):
    """cv2.imwrite that survives a non-ASCII path on Windows.

    The read-side failure is loud; this one is not. cv2.imwrite returns True and
    writes a real file whose name is the UTF-8 bytes reinterpreted as cp1252, so
    the COCO json ends up pointing at files that do not exist — how the
    2026-07-24 dataset lost 961 of its 2,936 train crops. This function raises
    rather than returning a status, so — unlike cv2.imwrite — there is no path
    through it that reports success without the file landing on disk. In
    practice cv2.imencode raises cv2.error on a missing or unknown extension
    rather than returning ok=False; the `if not ok` check below is a defensive
    backstop, not the mechanism this guarantee actually rests on.
    """
    import cv2

    ok, buffer = cv2.imencode(Path(path).suffix, image, params or [])
    if not ok:
        raise OSError(f"cv2 could not encode {path}")
    buffer.tofile(str(path))


def find_clip_videos(clips_dir, records):
    """{record["clip"]: source video Path} for every clip a record needs.

    `record["clip"]` (from `clip_and_frame`) is the readable source name --
    spaces, unicode, whatever the export carried -- not a slug. Matching it
    against a video filename therefore has to slug *both* sides through
    `ascii_slug` (the same transform crop filenames go through), not just
    the video's: comparing a raw readable clip name against an already-
    slugged video stem never matches, which would make every video look
    missing. The returned dict is still keyed by the readable name, since
    that's what callers already have on hand as `record["clip"]`.

    Dies listing every missing clip by its readable name (so an operator can
    act on it) at once rather than one-by-one, so a --clips-dir pointed at
    the wrong folder fails in a single readable message. Also dies loudly if
    --clips-dir itself doesn't exist or isn't a directory, rather than
    leaking a raw FileNotFoundError.
    """
    clips_dir = Path(clips_dir)
    try:
        candidates = sorted(clips_dir.iterdir())
    except FileNotFoundError:
        raise SystemExit(f"--clips-dir {clips_dir} does not exist.")
    except NotADirectoryError:
        raise SystemExit(f"--clips-dir {clips_dir} is not a directory.")

    by_slug = {}
    for path in candidates:
        if path.suffix in VIDEO_EXTENSIONS:
            by_slug.setdefault(ascii_slug(path.stem), path)
    needed = sorted({r["clip"] for r in records})
    missing = [clip for clip in needed if ascii_slug(clip) not in by_slug]
    if missing:
        raise SystemExit(
            f"--clips-dir {clips_dir} has no video for clip(s): {missing}. "
            f"Sequence crops need the source footage for neighbour frames.")
    return {clip: by_slug[ascii_slug(clip)] for clip in needed}


def decode_frames(video_path, indices):
    """{index: frame_bgr} for the requested frame indices, one sequential pass.

    Sequential read only, never a CAP_PROP_POS_FRAMES seek: HEVC seeking
    lands on the nearest keyframe unreliably across OpenCV builds, and a
    silently wrong frame is exactly the poison sequence crops must not
    produce. Fatal if the video ends before a requested index is reached --
    callers are expected to have already clamped indices to the clip's known
    frame range (see `_clamp_frame`), so hitting this means the clip is
    shorter than the export claims, which is worth stopping for.
    """
    import cv2

    wanted = sorted(set(indices))
    out = {}
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video_path}")
        index = 0
        current = None
        for target in wanted:
            while index <= target:
                ok, decoded = cap.read()
                if not ok:
                    raise SystemExit(
                        f"{video_path} ended at frame {index}; needed {target}")
                current, index = decoded, index + 1
            out[target] = current
    finally:
        cap.release()
    return out


def verify_frame_alignment(video_frame, export_image):
    """Mean absolute per-pixel difference between a decoded video frame and
    the export image it should match. Resizes the export image first if the
    export was scaled down from the source resolution, since a resolution
    mismatch would otherwise swamp a genuine content mismatch."""
    import cv2
    import numpy as np

    if video_frame.shape[:2] != export_image.shape[:2]:
        export_image = cv2.resize(
            export_image, (video_frame.shape[1], video_frame.shape[0]))
    return float(np.mean(np.abs(
        video_frame.astype(np.int16) - export_image.astype(np.int16))))


def _clamp_frame(index, last):
    """Keep a frame index inside [0, last] (no upper clamp if last is None).

    This is the dataset-side counterpart of ball_track_offline's
    `_centered_windows`, which pads a clip's edges by repeating the first/
    last frame rather than reaching outside the clip; using the same clamp
    here keeps training aligned with what the runtime model actually sees.

    `last` (`clip_last_frame` in callers) is the *max labelled* frame in the
    clip, not the video's real frame count -- the dataset side has no other
    signal for where a clip ends. That means the highest-labelled record in
    every clip gets a padded (repeated-last-frame) tp1 even when real footage
    exists beyond it. Deliberate: a slightly duller tp1 for one record per
    clip is a small, known density cost, versus needing extra video-metadata
    plumbing just to avoid it.
    """
    index = max(index, 0)
    if last is not None:
        index = min(index, last)
    return index


def _sequence_targets(frame_index, last_frame):
    """(tm1, t, tp1) indices for a centred window around `frame_index`,
    clamped into the clip so the first/last frame repeats at the edges."""
    return (_clamp_frame(frame_index - 1, last_frame), frame_index,
            _clamp_frame(frame_index + 1, last_frame))


def _needed_indices(clip_records, last_frame):
    """Every index that could be needed to resolve alignment at offset
    -1/0/+1 and then build the resulting window, for one clip's records --
    the ±2 span around each labelled frame covers every offset's window."""
    wanted = set()
    for record in clip_records:
        for delta in (-2, -1, 0, 1, 2):
            wanted.add(_clamp_frame(record["frame"] + delta, last_frame))
    return sorted(wanted)


def _resolve_clip_offset(clip, clip_records, decoded, export_images, last_frame,
                         tolerance):
    """0, -1, or +1: the single frame-index shift whose whole-clip alignment
    is best, among candidates under `tolerance`.

    Selection is argmin, not first-pass-wins. On static-camera footage the
    whole-frame mean|diff| between adjacent frames is usually already under
    `tolerance` at offset 0 -- two near-duplicate frames just don't differ by
    much -- so a first-offset-that-passes rule would silently keep offset 0
    even when +1 or -1 is a strictly better fit. That is exactly the silent
    poison this check exists to prevent: it would keep passing the tolerance
    gate while quietly aligning every sequence one frame off. Each
    candidate's score is the WORST (max) per-record diff, which is what
    preserves the "a single offset aligns every record in the clip"
    requirement -- a candidate is only picked if it is both the best of the
    three AND still under tolerance. `export_images` is precomputed once per
    record by the caller so this reads each export JPEG once, not once per
    candidate offset.

    Residual limitation: a genuinely ambiguous clip whose diffs tie across
    offsets resolves to whichever candidate is listed first in `(0, -1, 1)`
    -- ties are not this check's problem to solve, only misalignment is.
    """
    scores = {}
    for offset in (0, -1, 1):
        diffs = []
        for record in clip_records:
            idx = _clamp_frame(record["frame"] + offset, last_frame)
            frame = decoded.get(idx)
            export_image = export_images.get(record["path"])
            if frame is None or export_image is None:
                diffs = None
                break
            diffs.append(verify_frame_alignment(frame, export_image))
        scores[offset] = max(diffs) if diffs else float("inf")

    best_offset = min(scores, key=lambda o: scores[o])
    if scores[best_offset] <= tolerance:
        return best_offset
    raise SystemExit(
        f"sequence alignment failed for clip {clip!r}: no single frame offset "
        f"(0, -1, +1) aligns every export image in this clip to its decoded "
        f"video frame -- the clip's video and export are out of sync.")


def _decode_sequences(records, plans_by_record, clip_videos, clip_last_frame, crop,
                      tolerance=ALIGNMENT_MAX_ABS_DIFF):
    """{(record.path, plan_index): (tm1_tile, tp1_tile)} cropped-to-window
    neighbour-frame patches for every planned crop of every record with a
    matched video.

    Decoded and cropped one clip at a time -- peak memory is bounded by a
    single clip's native-resolution frames, not the whole split's. The naive
    version of this function decoded every clip in the split up front and
    stored two full native-resolution frames per record before any cropping
    happened, which is multiple GB at 1080p and tens of GB at 4K on the real
    dataset. Cropping to the planned window immediately, inside this
    per-clip loop, means only tiny tiles (`crop`x`crop`) survive past a
    clip's iteration -- the full decoded frames for one clip are eligible
    for garbage collection as soon as the loop moves to the next clip.
    """
    import cv2                            # lazy: geometry is testable without it

    by_clip = defaultdict(list)
    for record in records:
        if (record["path"] in plans_by_record and record["frame"] is not None
                and record["clip"] in clip_videos):
            by_clip[record["clip"]].append(record)

    tiles = {}
    for clip, clip_records in sorted(by_clip.items()):
        video = clip_videos[clip]
        last_frame = clip_last_frame.get(clip)
        decoded = decode_frames(video, _needed_indices(clip_records, last_frame))
        # Read each record's export image once here -- shared by the offset
        # probe below and the native-shape resize that follows -- rather
        # than once per candidate offset (up to 3x the same JPEG).
        export_images = {record["path"]: _imread_unicode(record["path"])
                         for record in clip_records}
        offset = _resolve_clip_offset(
            clip, clip_records, decoded, export_images, last_frame, tolerance)
        for record in clip_records:
            # Clamp the shifted anchor into the clip before deriving tm1/tp1
            # from it: without this, a record at the max-labelled edge plus
            # a nonzero offset computes an anchor past `last_frame`, and the
            # tp1 clamp (below, against the unshifted `last_frame`) silently
            # collapses onto tm1's index instead of repeating the anchor --
            # [t-1, t, t-1] instead of the intended edge-padding convention
            # [t-1, t, t].
            true_t = _clamp_frame(record["frame"] + offset, last_frame)
            tm1, _, tp1 = _sequence_targets(true_t, last_frame)
            tm1_frame, tp1_frame = decoded[tm1], decoded[tp1]

            export_image = export_images[record["path"]]
            # The decoded video frame and the export image are both meant to
            # be the same source pixels, but the export may have been
            # resized on the way out of Roboflow; match the export's native
            # size before the dataset-level `scale` (below) is applied, same
            # as the anchor frame gets in render_split.
            native_shape = export_image.shape[:2]
            if tm1_frame.shape[:2] != native_shape:
                tm1_frame = cv2.resize(tm1_frame, (native_shape[1], native_shape[0]))
            if tp1_frame.shape[:2] != native_shape:
                tp1_frame = cv2.resize(tp1_frame, (native_shape[1], native_shape[0]))

            plans = plans_by_record[record["path"]]
            scale = plans[0]["scale"]
            if scale != 1.0:
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                tm1_frame = cv2.resize(tm1_frame, None, fx=scale, fy=scale,
                                       interpolation=interp)
                tp1_frame = cv2.resize(tp1_frame, None, fx=scale, fy=scale,
                                       interpolation=interp)

            for index, plan in enumerate(plans):
                ox, oy = plan["origin"]
                # .copy(): a bare slice is a numpy VIEW, which keeps the
                # full-resolution tm1_frame/tp1_frame array alive in memory
                # for as long as the tile survives -- i.e. for the rest of
                # this function, defeating the whole point of cropping early.
                # The copy is what actually lets the full frame be collected
                # once this clip's iteration ends.
                tiles[(record["path"], index)] = (
                    tm1_frame[oy:oy + crop, ox:ox + crop].copy(),
                    tp1_frame[oy:oy + crop, ox:ox + crop].copy())
        # `decoded`, `export_images`, and every tm1_frame/tp1_frame above go
        # out of scope here as the loop advances to the next clip -- nothing
        # from this clip is retained past this point except the small tiles
        # already copied into `tiles`.
    return tiles


def render_split(records, plans_by_record, out_dir, split, crop, quality,
                 *, seq_frames=1, clip_videos=None, clip_last_frame=None):
    """Write crop JPEGs and the split's COCO json. Returns the manifest slice.

    `crop_file_name` keys off `record["path"].stem`, dropping the parent
    directory, so two source records with the same file name in different
    export split dirs would otherwise plan to the same crop filename and the
    second write would silently overwrite the first's pixels while the COCO
    json still carried two distinct `images` entries. `emitted` below turns
    that into a loud failure instead.

    `seq_frames=1` (the default) is byte-identical to the original single-
    frame behaviour. `seq_frames > 1` also cuts `<stem>.tm1.jpg` /
    `<stem>.tp1.jpg` beside each anchor crop from `clip_videos`' source
    footage, and records `"sequence": [tm1, anchor, tp1]` on the image entry
    -- see `_decode_sequences` for the decode/alignment work.
    """
    import cv2                            # lazy: geometry is testable without it

    images_dir = out_dir / split
    images_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    emitted = {}                           # crop filename -> source path

    # {(record.path, plan_index): (tm1_tile, tp1_tile)} -- already cropped to
    # the planned window and dataset-scale by _decode_sequences, one clip's
    # native frames at a time, so this dict holds only small tiles rather
    # than full-resolution frames for the whole split.
    tiles = {}
    if seq_frames > 1 and clip_videos:
        tiles = _decode_sequences(
            records, plans_by_record, clip_videos, clip_last_frame or {}, crop)

    for record in records:
        plans = plans_by_record.get(record["path"])
        if not plans:
            continue
        frame = _imread_unicode(record["path"])
        if frame is None:
            continue
        scale = plans[0]["scale"]

        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interp)

        for index, plan in enumerate(plans):
            ox, oy = plan["origin"]
            tile = frame[oy:oy + crop, ox:ox + crop]
            if tile.shape[0] != crop or tile.shape[1] != crop:
                continue
            name = crop_file_name(record["path"].stem, index)
            other = emitted.get(name)
            if other is not None and other != record["path"]:
                raise SystemExit(
                    f"crop filename collision in split {split!r}: {name!r} "
                    f"would be written for both {other} and {record['path']}")
            emitted[name] = record["path"]
            _imwrite_unicode(images_dir / name, tile,
                             [int(cv2.IMWRITE_JPEG_QUALITY), quality])

            sequence_names = None
            tm1_tp1 = tiles.get((record["path"], index))
            if tm1_tp1 is not None:
                tm1_tile, tp1_tile = tm1_tp1
                stem = name[:-len(".jpg")] if name.endswith(".jpg") else name
                tm1_name, tp1_name = f"{stem}.tm1.jpg", f"{stem}.tp1.jpg"
                _imwrite_unicode(images_dir / tm1_name, tm1_tile,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                _imwrite_unicode(images_dir / tp1_name, tp1_tile,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                # Oldest-first, labelled frame in the middle -- the centred-
                # window convention _centered_windows also serves at runtime.
                sequence_names = [tm1_name, name, tp1_name]

            image_id = len(images) + 1
            image_entry = {"id": image_id, "file_name": name,
                           "width": crop, "height": crop,
                           "clip": record["clip"], "source_frame": record["frame"]}
            if sequence_names:
                image_entry["sequence"] = sequence_names
            images.append(image_entry)
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
          seed, quality, *, seq_frames=1, clips_dir=None):
    if seq_frames % 2 == 0:
        raise SystemExit(
            f"--seq-frames must be odd (got {seq_frames}): sequence length "
            f"must be odd -- the labeled frame sits in the middle of a "
            f"centered window.")
    if seq_frames > 1 and not clips_dir:
        raise SystemExit(
            f"--seq-frames {seq_frames} needs --clips-dir pointing at the "
            f"source videos, to decode the t-1/t+1 neighbour frames.")

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

    clip_videos = find_clip_videos(clips_dir, kept) if seq_frames > 1 else None
    clip_last_frame = {}
    if seq_frames > 1:
        for record in kept:
            if record["frame"] is not None:
                clip_last_frame[record["clip"]] = max(
                    clip_last_frame.get(record["clip"], record["frame"]), record["frame"])

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
            splits.get(split, []), plans_by_record, out, split, crop, quality,
            seq_frames=seq_frames, clip_videos=clip_videos, clip_last_frame=clip_last_frame)

    ball_widths = sorted(b["bbox"][2] for r in kept for b in r["balls"])
    manifest = {
        "schema_version": SCHEMA_VERSION_SEQ if seq_frames > 1 else SCHEMA_VERSION,
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
        # The readable name lives in each image's `clip`; this flags the clips
        # whose crops therefore carry a digest suffix.
        "slugified_clips": slugified_clips(kept),
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
    parser.add_argument("--clips-dir", type=Path, default=None,
                        help="Source video folder, for decoding sequence "
                             "neighbour frames. Required if --seq-frames > 1.")
    parser.add_argument("--seq-frames", type=int, default=1,
                        help="1 = today's single-frame crops (default). >1 "
                             "(must be odd) also cuts t-1/t+1 crops from "
                             "--clips-dir, aligned to the labelled frame.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build(
        args.source.expanduser(), args.out, args.crop, args.positives_per_ball,
        args.negatives_per_frame, args.jitter, args.min_visible,
        args.min_source_width, args.min_frame_gap, args.target_ball_px,
        args.val_clips, args.test_clips, args.seed, args.jpeg_quality,
        seq_frames=args.seq_frames,
        clips_dir=args.clips_dir.expanduser() if args.clips_dir else None)

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
