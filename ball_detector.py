"""Tiled native-resolution ball detection.

ios/MODEL.md §6: never downscale the whole frame. The model was trained on
416 px windows cut at native source resolution, where the ball is 7-24 px.
Resizing a 4K frame to 960 px wide makes a p50 10.7 px ball ~2.7 px -- outside
anything the model has seen. So we slide the trained window over the frame at
native scale instead.

The window strategy is deliberately a seam: §6's velocity-extrapolated single
crop can replace tile_windows later without touching the model adapter.
"""


def tile_windows(frame_w, frame_h, tile, overlap):
    """Top-left origins of tiles covering the frame, clamped in-bounds.

    Overlap must exceed the largest expected ball (p90 is 24 px) so no ball is
    clipped across a seam in every tile that sees it.
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap must be in [0, {tile}), got {overlap}")

    stride = tile - overlap

    def origins(extent):
        if extent <= tile:
            return [0]
        starts = list(range(0, extent - tile + 1, stride))
        if starts[-1] != extent - tile:
            starts.append(extent - tile)
        return starts

    return [(x, y) for y in origins(frame_h) for x in origins(frame_w)]


def _corners(box):
    half_w = box["width"] / 2.0
    half_h = box["height"] / 2.0
    return (box["x"] - half_w, box["y"] - half_h,
            box["x"] + half_w, box["y"] + half_h)


def iou(a, b):
    ax0, ay0, ax1, ay1 = _corners(a)
    bx0, by0, bx1, by1 = _corners(b)
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    union = (a["width"] * a["height"]) + (b["width"] * b["height"]) - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(detections, iou_threshold):
    """Greedy NMS across tiles. Overlapping tiles see the same ball twice."""
    kept = []
    for detection in sorted(detections,
                            key=lambda d: d.get("confidence", 0.0),
                            reverse=True):
        if all(iou(detection, k) <= iou_threshold for k in kept):
            kept.append(detection)
    return kept


def _crop(frame, x0, y0, tile):
    """Tile-sized crop, zero-padded when the frame is smaller than one tile.

    Padding keeps the network input shape static, which the traced TorchScript
    graph requires.
    """
    import numpy as np

    patch = frame[y0:y0 + tile, x0:x0 + tile]
    height, width = patch.shape[:2]
    if height == tile and width == tile:
        return patch
    padded = np.zeros((tile, tile, patch.shape[2]), dtype=patch.dtype)
    padded[:height, :width] = patch
    return padded


def _tile_edge(input_size):
    """The single tile edge, refusing a non-square manifest.

    A (416, 320) manifest would otherwise silently cut 416x416 windows and feed
    the network a shape it was never traced for.
    """
    try:
        width, height = (int(v) for v in input_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"input_size must be a (width, height) pair, got {input_size!r}"
        ) from exc
    if width != height:
        raise ValueError(
            f"input_size must be square for the tiled sweep, got "
            f"({width}, {height}); one square window per tile is the only "
            f"shape the traced graph accepts.")
    return width


def _class_name(class_names, class_index):
    """Manifest class name for an index, or a loud failure.

    Never fabricates a label: a negative index would otherwise wrap onto the
    last class name, and an over-range one onto a made-up string. Like
    ball_model's manifest checks, an inconsistency raises rather than guesses.
    """
    index = int(class_index)
    if not 0 <= index < len(class_names):
        raise ValueError(
            f"detector emitted class_index {index}, outside the manifest's "
            f"{len(class_names)} class name(s) {tuple(class_names)}; the label "
            f"is not guessed. Re-export the model and manifest together.")
    return class_names[index]


def detect_frame(runner, frame, manifest):
    """Full-frame detections from a tiled sweep.

    `runner.run_batch(crops)` returns, per crop, a list of
    (cx, cy, w, h, score, class_index) in TILE-LOCAL pixels. This function owns
    the tile-local -> full-frame mapping and the cross-tile merge.
    """
    tile = _tile_edge(manifest.input_size)
    frame_h, frame_w = frame.shape[:2]
    windows = tile_windows(frame_w, frame_h, tile, manifest.tile_overlap_px)

    detections = []
    batch = max(1, manifest.max_batch_tiles)
    for start in range(0, len(windows), batch):
        chunk = windows[start:start + batch]
        crops = [_crop(frame, x0, y0, tile) for x0, y0 in chunk]
        # strict: a runner that drops empty tiles (or returns short for any
        # other reason) would otherwise re-align every later result onto the
        # wrong tile origin -- silently, which is the one failure this module
        # exists to prevent.
        for (x0, y0), boxes in zip(chunk, runner.run_batch(crops), strict=True):
            for cx, cy, width, height, score, class_index in boxes:
                if score < manifest.conf_threshold:
                    continue
                name = _class_name(manifest.class_names, class_index)
                detections.append({
                    "x": float(cx) + x0,
                    "y": float(cy) + y0,
                    "width": float(width),
                    "height": float(height),
                    "confidence": float(score),
                    "class": name,
                    "class_name": name,
                })

    return merge_detections(detections, manifest.nms_iou)


def detect_frame_stack(runner, frames, manifest):
    """Full-frame detections for the MIDDLE frame of an oldest-first stack.

    `frames` is `[t-1, t, t+1, ...]`, length `manifest.frames_per_input`. This
    cuts co-located tiles from every frame, concatenates them along the
    channel axis (oldest first -- the temporal ordering the model was trained
    on, see HeatmapRunner), and reuses the same tile-local -> full-frame
    mapping and cross-tile merge as detect_frame. The runner decodes the
    centre MIMO channel, so the returned detections locate the ball in
    frames[len(frames) // 2], not in every frame of the stack.
    """
    import numpy as np

    expected = int(manifest.frames_per_input)
    if len(frames) != expected:
        raise ValueError(
            f"detect_frame_stack got {len(frames)} frames but "
            f"manifest.frames_per_input is {expected}")
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"detect_frame_stack frames disagree on shape: {sorted(shapes)}")

    tile = _tile_edge(manifest.input_size)
    frame_h, frame_w = frames[0].shape[:2]
    windows = tile_windows(frame_w, frame_h, tile, manifest.tile_overlap_px)

    detections = []
    batch = max(1, manifest.max_batch_tiles)
    for start in range(0, len(windows), batch):
        chunk = windows[start:start + batch]
        stacks = [
            np.concatenate([_crop(frame, x0, y0, tile) for frame in frames], axis=2)
            for x0, y0 in chunk
        ]
        # strict: see detect_frame's comment -- a runner that drops a stack
        # would otherwise re-align every later result onto the wrong origin.
        for (x0, y0), boxes in zip(chunk, runner.run_batch(stacks), strict=True):
            for cx, cy, width, height, score, class_index in boxes:
                if score < manifest.conf_threshold:
                    continue
                name = _class_name(manifest.class_names, class_index)
                detections.append({
                    "x": float(cx) + x0,
                    "y": float(cy) + y0,
                    "width": float(width),
                    "height": float(height),
                    "confidence": float(score),
                    "class": name,
                    "class_name": name,
                })

    return merge_detections(detections, manifest.nms_iou)
