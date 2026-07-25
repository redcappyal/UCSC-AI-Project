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
