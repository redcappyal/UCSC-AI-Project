# ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
# Not imported by any runtime module, not collected by pytest, not linted.
# Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
"""Offline clock-offset refinement for paired two-camera runs (spec Phase 5).

The live path syncs two phones over the radio to a ~2 ms budget. Offline we can
do better, because we have the whole recording and no latency constraint: the
correct offset is the one that makes both pixel tracks agree on where the ball
was in 3D.

The cost is the mean **clipped** ray gap from `stereo_engine.triangulate` over a
timeline held fixed across every candidate. Both of those details matter:

- *Clipped*, because one frame where a camera lost the ball produces a gap
  large enough to dominate a raw mean and drag the minimum somewhere arbitrary.
- *Fixed timeline*, because if the evaluation grid moves with the offset then
  different candidates are scored on different numbers of points, and the
  minimum becomes an artefact of the grid rather than a fact about the ball.

Search is a two-stage grid, not gradient descent: deterministic, dependency-free
(no scipy), and its evaluation count is reportable. When the result is not
trustworthy — no improvement over the seed, or a minimum sitting on the edge of
the search range — the seed is kept and the reason is recorded. A confident
wrong offset is worse than an honest miss.

**Offset convention, used everywhere in Phase 5:** the offset is the number of
seconds ADDED to camera_role "b"'s timestamps to place both tracks on camera_role
"a"'s clock.
"""

import csv

import numpy as np

import stereo_engine
from stereo_engine import TrackSample

# A gap this large means the two rays never came close — one view had lost the
# ball, or the offset is badly wrong. Clipping here keeps a handful of such
# frames from deciding the answer.
MISS_PENALTY_FT = 2.0

# Above this mean gap the two rays are not looking at the same object, whatever
# the offset. Calibrated against measurement, not taste: with this camera pair a
# real trajectory costs 0.0038 ft clean, and still only 0.082 ft at an absurd
# 12 px of detection noise, while two unrelated pixel streams cost 1.39 ft. The
# spec's own free-triangulation error is ~0.43 ft at the front wall, so half a
# foot is both comfortably above every real case and far below nonsense.
MAX_AGREEING_GAP_FT = 0.5

# Evaluation timeline density. Matches stereo_engine.build_track3d's default so
# the refined offset is measured on the same grid the fusion will later use.
TIMELINE_HZ = 120.0

# Cost within this factor of the minimum is "indistinguishable"; the width of
# that region is reported as an uncertainty proxy.
HALF_WIDTH_TOLERANCE = 1.1


def track_samples_from_csv(csv_path, fps, offset_s=0.0):
    """`ball_coordinates.csv` -> time-sorted `[stereo_engine.TrackSample]`.

    Rows with no detection are skipped — they carry no pixel. `offset_s` is
    added to every timestamp, so a caller can load camera b already on camera
    a's clock.

    `timestamp_seconds` is preferred when present; older CSVs predate it, and
    `source_frame / fps` locates those rows just as well.
    """
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("detected", "") or "").strip().lower() not in {"true", "1", "yes"}:
                continue
            x_raw, y_raw = row.get("x_center", ""), row.get("y_center", "")
            if not x_raw or not y_raw:
                continue
            stamp = (row.get("timestamp_seconds", "") or "").strip()
            try:
                if stamp:
                    t_s = float(stamp)
                else:
                    t_s = float(row["source_frame"]) / float(fps)
                px = (float(x_raw), float(y_raw))
            except (TypeError, ValueError, KeyError, ZeroDivisionError):
                continue
            samples.append(TrackSample(t_s=t_s + offset_s, px=px))
    samples.sort(key=lambda s: s.t_s)
    return samples


def seed_offset_from_manifest(manifest):
    """Best starting offset the phones themselves reported -> (offset_s, source).

    The clap anchor is an acoustic fix on one shared instant and beats the
    network estimator, so it wins when present. Everything here is tolerant:
    the manifest crosses a network from a client, and an unusable field means
    "start at zero", never a crash.
    """
    if isinstance(manifest, dict):
        anchor = manifest.get("clap_anchor_s")
        if isinstance(anchor, (int, float)) and not isinstance(anchor, bool):
            if np.isfinite(anchor):
                return float(anchor), "clap_anchor"

        series = manifest.get("offset_series")
        if isinstance(series, (list, tuple)):
            values = [float(v) for v in series
                      if isinstance(v, (int, float)) and not isinstance(v, bool)
                      and np.isfinite(v)]
            if values:
                return float(np.median(values)), "offset_series_median"
    return 0.0, "none"


def _timeline(samples_a, samples_b, offset_s, hz=TIMELINE_HZ):
    """The fixed evaluation grid: the overlap of both streams at the seed."""
    if not samples_a or not samples_b:
        return np.empty(0)
    t_lo = max(samples_a[0].t_s, samples_b[0].t_s + offset_s)
    t_hi = min(samples_a[-1].t_s, samples_b[-1].t_s + offset_s)
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
        return np.empty(0)
    count = int((t_hi - t_lo) * hz)
    if count < 3:
        return np.empty(0)
    return t_lo + np.arange(count) / hz


def _cost_at(model_a, samples_a, model_b, samples_b, timeline, offset_s):
    """Mean clipped ray gap, and how many timeline points actually evaluated.

    Points where either view cannot be interpolated, or where triangulation
    refuses (near-parallel rays, or a closest approach behind a camera), score
    the miss penalty rather than being dropped — otherwise an offset that
    destroys the overlap would look *better* than one that preserves it, by
    quietly scoring fewer, easier points.
    """
    if timeline.size == 0:
        return float("inf"), 0
    total, evaluated = 0.0, 0
    for t in timeline:
        px_a = stereo_engine.eval_pixel_track(samples_a, float(t))
        px_b = stereo_engine.eval_pixel_track(samples_b, float(t) - offset_s)
        if px_a is None or px_b is None:
            total += MISS_PENALTY_FT
            continue
        _, gap = stereo_engine.triangulate(model_a, model_b, px_a, px_b)
        total += min(float(gap), MISS_PENALTY_FT) if np.isfinite(gap) else MISS_PENALTY_FT
        evaluated += 1
    return total / float(timeline.size), evaluated


def refine_offset(model_a, samples_a, model_b, samples_b, seed_s=0.0,
                  coarse_range_s=0.05, coarse_step_s=0.002, fine_step_s=0.0001):
    """Search the offset that best explains both pixel tracks as one 3D ball.

    `samples_b` are on camera b's own clock. The returned offset is what must be
    ADDED to those timestamps to land on camera a's clock.

    Returns a JSON-serialisable report — it is written verbatim into
    `sync_report.json`.
    """
    timeline = _timeline(samples_a, samples_b, seed_s)
    seed_cost, seed_evaluated = _cost_at(
        model_a, samples_a, model_b, samples_b, timeline, seed_s)

    report = {
        "offset_convention": (
            "seconds added to camera_role 'b' timestamps to place both tracks "
            "on camera_role 'a' clock"),
        "seed": {"offset_s": float(seed_s)},
        "refined": {"offset_s": float(seed_s), "accepted": False, "reason": ""},
        "cost": {"metric": "mean_clipped_ray_gap_ft",
                 "at_seed": None if not np.isfinite(seed_cost) else float(seed_cost),
                 "at_refined": None,
                 "miss_penalty_ft": MISS_PENALTY_FT},
        "search": {"coarse_range_s": float(coarse_range_s),
                   "coarse_step_s": float(coarse_step_s),
                   "fine_step_s": float(fine_step_s),
                   "evaluations": 0,
                   "argmin_interior": False},
        "timeline": {"count": int(timeline.size), "hz": TIMELINE_HZ,
                     "t_lo_s": float(timeline[0]) if timeline.size else None,
                     "t_hi_s": float(timeline[-1]) if timeline.size else None,
                     "evaluated_at_seed": int(seed_evaluated)},
        "half_width_s": None,
        "sample_counts": {"a": len(samples_a), "b": len(samples_b)},
    }

    if timeline.size == 0:
        report["refined"]["reason"] = "no overlapping timeline at the seed offset"
        return report

    def scan(centre, span, step):
        """Cost over a symmetric grid. Returns (offsets, costs)."""
        steps = max(1, int(round(span / step)))
        grid = centre + np.arange(-steps, steps + 1) * step
        return grid, np.array([
            _cost_at(model_a, samples_a, model_b, samples_b, timeline, float(d))[0]
            for d in grid])

    coarse_grid, coarse_costs = scan(seed_s, coarse_range_s, coarse_step_s)
    best_index = int(np.argmin(coarse_costs))
    interior = 0 < best_index < len(coarse_grid) - 1
    evaluations = len(coarse_grid)

    # Only polish a minimum the coarse pass actually bracketed. Refining an
    # edge minimum would just walk further out on a slope with no evidence
    # the true offset is there at all.
    if interior:
        fine_grid, fine_costs = scan(
            float(coarse_grid[best_index]), coarse_step_s, fine_step_s)
        evaluations += len(fine_grid)
        fine_index = int(np.argmin(fine_costs))
        best_offset = float(fine_grid[fine_index])
        best_cost = float(fine_costs[fine_index])
        # Where does the cost stop being distinguishable from the minimum?
        within = fine_grid[fine_costs <= best_cost * HALF_WIDTH_TOLERANCE]
        report["half_width_s"] = (
            float(max(abs(within[0] - best_offset), abs(within[-1] - best_offset)))
            if within.size else float(fine_step_s))
    else:
        best_offset = float(coarse_grid[best_index])
        best_cost = float(coarse_costs[best_index])

    report["search"]["evaluations"] = int(evaluations)
    report["search"]["argmin_interior"] = bool(interior)
    report["cost"]["at_refined"] = float(best_cost)

    if not interior:
        reason = ("coarse minimum sits on the edge of the search range — the true "
                  "offset is probably outside it")
    elif not np.isfinite(seed_cost) or best_cost >= seed_cost:
        reason = "refinement did not improve on the seed"
    elif best_cost > MAX_AGREEING_GAP_FT:
        # The best offset still leaves the rays feet apart: these two streams
        # are not watching the same ball. A marginal improvement over an
        # equally-meaningless seed is not evidence of a real sync.
        reason = "no offset makes the two tracks agree — check the pairing"
    else:
        report["refined"] = {"offset_s": best_offset, "accepted": True,
                             "reason": "cost_improved"}
        return report

    report["refined"] = {"offset_s": float(seed_s), "accepted": False, "reason": reason}
    return report
