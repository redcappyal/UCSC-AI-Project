"""Detection tests for court_detect, scored against a synthetic court.

The synthetic court is rendered from a known CameraModel, so every assertion
compares the detector's answer to the camera that produced the image. Paint
bands carry real WSF width (50 mm), which is what makes the datum rules in
docs/superpowers/specs/2026-07-27-auto-court-detection-design.md §5 testable:
returning a stripe's centre instead of its named edge is a half-line-width
bias, and it must fail here rather than in a match.
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import court_detect
from court_model import LEFT_BOX_INNER_CENTER_X_FT, LINE_WIDTH_FT
from synthetic_court import FLOOR_BGR, LINE_BGR, WALL_BGR, court_camera, render_court

REPO_ROOT = Path(__file__).resolve().parents[1]
SQUASH_ANALYTICS_MP4 = REPO_ROOT / "SquashAnalytics.mp4"
BAY_CLUB_MOV = Path("/Users/Ian2/Desktop/Training Data/Bay Club Squash 1 Rally+audio.mov")


def test_render_court_paints_lines_where_the_camera_projects_them():
    camera = court_camera()
    image, truth = render_court(camera)

    assert image.shape == (1080, 1920, 3)

    # The out line's stored datum is its LOWER edge, so the pixel just ABOVE
    # that datum is paint and the pixel well above it is bare wall.
    (x1, y1), (x2, y2) = truth["out_line_lower_edge"]
    mid_x, mid_y = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
    assert np.allclose(image[mid_y - 1, mid_x], LINE_BGR, atol=2)
    assert np.allclose(image[mid_y - 40, mid_x], WALL_BGR, atol=2)

    # Below the front-wall/floor seam is floor, above it is wall.
    (sx1, sy1), (sx2, sy2) = truth["front_seam"]
    seam_x, seam_y = int(round((sx1 + sx2) / 2)), int(round((sy1 + sy2) / 2))
    assert np.allclose(image[seam_y + 25, seam_x], FLOOR_BGR, atol=2)
    assert np.allclose(image[seam_y - 25, seam_x], WALL_BGR, atol=2)


def test_service_box_lines_straddle_their_centreline_not_their_edge():
    """The left service box's inner side-line band must straddle
    LEFT_BOX_INNER_CENTER_X_FT, not the raw SERVICE_BOX_FT edge datum.

    SERVICE_BOX_FT is a WSF EDGE datum (the paint's interior-facing
    boundary, court_model.py:27-35), not a centreline. An earlier version of
    the renderer straddled that raw datum, which draws the whole band a half
    line-width (~25 mm) toward the box interior -- silently disagreeing with
    the *_CENTER_* constants that Task 5 scores its self-verification
    against. Neither existing render test would have caught this, since both
    only check `out_line_lower_edge` and `front_seam`.
    """
    # court_camera()'s default framing (focal_px=1600) is narrow enough that
    # y >= ~10.5 ft projects below row 1080 -- it was tuned in Task 1 only to
    # keep the out line and front seam in frame. The service box (y in
    # [18.0, 23.25]) needs a wider FOV to be visible at all, so this test
    # overrides focal_px; every other test in the suite still uses the
    # canonical framing.
    camera = court_camera(focal_px=500.0)
    image, _truth = render_court(camera)

    # A y comfortably inside [SHORT_LINE_FROM_FRONT_FT, SERVICE_BOX_BACK_FT]
    # so the sample point sits on the box's inner side line, away from its
    # front/back ends.
    y_test = 20.0
    quarter_width = LINE_WIDTH_FT / 4.0

    def pixel_at(x_ft):
        px, py = camera.project((x_ft, y_test, 0.0))
        return image[int(round(py)), int(round(px))]

    # Correct band: [CENTER - half, CENTER + half]. Paint must appear both
    # at the centreline and on either side of it, within the band.
    assert np.allclose(pixel_at(LEFT_BOX_INNER_CENTER_X_FT), LINE_BGR, atol=2)
    assert np.allclose(
        pixel_at(LEFT_BOX_INNER_CENTER_X_FT - quarter_width), LINE_BGR, atol=2)

    # The buggy version straddled SERVICE_BOX_FT instead, whose band ends
    # exactly at LEFT_BOX_INNER_CENTER_X_FT (SERVICE_BOX_FT + half). A point
    # a quarter line-width further out is inside the correct band but past
    # the end of the buggy one, so it is bare floor under the bug and paint
    # once fixed.
    assert np.allclose(
        pixel_at(LEFT_BOX_INNER_CENTER_X_FT + quarter_width), LINE_BGR, atol=2)


def _frames_with_moving_player(base, count=5):
    """Static court, one dark rectangle that moves — the real occlusion case."""
    frames = []
    for index in range(count):
        frame = base.copy()
        left = 300 + index * 180
        frame[500:900, left:left + 150] = (30, 30, 35)
        frames.append(frame)
    return frames


def test_median_frame_erases_a_moving_player():
    camera = court_camera()
    base, _ = render_court(camera)
    frames = _frames_with_moving_player(base)
    median, moved = court_detect.median_frame(frames)

    assert moved is False

    # A bare "brightness floor" check over this crop is not specific enough:
    # LINE_BGR = (90, 45, 30) paint sits in this region and its minimum
    # channel (30) is identical to the player patch's minimum channel (30 in
    # (30, 30, 35)), so `.min() > threshold` would fail on a clean render for
    # a reason that has nothing to do with the player. Instead compare the
    # swept crop directly to the same crop of the untouched, player-free
    # render (deterministic: render_court has no randomness at noise_sigma=0).
    # Each column is covered by the player in exactly one of the five
    # non-overlapping sweeps, so the per-pixel temporal median always picks
    # one of the four untouched samples and should reproduce the clean crop
    # exactly.
    crop_median = median[500:900, 300:1350]
    crop_clean = base[500:900, 300:1350]
    assert np.array_equal(crop_median, crop_clean)


def test_median_frame_flags_a_panning_camera():
    camera = court_camera()

    # A bare noise-free render will not do here: it is two huge flat-colour
    # slabs (wall, floor) plus a few thin lines, and most full-width rows of
    # the wall slab are one uniform colour end to end. np.roll along either
    # axis maps those uniform runs onto themselves, so no amount of shift
    # ever moves more than ~15% of pixels past MOTION_DELTA -- the fixture
    # would fail to prove a real pan regardless of the implementation under
    # test. noise_sigma is NOT a stand-in for real sensor/lens noise -- it is
    # fixed per-pixel texture painted onto the render, which np.roll then
    # displaces, so the fraction of changed pixels reflects the roll instead
    # of the render's artificial flatness. This only proves the fixture can
    # cross MOTION_FRACTION at a noise level (40.0) 20x this suite's normal
    # 2.0; it says nothing about whether 0.25 is the right threshold for real
    # footage. That validation lives in
    # test_median_frame_flags_a_real_pan_on_squashanalytics and
    # test_median_frame_does_not_flag_the_bay_club_fixed_mount below, which
    # measure actual video and bracket 0.25 with wide margins on both sides.
    base, _ = render_court(camera, noise_sigma=40.0, seed=0)
    frames = [np.roll(base, shift * 90, axis=1) for shift in range(5)]

    _, moved = court_detect.median_frame(frames)

    assert moved is True


def _sample_frames(path, count=5):
    """Seek `count` frames spread evenly across a video, cv2.VideoCapture-only.

    Mirrors the real caller (design spec §7: "the client posts 5 JPEG
    frames, sampled across the clip using the seek machinery"). Only the
    `count` target frames are decoded -- never the whole video.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fracs = [i / (count - 1) for i in range(count)]
        frames = []
        for frac in fracs:
            target = min(int(round(frac * (total - 1))), total - 1)
            # Some containers (.mov) fail to decode the exact last frame via
            # POS_FRAMES seeking; step back a few frames until one decodes.
            ok = False
            for backoff in range(10):
                probe = max(target - backoff, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, probe)
                ok, frame = cap.read()
                if ok:
                    break
            if not ok:
                raise RuntimeError(f"failed to read a frame near {target} in {path}")
            frames.append(frame)
        return frames
    finally:
        cap.release()


def test_median_frame_flags_a_real_pan_on_squashanalytics():
    """SquashAnalytics.mp4 (in repo) genuinely pans -- this is the real-frame
    smoke test promised by the auto-court-detection design spec §11.

    Measured 2026-07-27 with `.venv/bin/python`, sampling 5 frames spread
    evenly across the clip's 18713 frames (indices 0, 4678, 9356, 14034,
    18712): mean changed-pixel fraction 0.63, comfortably above
    MOTION_FRACTION (0.25). A second sampling at fractions
    [0.1, 0.3, 0.5, 0.7, 0.9] measured 0.31 -- still clear of the threshold,
    so the result isn't an artifact of which frames get picked.
    """
    frames = _sample_frames(SQUASH_ANALYTICS_MP4)

    _, moved = court_detect.median_frame(frames)

    assert moved is True


@pytest.mark.skipif(
    not BAY_CLUB_MOV.exists(),
    reason="Bay Club footage lives outside the repo (real fixed-mount test asset)",
)
def test_median_frame_does_not_flag_the_bay_club_fixed_mount():
    """The false-positive side of the guard: real footage from the product's
    actual fixed fin mount must not be flagged as a pan.

    Measured 2026-07-27, same 5-frames-evenly-spread sampling as the
    SquashAnalytics test above (indices 0, 127, 254, 381, 502 of 509 total
    frames): mean changed-pixel fraction 0.12, comfortably below
    MOTION_FRACTION (0.25). A second sampling at fractions
    [0.1, 0.3, 0.5, 0.7, 0.9] measured 0.10 -- consistently well clear of the
    threshold. Together with the SquashAnalytics measurement above, 0.25
    separates real panning from a real fixed mount by a wide margin on both
    sides (~0.13-0.15 below across samplings, ~0.06-0.38 above).
    """
    frames = _sample_frames(BAY_CLUB_MOV)

    _, moved = court_detect.median_frame(frames)

    assert moved is False


def _nearest_line_to(lines, point_a, point_b):
    """The detected line whose ends best match a truth segment's ends."""
    def error(line):
        forward = (np.hypot(line.x1 - point_a[0], line.y1 - point_a[1])
                   + np.hypot(line.x2 - point_b[0], line.y2 - point_b[1]))
        backward = (np.hypot(line.x1 - point_b[0], line.y1 - point_b[1])
                    + np.hypot(line.x2 - point_a[0], line.y2 - point_a[1]))
        return min(forward, backward)
    return min(lines, key=error)


def test_find_lines_recovers_the_front_wall_paint():
    camera = court_camera()
    image, truth = render_court(camera, noise_sigma=2.0)

    lines = court_detect.find_lines(court_detect.line_response(image),
                                    min_length_px=image.shape[1] * 0.10)

    for name in ("out_line_lower_edge", "service_line_top_edge", "tin_top_edge"):
        start, end = truth[name]
        found = _nearest_line_to(lines, start, end)
        # The stripe is ~LINE_WIDTH_FT wide, so a line fitted to the whole
        # stripe may sit up to half a width off the named datum. Half
        # LINE_WIDTH_FT at this camera scale measures ~4.4 px, and the three
        # named lines measure 3.1-4.1 px here -- Task 5 pulls it onto the
        # exact edge; here we only require the right stripe.
        assert abs(found.y_at(start[0]) - start[1]) < 6, name


def test_edge_response_finds_the_wall_floor_seams():
    camera = court_camera()
    image, truth = render_court(camera, noise_sigma=2.0)

    lines = court_detect.find_lines(court_detect.edge_response(image),
                                    min_length_px=image.shape[1] * 0.10)

    start, end = truth["front_seam"]
    found = _nearest_line_to(lines, start, end)
    assert abs(found.y_at(start[0]) - start[1]) < 8


def test_find_lines_merges_a_broken_near_vertical_line_into_one():
    """find_lines-level regression test for the merge-frame bug: the
    half-court line and service-box side lines are near-vertical court
    features that get interrupted by players and glare like any other line.
    Real HoughLinesP output for a near-vertical line already exhibits a
    direction sign flip between different fragments -- verified empirically
    here with nothing more than a 4 px total tilt over a 700 px run -- so a
    merge rule that canonicalises each fragment on its own (rather than in
    its comparison group's frame) under-merges this broken line into two
    DetectedLines instead of one.
    """
    height, width = 900, 400
    y_top, y_bottom = 100, 800
    x_top, x_bottom = 200.0, 196.0
    gaps = ((300, 340), (500, 540))

    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(y_top, y_bottom):
        if any(lo <= y < hi for lo, hi in gaps):
            continue
        fraction = (y - y_top) / (y_bottom - y_top)
        x = int(round(x_top + (x_bottom - x_top) * fraction))
        cv2.line(image, (x, y), (x, y), 255, thickness=1)

    lines = court_detect.find_lines(image, min_length_px=width * 0.10)

    assert len(lines) == 1
    (line,) = lines
    # Spans most of the drawn extent (measured ~678 px of the 700 px run)
    # despite the two gaps, not just one fragment's worth (~160 px).
    assert abs(line.y1 - line.y2) > 600


def test_find_lines_merges_a_broken_near_horizontal_line_into_one():
    """The near-HORIZONTAL analogue of the test above -- the orientation that
    matters in this app, since every real court line here (out line, service
    line, tin, front seam, short line, box backs) is near-horizontal.

    This is the case a first attempted fix broke: it canonicalised each
    fragment's own direction into the upper half-plane (`uy >= 0`) before
    taking its offset. That does not remove the sign-flip boundary, it only
    relocates it -- the original bug wrapped the angle and put the flip at
    near-vertical (dormant for every line in this app); the first fix put it
    at near-horizontal (live for every line in this app). A 200-trial
    measurement against realistic near-horizontal broken lines found the
    original code over-splitting in 61/200 trials (offset gap 6-16 px) and
    the first fix over-splitting in 125/200 (offset gap 6-813 px, median
    804 px) -- strictly worse. The correct fix never canonicalises a
    fragment on its own: it aligns each fragment to the direction of the
    group it is being compared against, so there is no boundary anywhere.

    The roll here (~0.06 degrees) is small but deliberately nonzero: an
    exactly-horizontal fixture has dy identically 0 for every fragment, which
    can never produce the sign flip that this bug depends on.

    Finding parameters that discriminate took a search, the same way the
    near-vertical test above did originally: at thickness=1 this exact shape
    happens to survive even the first fix, because a single-pixel-wide
    near-horizontal line rasterises to long flat (dy=0) runs between rounding
    steps and HoughLinesP's own fragments rarely straddle the flip. A 2 px
    thickness produces enough sub-pixel variation in each fragment's fitted
    local slope to reproduce the flip reliably. Swept roll from 0.01 to 0.2
    degrees in 0.01 steps at this thickness/gap configuration: 0.05-0.06
    degrees sit in a noisy transitional zone where even the correct fix
    fragments into 2-3 groups (an unrelated Hough-sampling artefact, not this
    bug -- see the property test below for why that range is avoided there
    too), 0.10 degrees happens to still merge under all three versions, and
    every other value in the sweep reproduces the target split (original: 1
    group, this fix: 1 group, first fix: 2 groups). 0.08 sits mid-range in
    that split with margin on both sides.
    """
    height, width = 400, 900
    x_left, x_right = 100, 800
    y_at_left = 200.0
    roll_deg = 0.08
    slope = math.tan(math.radians(roll_deg))
    gaps = ((300, 340), (500, 540))

    image = np.zeros((height, width), dtype=np.uint8)
    for x in range(x_left, x_right):
        if any(lo <= x < hi for lo, hi in gaps):
            continue
        y = int(round(y_at_left + slope * (x - x_left)))
        cv2.line(image, (x, y), (x, y), 255, thickness=2)

    lines = court_detect.find_lines(image, min_length_px=width * 0.10)

    assert len(lines) == 1
    (line,) = lines
    # Spans most of the drawn extent (the 700 px run minus the two gaps)
    # rather than just one fragment's worth.
    assert abs(line.x1 - line.x2) > 600


def test_find_lines_merges_broken_lines_across_orientations_and_noise():
    """Randomized property test for the merge-frame fix.

    Neither prior version survives being checked at BOTH orientations across
    a spread of rolls and pixel noise: the original code's sign-flip
    boundary sits at near-vertical, the first fix's sits at near-horizontal,
    and each version fails roughly half of these trials depending on which
    orientation they land on. Only aligning each fragment to its comparison
    group's frame (rather than canonicalising either fragment on its own)
    has no boundary for either orientation to land on.

    Deterministic (fixed per-trial seeds via `numpy.random.default_rng`) so
    a failure is reproducible. Roll and noise are kept in a realistic
    near-axis range (a court line photographed close to level, with a few
    tenths of a pixel of edge/detection jitter) -- pushed much past this
    (e.g. +-0.5 degrees roll or 0.8 px noise), a single Hough fragment can
    legitimately land outside MERGE_OFFSET_PX on its own local fit even
    under a correct merge rule, which would make the assertion flaky for a
    reason unrelated to the bug this test targets.
    """
    trials_per_orientation = 50
    span, margin, cross = 700, 100, 400
    gaps = ((300, 340),)
    min_length_px = cross * 0.10

    def make_image(orientation, seed):
        rng = np.random.default_rng(seed)
        roll_deg = rng.uniform(-0.2, 0.2)
        noise_sigma = rng.uniform(0.0, 0.3)
        slope = math.tan(math.radians(roll_deg))
        base = cross / 2.0
        length = span + 2 * margin
        shape = (cross, length) if orientation == "horizontal" else (length, cross)
        image = np.zeros(shape, dtype=np.uint8)
        for offset in range(span):
            if any(lo <= offset < hi for lo, hi in gaps):
                continue
            along = margin + offset
            jitter = float(rng.normal(0.0, noise_sigma)) if noise_sigma > 0 else 0.0
            coord = int(round(base + slope * offset + jitter))
            if not (0 <= coord < cross):
                continue
            point = (along, coord) if orientation == "horizontal" else (coord, along)
            cv2.line(image, point, point, 255, thickness=1)
        return image

    failures = []
    for orientation in ("vertical", "horizontal"):
        for trial in range(trials_per_orientation):
            seed = (0 if orientation == "vertical" else 1) * 100_000 + trial
            image = make_image(orientation, seed)
            lines = court_detect.find_lines(image, min_length_px=min_length_px)
            if len(lines) != 1:
                failures.append((orientation, trial, seed, len(lines)))

    total = 2 * trials_per_orientation
    assert not failures, (
        f"{len(failures)}/{total} trials failed to merge into a single "
        f"group: {failures[:10]}"
    )


def test_detected_line_intersect_returns_none_for_parallels():
    first = court_detect.DetectedLine(0.0, 0.0, 100.0, 0.0, support=1)
    second = court_detect.DetectedLine(0.0, 50.0, 100.0, 50.0, support=1)
    assert first.intersect(second) is None

    crossing = court_detect.DetectedLine(50.0, -50.0, 50.0, 50.0, support=1)
    point = first.intersect(crossing)
    assert point is not None and abs(point[0] - 50.0) < 1e-6


# --- geometry budgets for test_assign_lines_names_every_required_entity ----
#
# A stripe fitted by find_lines can sit up to half its paint width off the
# datum EDGE stored in `truth` (e.g. the "out" line's fitted centre sits half
# a line-width below the LOWER edge that datum names). At court_camera's
# focal_px=700.0 scale, half of LINE_WIDTH_FT projects to ~1.9-1.95 px on the
# front wall (camera.project of a LINE_WIDTH_FT-tall band at the out/
# service/tin heights). A 30-seed sweep at this test's noise_sigma=2.0
# (varying only the render's per-pixel noise seed, everything else held
# fixed) measured actual assign_lines-vs-truth error up to 2.13 px worst
# case (the service line) -- consistent with that half-width bound plus a
# little Hough/threshold fitting jitter. 3.0 px clears the measured worst
# case with ~40% headroom, and is still far inside the tightest gap between
# any two of these five entities' true y-positions at this camera scale --
# tin to front_seam, ~36.7 px apart -- so the tolerance window can never
# reach a neighbouring line.
NAMED_LINE_TOLERANCE_PX = 3.0

# left_seam/right_seam have no truth key, so they are checked against the
# camera model directly instead: the true angle of the world edge each seam
# traces (x=0 or x=COURT_WIDTH_FT, from the front wall to render_court's
# default visible_depth_ft=26.0), computed the same way `truth`'s own
# entries are. The same 30-seed sweep found the assigned seam's angle
# within 0.4 degrees of that true value. 5 degrees leaves over 10x that
# margin while staying far short of the ~56-degree gap to a near-vertical
# false read (e.g. a half-court or service-box side line near 90 degrees)
# and clear of the 25-degree floor HORIZONTAL_MAX_DEG already guarantees --
# so this is a real "plausible diagonal band" check, not a restatement of
# the >25-degree filter.
SEAM_ANGLE_TOLERANCE_DEG = 5.0

# The same sweep measured up to 4.2 px of noise-driven scatter in where
# front_seam.intersect(left_seam/right_seam) lands versus the true front
# corner. 8 px doubles that headroom and is tiny next to the ~488 px
# separation between the two front corners at this camera scale, so it
# cannot mistake "near the wrong corner" for "near the right one".
SEAM_CORNER_TOLERANCE_PX = 8.0

# render_court's default -- not overridden by the test below.
_VISIBLE_DEPTH_FT = 26.0


def _assert_assignment_matches_truth(assigned, truth, camera, frame_shape):
    """Score `assigned` against the camera/geometry that produced `truth`.

    Factored out of the test so the "is this actually discriminating"
    check (see test_assign_lines_assignment_check_rejects_bad_assignments
    below) can run the exact same assertions against hand-corrupted
    `assigned` dicts.
    """
    for name in ("out", "service", "tin", "front_seam",
                 "left_seam", "right_seam", "short_line"):
        assert assigned.get(name) is not None, f"missing {name}"

    # The five horizontal entities must sit where the camera actually put
    # them, not merely in the right relative order.
    for name, key in (("out", "out_line_lower_edge"),
                       ("service", "service_line_top_edge"),
                       ("tin", "tin_top_edge"),
                       ("front_seam", "front_seam"),
                       ("short_line", "short_line")):
        line = assigned[name]
        (x1, y1), (x2, y2) = truth[key]
        for x, true_y in ((x1, y1), (x2, y2)):
            found_y = line.y_at(x)
            assert found_y is not None, f"{name} is vertical at x={x}"
            assert abs(found_y - true_y) < NAMED_LINE_TOLERANCE_PX, (
                name, found_y, true_y)

    # The seams are on the sides they are named for (guaranteed by
    # assign_lines' own candidate filter, kept as a sanity check).
    left, right = assigned["left_seam"], assigned["right_seam"]
    front_seam = assigned["front_seam"]
    assert left.midpoint[0] < frame_shape[1] / 2
    assert right.midpoint[0] > frame_shape[1] / 2
    assert left.angle_deg < 0
    assert right.angle_deg > 0

    # Angle magnitude: matched against the camera's own projection of the
    # world edge each seam traces, not a guessed band.
    def edge_angle_deg(x_ft):
        near = camera.project((x_ft, 0.0, 0.0))
        far = camera.project((x_ft, _VISIBLE_DEPTH_FT, 0.0))
        return math.degrees(math.atan2(far[1] - near[1], far[0] - near[0]))

    expected_mag = abs(edge_angle_deg(0.0))
    assert abs(abs(left.angle_deg) - expected_mag) < SEAM_ANGLE_TOLERANCE_DEG, (
        left.angle_deg, expected_mag)
    assert abs(abs(right.angle_deg) - expected_mag) < SEAM_ANGLE_TOLERANCE_DEG, (
        right.angle_deg, expected_mag)

    # Each seam must actually cross the front seam near a real front
    # corner, and the two seams must land near DIFFERENT corners -- "near
    # opposite ends of the court", not "both near the same spot" or
    # "anywhere along the line".
    corners = [truth["wall_bottom_left"], truth["wall_bottom_right"]]

    def nearest_corner(point):
        dists = [math.hypot(point[0] - c[0], point[1] - c[1]) for c in corners]
        index = 0 if dists[0] < dists[1] else 1
        return index, dists[index]

    left_hit = front_seam.intersect(left)
    right_hit = front_seam.intersect(right)
    assert left_hit is not None, "left_seam is parallel to front_seam"
    assert right_hit is not None, "right_seam is parallel to front_seam"

    left_corner, left_dist = nearest_corner(left_hit)
    right_corner, right_dist = nearest_corner(right_hit)
    assert left_dist < SEAM_CORNER_TOLERANCE_PX, (left_hit, corners)
    assert right_dist < SEAM_CORNER_TOLERANCE_PX, (right_hit, corners)
    assert left_corner != right_corner, "both seams hit the same front corner"


def test_assign_lines_names_every_required_entity():
    # court_camera()'s default framing (focal_px=1600) puts the short line
    # (y ~= 17.9 ft) at pixel row ~1430 in a 1080-tall frame -- off-screen,
    # per the same framing limit the service-box test above documents for
    # y >= ~10.5 ft. No short-line paint is rendered at all at the default
    # focal length, so assign_lines could never find it regardless of the
    # heuristic. A wider FOV is required, but 500.0 (what the service-box
    # test uses) is too wide here: it also brings the service box BACK line
    # into frame, and that line's two halves (either side of the boxes) sit
    # on the same infinite line, so find_lines' collinear merge -- which by
    # design ignores gaps, so a line can survive real occlusion -- fuses
    # them into one line spanning nearly the full frame width. That merged
    # line is longer than the true short line, so assign_lines' "widest
    # below-seam horizontal" heuristic would pick it instead. (It happens to
    # sit even lower in the image than the real short line, so the ordering
    # assertion below would still pass with the wrong line selected -- a
    # "right for the wrong reason" trap worth avoiding rather than shipping
    # unnoticed.) 700.0 clears the short line (measured row ~930) with
    # margin while keeping the box-back line's row (~1250) safely below the
    # visible frame, so only the real short line is a candidate.
    camera = court_camera(focal_px=700.0)
    image, truth = render_court(camera, noise_sigma=2.0)
    minimum = image.shape[1] * 0.10

    assigned = court_detect.assign_lines(
        court_detect.find_lines(court_detect.line_response(image), minimum),
        court_detect.find_lines(court_detect.edge_response(image), minimum),
        image.shape,
    )

    _assert_assignment_matches_truth(assigned, truth, camera, image.shape)
