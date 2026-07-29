"""Per-player movement in court feet -- tier 2 of the analysis ladder.

Distance covered, speed, time on the T, front/back balance and a floor
heatmap: the numbers a player would actually change their training over. They
need a solved court and two person tracks, and nothing at all from the ball,
which is why this tier survives on footage the ball tier is gated off.

The module is pure over court-space samples. Pixel-to-feet conversion is
`to_court_samples`, one function, so the homography is applied in exactly one
place and the statistics can be tested in feet without a video. (The plan's
signature took TrackSamples directly; splitting the conversion out keeps the
stats testable and stops court feet and image pixels sharing a variable name,
which is the mistake that makes a distance silently wrong by a scale factor.)

Two things get more care than their size suggests, because getting them wrong
is invisible in the output:

  * Samples outside a rally are excluded. Walking over to pick up the ball is
    not court movement, and counting it inflates every distance on the page.
  * Coasted samples -- positions the tracker carried forward while the
    detector saw nothing -- reduce `sample_coverage` rather than being
    averaged in as observations. A distance computed from a quarter of a rally
    looks exactly like one computed from all of it.
"""

from collections import namedtuple

from court_model import (
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    HALF_COURT_X_FT,
    SHORT_LINE_CENTER_Y_FT,
)

# A player is "on the T" within this radius of it. Six feet is roughly a lunge
# and recovery -- the distance from which the T is still being held rather than
# merely passed through.
T_RADIUS_FT = 6.0

# Fastest a squash player plausibly moves. Above this the cause is a tracker
# identity swap, not an athlete: one swapped frame teleports a player across
# the court and, unclamped, becomes a 200 ft/s "sprint" that quietly drags the
# average. Clamping keeps a tracking error from becoming a movement claim.
SPEED_MAX_FTPS = 30.0

# Positions are smoothed over this CENTERED window before speed is
# differenced. Detection jitter of a few pixels per frame is a large apparent
# speed once divided by a 60th of a second.
#
# The width is measured, not assumed. Mean absolute error in total distance
# over a synthetic 40 ft shuttle, 20 seeds per cell, positional jitter in feet:
#
#     jitter | none   | 0.2s   | 0.4s
#      0.00  |   0.0% |   7.5% |  17.0%
#      0.10  |   2.2% |   7.0% |  16.6%
#      0.25  |   9.7% |   5.8% |  15.9%
#      0.50  |  35.5% |   3.0% |  14.1%
#      1.00  | 104.3% |  10.4% |   8.4%
#
# Both directions of the error are invisible in a report. Unsmoothed inflates
# distance with path nobody ran, and it diverges -- jitter is a random walk, so
# the error has no ceiling (104% at a foot of jitter). Smoothing rounds the
# corners off instead, and in squash the direction changes ARE the movement, so
# that error is a systematic UNDER-count. 0.2s is the width whose worst case
# across the whole range is smallest.
#
# Two known biases, both real and neither yet corrected:
#   * ~7.5% under-count on sharp direction changes with clean input.
#   * The design plan proposed 0.5s, which costs ~17% -- worse everywhere that
#     matters. Not adopted.
SMOOTH_WINDOW_S = 0.2

# Longest gap between consecutive samples that still counts as continuously
# observed, when computing sample_coverage. Separate from SMOOTH_WINDOW_S even
# though they were once the same number: they answer different questions, and
# tuning the smoothing width should not silently move the coverage figure.
COVERAGE_MAX_GAP_S = 0.5

# Heatmap resolution. 7x8 puts a cell at roughly 3 ft x 4 ft -- coarse enough
# that a few samples do not read as a hotspot, fine enough to separate the
# service boxes from the back corners.
GRID_X, GRID_Y = 7, 8

# The T, from court_model's landmark table rather than a literal.
T_POINT_FT = (HALF_COURT_X_FT, SHORT_LINE_CENTER_Y_FT)

# Front/back split. The short line is the natural divider and sits at ~17.9 ft;
# 16.0 is the design spec's value, a little in front of it, which puts the
# service boxes in "back" where a coach would put them.
FRONT_BACK_SPLIT_FT = 16.0

CourtSample = namedtuple("CourtSample", "t_s x_ft y_ft coasted")


def to_court_samples(track_samples, floor_map):
    """TrackSamples (image pixels) -> CourtSamples (court feet).

    Returns [] when there is no floor map: without a solved court there are no
    feet to report, and reporting pixels as feet is the single worst thing
    this module could do.
    """
    if floor_map is None:
        return []

    court = []
    for sample in track_samples:
        try:
            x_ft, y_ft = floor_map.image_to_court(*sample.foot_px)
        except Exception:
            # A point the homography cannot map (behind the camera plane,
            # degenerate) is dropped rather than clamped: a clamped position
            # is a fabricated one, and it would land on the court edge and
            # read as a real visit to the wall.
            continue
        court.append(
            CourtSample(float(sample.t_s), float(x_ft), float(y_ft),
                        bool(sample.coasted))
        )
    return court


def _within_rallies(samples, rallies):
    if not rallies:
        return []
    spans = [(float(r["start_s"]), float(r["end_s"])) for r in rallies]
    return [
        sample for sample in samples
        if any(start <= sample.t_s <= end for start, end in spans)
    ]


def _smoothed(samples, window_s=SMOOTH_WINDOW_S):
    """Positions averaged over a CENTERED window, in sample order.

    Speed is a difference of positions, so a few pixels of detection jitter
    becomes a large apparent speed once divided by a frame interval. Smoothing
    before differencing is what makes the speed a movement measure rather than
    a noise measure.

    Centered rather than trailing: a trailing window lags the true position by
    half its width, and that lag is what turns a change of direction into a
    cut corner. See the measurement table at SMOOTH_WINDOW_S.
    """
    ordered = sorted(samples, key=lambda sample: sample.t_s)
    # Sample times accumulate float error (1.1 arrives as 1.1000000000000003),
    # so an exact comparison against the window edge admits two samples at one
    # timestamp and three at the next. The window then varies in width from
    # sample to sample, which shows up as jitter in the very measure the
    # smoothing exists to remove. The epsilon makes the edge deterministic.
    half = window_s / 2.0 + 1e-9
    smoothed = []
    start = 0
    for index, sample in enumerate(ordered):
        while ordered[start].t_s < sample.t_s - half:
            start += 1
        end = index
        while end + 1 < len(ordered) and ordered[end + 1].t_s <= sample.t_s + half:
            end += 1
        window = ordered[start:end + 1] or [sample]
        smoothed.append(CourtSample(
            sample.t_s,
            sum(other.x_ft for other in window) / len(window),
            sum(other.y_ft for other in window) / len(window),
            sample.coasted,
        ))
    return smoothed


def _empty_heatmap():
    return [[0.0] * GRID_X for _ in range(GRID_Y)]


def _heatmap(samples):
    grid = _empty_heatmap()
    if not samples:
        return grid

    for sample in samples:
        column = min(GRID_X - 1, max(0, int(sample.x_ft / COURT_WIDTH_FT * GRID_X)))
        row = min(GRID_Y - 1, max(0, int(sample.y_ft / COURT_LENGTH_FT * GRID_Y)))
        grid[row][column] += 1.0

    total = sum(sum(row) for row in grid)
    return [[value / total for value in row] for row in grid]


def _rally_seconds(rallies):
    return sum(
        max(0.0, float(r["end_s"]) - float(r["start_s"])) for r in (rallies or [])
    )


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def movement_stats(samples, rallies):
    """Movement statistics for one player over the rallies of a run.

    `samples` are CourtSamples; `rallies` are `{"start_s","end_s"}` spans from
    the rally timeline. Everything is scoped to rally time, so the numbers
    describe play rather than the whole recording.
    """
    in_play = _within_rallies(samples, rallies)
    observed = [sample for sample in in_play if not sample.coasted]

    empty = {
        "distance_ft": 0.0,
        "avg_speed_ftps": 0.0,
        "p95_speed_ftps": 0.0,
        "t_time_pct": 0.0,
        "front_pct": 0.0,
        "back_pct": 0.0,
        "heatmap": _empty_heatmap(),
        "sample_coverage": 0.0,
    }
    if not observed:
        return empty

    smoothed = _smoothed(observed)

    distance = 0.0
    speeds = []
    for earlier, later in zip(smoothed, smoothed[1:]):
        elapsed = later.t_s - earlier.t_s
        if elapsed <= 0:
            continue
        step = ((later.x_ft - earlier.x_ft) ** 2
                + (later.y_ft - earlier.y_ft) ** 2) ** 0.5
        speed = min(step / elapsed, SPEED_MAX_FTPS)
        # Distance is recomputed from the clamped speed so a tracker identity
        # swap cannot contribute a teleport's worth of "running" either.
        distance += speed * elapsed
        speeds.append(speed)

    on_t = sum(
        1 for sample in smoothed
        if ((sample.x_ft - T_POINT_FT[0]) ** 2
            + (sample.y_ft - T_POINT_FT[1]) ** 2) ** 0.5 <= T_RADIUS_FT
    )
    front = sum(1 for sample in smoothed if sample.y_ft < FRONT_BACK_SPLIT_FT)

    rally_seconds = _rally_seconds(rallies)
    observed_seconds = sum(
        later.t_s - earlier.t_s
        for earlier, later in zip(smoothed, smoothed[1:])
        if 0 < later.t_s - earlier.t_s <= COVERAGE_MAX_GAP_S
    )

    return {
        "distance_ft": distance,
        "avg_speed_ftps": sum(speeds) / len(speeds) if speeds else 0.0,
        "p95_speed_ftps": _percentile(speeds, 0.95) if speeds else 0.0,
        "t_time_pct": on_t / len(smoothed),
        "front_pct": front / len(smoothed),
        "back_pct": (len(smoothed) - front) / len(smoothed),
        "heatmap": _heatmap(smoothed),
        "sample_coverage": (
            min(1.0, observed_seconds / rally_seconds) if rally_seconds else 0.0
        ),
    }
