"""Rally boundaries from audio impacts and frame motion, with no ball.

This is the bottom rung of the analysis ladder, and the reason the ladder
exists. Rallies were segmented from front-wall hits, so rally counts, lengths,
tempo and work:rest all inherited the ball detector's recall -- near 35% on the
standing baseline. A rally count computed from a third of the events reads
exactly as plausible as a correct one, which in a coaching product is worse
than a missed line call: a missed call is visibly missing, a wrong number is
not.

Audio transients and frame-motion energy are available on any clip at any
frame rate, including the 30 fps camera-roll footage where per-frame ball
detection is hopeless. So this module deliberately imports nothing from the
pipeline -- not job_runner, not the detectors. A test asserts that. The moment
this module *can* read a hit, something will make it, and tier 1 silently
re-acquires the problem it was built to escape.

Segmentation itself is a pure function of two time series, which is what makes
the thresholds testable without video. Two things sit alongside it:
`motion_energy_step`, which produces one of those series from frames, and
`build_rally_timeline`, which is allowed to *compare* the result against
hit-derived rallies passed in as plain data. Reconciliation lives there
precisely so the segmenting stays unable to see a ball.
"""

import statistics

import cv2
import numpy as np

# A rally shorter than this is a knock-up, a let, or a door closing. Two
# seconds is roughly the shortest real exchange -- serve plus one return --
# and is the value the design spec fixes.
MIN_RALLY_S = 2.0

# Fallback gap between rallies when the impact spacing gives no clear answer.
# Mirrors job_runner.DEFAULT_RALLY_GAP_SECONDS; kept as a literal rather than
# imported because importing it would drag the ball pipeline in here.
DEFAULT_GAP_S = 5.0

# No rally boundary is ever inferred below this. Squash rallies contain pauses
# -- a let, a slow serve routine -- and a threshold under a few seconds splits
# one rally into several. Mirrors job_runner.RALLY_GAP_MIN_SPLIT_SECONDS.
MIN_GAP_S = 4.0

# How much bigger a gap must be than the one below it to read as a rally
# boundary rather than a slow shot. Mirrors job_runner.RALLY_GAP_RATIO_SPLIT.
GAP_RATIO_SPLIT = 2.0

# Motion is "active" above median + this many MADs. MAD rather than standard
# deviation because the rallies themselves are the outliers: with a third of
# the clip in play, the standard deviation is inflated by the very signal
# being detected, and the threshold drifts up until it finds nothing.
MOTION_MAD_MULTIPLIER = 3.0

# Rallies are padded by this much on each side. The first sound of a rally is
# the serve contact, which is preceded by the toss and the step in; the last
# is the final contact, followed by the ball dying. Half a second recovers
# both without merging neighbours.
RALLY_PAD_S = 0.5

# A motion-only rally is real but unverified -- something moved for long
# enough, with nothing corroborating that it was play. It is reported at a
# fixed low confidence rather than hidden, per Principle 3.
MOTION_ONLY_CONFIDENCE = 0.3

# Impacts needed before a rally is called fully confident. Four contacts is a
# serve, a return, and a reply -- the point at which "someone hit something
# twice" becomes "a rally happened".
CONFIDENT_IMPACT_COUNT = 4

# Frames are downscaled to this width before differencing. Motion energy must
# mean the same thing at 1080p and 4K, or the same match filmed on two phones
# segments into different rallies; a fixed working width is what makes the
# threshold comparable across clips. It also makes the difference cheap enough
# to run on every decoded frame.
MOTION_WORK_WIDTH = 160

# Active motion spans separated by less than this are one span. Real motion
# energy is spiky rather than a plateau: measured on the repo's own 5-minute
# clip, 12.3% of samples cleared the threshold but as 98 fragments with a
# MEDIAN duration of 0.00 s and a longest of 1.83 s, so requiring strictly
# contiguous samples found zero rallies in a real match. A rally is *sustained*
# activity -- active most of the time, not at every sample.
#
# The value has to sit well under MIN_GAP_S: bridging must never reach across
# a genuine between-points pause and fuse two rallies, which would silently
# halve the rally count. Half of MIN_GAP_S is the ceiling, and a test asserts
# it stays there.
MOTION_BRIDGE_S = 1.5


def infer_gap_seconds(impact_times):
    """The pause length that separates rallies, inferred from the impacts.

    Sorted inter-impact gaps in squash are bimodal: shot-to-shot gaps cluster
    under a second or two, and rally-to-rally gaps sit far above. The largest
    *ratio* jump between consecutive sorted gaps is therefore the boundary
    between the two populations, and it adapts to a fast rally pace or a slow
    one without a per-clip setting.

    Falls back to DEFAULT_GAP_S when no jump is convincing, which is the case
    for a clip holding a single rally -- there is no second population to find.
    """
    ordered = sorted(impact_times)
    gaps = sorted(
        later - earlier
        for earlier, later in zip(ordered, ordered[1:])
        if later - earlier > 0
    )

    best = None
    for lower, upper in zip(gaps, gaps[1:]):
        if upper < MIN_GAP_S or upper / lower < GAP_RATIO_SPLIT:
            continue
        if best is None or upper / lower > best[0]:
            best = (upper / lower, (lower + upper) / 2.0)

    if best is None:
        return DEFAULT_GAP_S
    return max(MIN_GAP_S, best[1])


def _cluster(impact_times, gap_s):
    """Impact times grouped into runs, split wherever the pause exceeds gap_s."""
    ordered = sorted(impact_times)
    if not ordered:
        return []

    runs = [[ordered[0]]]
    for time in ordered[1:]:
        if time - runs[-1][-1] > gap_s:
            runs.append([time])
        else:
            runs[-1].append(time)
    return runs


def _percentile(values, fraction):
    """Nearest-rank percentile. Small-n safe, unlike interpolating variants."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def motion_threshold(energies):
    """The energy above which a clip counts as being in play.

    Derived from the clip's own distribution, so it needs no knowledge of the
    camera, the lighting or the resolution -- only that a rally moves more
    than the same clip's idle time does.

    Median + 3*MAD is the robust default. MAD rather than standard deviation
    because the rallies are the outliers: with a third of the clip in play the
    standard deviation is inflated by the very signal being detected, and the
    threshold drifts up until it finds nothing.

    But MAD degenerates to exactly zero whenever more than half the samples
    sit at the median -- which here is the ordinary case, not a pathological
    one, because idle time dominates a match clip. When that happens the
    fallback is half the distance from the floor to the active level, which is
    the same "well clear of the noise" idea expressed with a statistic that
    does not collapse on a plateau.

    Returns None when nothing in the clip rises above its own floor. That is
    the honest answer for a still camera on an empty court: the alternative,
    thresholding at the median, makes the whole clip one long rally.
    """
    median = statistics.median(energies)
    peak = _percentile(energies, 0.95)
    if peak <= median:
        return None

    mad = statistics.median([abs(energy - median) for energy in energies])
    if mad > 0:
        return median + MOTION_MAD_MULTIPLIER * mad
    return median + (peak - median) / 2.0


def _active_motion_spans(motion):
    """Contiguous stretches where frame motion is above its own noise floor."""
    samples = [(float(t), float(e)) for t, e in motion]
    if len(samples) < 3:
        return []

    threshold = motion_threshold([energy for _, energy in samples])
    if threshold is None:
        return []

    spans = []
    span_start = None
    previous_time = samples[0][0]
    for time, energy in samples:
        if energy > threshold:
            if span_start is None:
                span_start = time
            previous_time = time
        elif span_start is not None:
            spans.append((span_start, previous_time))
            span_start = None
    if span_start is not None:
        spans.append((span_start, samples[-1][0]))

    return _bridge_short_gaps(spans, MOTION_BRIDGE_S)


def _bridge_short_gaps(spans, bridge_s):
    """Join spans separated by less than `bridge_s` into one.

    Without this a rally is only recognised if every single sample clears the
    threshold, which real footage never manages -- a player pausing, the ball
    out of frame, or the exposure settling all dip below it for a few tenths
    of a second, and the rally shatters into fragments too short to count.
    """
    if not spans:
        return []

    bridged = [spans[0]]
    for start, end in spans[1:]:
        if start - bridged[-1][1] <= bridge_s:
            bridged[-1] = (bridged[-1][0], max(bridged[-1][1], end))
        else:
            bridged.append((start, end))
    return bridged


def _overlaps(first, second):
    return first[0] <= second[1] and second[0] <= first[1]


def _merge_intervals(intervals):
    """Union of possibly-overlapping intervals, sorted."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def segment_rallies(
    impact_times_s,
    motion,
    duration_s,
    min_rally_s=MIN_RALLY_S,
    gap_s=None,
):
    """Rally spans for a clip, from impact times and motion energy.

    `impact_times_s` are audio transient times in seconds; `motion` is a list
    of `(time_s, energy)` with energy >= 0. Either may be empty -- a clip with
    no usable audio still segments on motion, and a clip whose camera never
    moves still segments on impacts. Requiring both would make tier 1 depend
    on two things when the design says it needs either.

    Returns dicts of `start_s`, `end_s`, `impact_count`, `source` and
    `confidence`, sorted and non-overlapping.
    """
    if not duration_s or duration_s <= 0:
        return []

    impacts = sorted(float(time) for time in (impact_times_s or []))
    if gap_s is None:
        gap_s = infer_gap_seconds(impacts)

    impact_runs = _cluster(impacts, gap_s)
    motion_spans = _active_motion_spans(motion or [])

    # Each impact run becomes a candidate span, extended by any motion it
    # overlaps -- motion knows when the rally really started, impacts know
    # that it was play.
    candidates = []
    claimed_motion = set()
    for run in impact_runs:
        start, end = run[0], run[-1]
        touched = [span for span in motion_spans if _overlaps(span, (start, end))]
        for span in touched:
            claimed_motion.add(span)
            start = min(start, span[0])
            end = max(end, span[1])
        candidates.append({
            "start_s": start,
            "end_s": end,
            "impact_count": len(run),
            "source": "audio+motion" if touched else "audio",
        })

    # Motion nobody's impacts claimed is still a rally, at low confidence: a
    # rally whose sounds the microphone missed is far more likely than a
    # six-second burst of court movement that was not play.
    for span in motion_spans:
        if span in claimed_motion:
            continue
        candidates.append({
            "start_s": span[0],
            "end_s": span[1],
            "impact_count": 0,
            "source": "motion",
        })

    kept = [
        candidate for candidate in candidates
        if candidate["end_s"] - candidate["start_s"] >= min_rally_s
    ]
    if not kept:
        return []

    kept.sort(key=lambda candidate: candidate["start_s"])

    # Pad, then clamp to the clip and to the neighbours. Clamping to
    # neighbours rather than merging keeps two rallies two rallies when the
    # padding would have run them together.
    rallies = []
    for index, candidate in enumerate(kept):
        previous_end = rallies[-1]["end_s"] if rallies else 0.0
        next_start = kept[index + 1]["start_s"] if index + 1 < len(kept) else duration_s

        start = max(0.0, previous_end, candidate["start_s"] - RALLY_PAD_S)
        end = min(duration_s, next_start, candidate["end_s"] + RALLY_PAD_S)
        if end <= start:
            continue

        if candidate["source"] == "motion":
            confidence = MOTION_ONLY_CONFIDENCE
        else:
            confidence = min(
                1.0, candidate["impact_count"] / CONFIDENT_IMPACT_COUNT
            )

        rallies.append({
            "start_s": start,
            "end_s": end,
            "impact_count": candidate["impact_count"],
            "source": candidate["source"],
            "confidence": confidence,
        })

    return rallies


# --- motion energy ----------------------------------------------------------
# The only part of this module that touches pixels. It stays here rather than
# in job_runner because it is the segmenter's input and nothing else consumes
# it; cv2 is not the ball pipeline, and the import test above is about the
# pipeline, not about images.


def motion_energy_step(previous_small, frame_bgr):
    """One frame's motion energy against its predecessor.

    Returns `(small, energy)`: the downscaled grayscale frame to carry into the
    next call, and the mean absolute difference from the previous one.

    Energy is 0.0 when there is no predecessor. Seeding with the frame itself
    would put a spike at the start of every clip, and a spike is exactly what
    the segmenter reads as the beginning of a rally.
    """
    height, width = frame_bgr.shape[:2]
    scale = MOTION_WORK_WIDTH / float(width)
    small = cv2.cvtColor(
        cv2.resize(frame_bgr, (MOTION_WORK_WIDTH, max(1, int(round(height * scale))))),
        cv2.COLOR_BGR2GRAY,
    )

    if previous_small is None or previous_small.shape != small.shape:
        return small, 0.0
    return small, float(np.mean(cv2.absdiff(small, previous_small)))


# --- timeline assembly ------------------------------------------------------


def build_rally_timeline(impact_times_s, motion_series, duration_s,
                         player_assignment):
    """The rally timeline a run reports, plus how it squares with the ball tier.

    `impact_times_s` is None when the clip's audio could not be read at all,
    and an empty list when it was read and was quiet. Those are different
    answers and are kept apart: collapsing them reports a clip with no audio
    track as one where nobody hit anything.

    `player_assignment` is the hit-derived rally structure, passed in as plain
    data. This function may *compare* against it -- that is reconciliation, and
    it happens here rather than in segment_rallies precisely so the segmenting
    itself stays unable to see a ball.
    """
    impacts = list(impact_times_s or [])
    rallies = segment_rallies(impacts, motion_series or [], duration_s)

    return {
        "rallies": rallies,
        "gap_s": infer_gap_seconds(impacts),
        "audio_available": impact_times_s is not None,
        "agrees_with_hits": _agreement_with_hit_rallies(rallies, player_assignment),
    }


def _agreement_with_hit_rallies(rallies, player_assignment):
    """Whether every hit-derived rally's midpoint falls inside a timeline span.

    None when there are no hit-derived rallies to compare against -- with the
    ball tier off there is nothing to agree with, and returning True would
    claim a corroboration that never happened.

    Midpoints rather than full containment: the two structures are built from
    different signals and their edges legitimately differ by a second or so.
    What matters is whether they found the same rallies, not the same borders.
    """
    hit_rallies = (player_assignment or {}).get("rallies") or []
    if not hit_rallies:
        return None

    for hit_rally in hit_rallies:
        start = float(hit_rally.get("start_time_seconds", 0.0))
        end = float(hit_rally.get("end_time_seconds", start))
        midpoint = (start + end) / 2.0
        if not any(r["start_s"] <= midpoint <= r["end_s"] for r in rallies):
            return False
    return True
