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

Everything here is a pure function of two time series, which is what makes the
thresholds testable without video.
"""

import statistics

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
    return spans


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
