"""Which analyses a clip can honestly support, and why not when it cannot.

Design-spec Principle 3: an analysis that cannot run on a given recording is
*disabled with a stated reason*, never silently degraded and never reported as
empty success. The failure this prevents is specific and was real -- the
pipeline used to finish a 30 fps clip with `hits=[]` and `status="complete"`,
which is indistinguishable from a rally where nothing happened.

The analysis tiers share inputs, but player movement and ball tracking are
sibling capabilities rather than a strict ladder:

    rally_structure   audio + frame motion        needs nothing
    player_movement   people through the floor    needs a solved court
    ball_tracking     per-frame ball detection    needs sharp, usable-size frames
    line_calls        judge ball impacts          needs ball_tracking

A floor homography is useful for player movement and court-position metrics,
but it is not required to locate a ball in image coordinates or judge it
against the calibrated front-wall lines. Keeping those requirements separate
prevents an optional floor solve from silently skipping the core tracker.
"""

CAPABILITIES_SCHEMA = "capabilities-v1"

# --- gates ------------------------------------------------------------------
# Every number here is a named constant because it will be retuned against real
# footage, and a threshold buried in an expression cannot be retuned honestly.

# The ball detector's operating range is 7-24 px (ios/MODEL.md). At 1080p the
# front-wall ball already sits at or below the bottom of that range, which is
# the standing eval headline (71/109 missed). 1600 px wide is the point below
# which the ball is smaller than the detector was ever shown.
BALL_MIN_WIDTH_PX = 1600

# Variance-of-Laplacian on a centre crop. Provisional: set from the sharp/soft
# separation the synthetic probe fixtures show, to be validated against the
# real 1080p60 / 1080p30 pair before anything is claimed about it.
BALL_MIN_SHARPNESS = 40.0

# Person tracking tolerates far more than the ball does -- a player crosses a
# fraction of the court between frames where the ball crosses the whole of it.
# 12 fps is roughly where a run between the T and a front corner stops being
# resolvable into a path at all.
MOVEMENT_MIN_FPS = 12.0

_NO_COURT = "no solved court"
_NO_PROBE = "clip was never probed"

_TIERS = ("rally_structure", "player_movement", "ball_tracking", "line_calls")


def _tier(reasons=()):
    """A tier verdict, enabled exactly when nothing disqualified it.

    Deriving `enabled` from the reasons rather than passing both means the two
    cannot drift apart -- an enabled tier carrying a stale reason is how a card
    ends up explaining why something is off while showing it on.
    """
    reasons = [reason for reason in reasons if reason]
    if not reasons:
        return {"enabled": True, "reason": None}
    return {"enabled": False, "reason": "; ".join(reasons)}


def _ball_reasons(probe):
    """Every failing ball gate, not just the first.

    A card naming only low resolution sends someone to fix one of two
    problems and be surprised when motion blur still keeps the tier off.
    """
    reasons = []

    if probe is None:
        reasons.append(_NO_PROBE)
        return reasons

    width = probe.get("width") or 0
    if width < BALL_MIN_WIDTH_PX:
        reasons.append(f"needs >={BALL_MIN_WIDTH_PX} px wide (got {width})")

    sharpness = probe.get("sharpness")
    if sharpness is None:
        reasons.append("could not measure sharpness")
    elif sharpness < BALL_MIN_SHARPNESS:
        reasons.append(
            f"too much motion blur (sharpness {sharpness:.0f} "
            f"< {BALL_MIN_SHARPNESS:.0f})"
        )

    return reasons


def _movement_reasons(probe, court_solved):
    reasons = []
    if not court_solved:
        reasons.append(_NO_COURT)

    if probe is None:
        # Frame rate is the only measured movement gate, and an unprobed clip
        # has none. The court gate above may already have disabled the tier;
        # this keeps the reason honest when it has not.
        reasons.append(_NO_PROBE)
        return reasons

    fps = probe.get("fps") or 0.0
    if fps < MOVEMENT_MIN_FPS:
        reasons.append(f"needs >={MOVEMENT_MIN_FPS:.0f} fps (got {fps:.0f})")

    return reasons


def compute_capabilities(probe, *, court_solved):
    """Turn a `media_probe.probe_video` result into per-tier verdicts.

    `probe` may be None -- legacy runs predate probing. That disables the
    measured tiers rather than assuming they would have passed, because
    assuming is how an unqualified clip produces a plausible wrong number.

    `court_solved` means a floor homography exists, from the tap wizard or
    from automatic detection. It gates player movement and floor-position
    metrics only; ball tracking operates in image coordinates.
    """
    ball = _tier(_ball_reasons(probe))

    return {
        # Unconditional by design. Audio absence is runtime information --
        # frame motion alone segments rallies -- and it surfaces on the
        # timeline itself, never as a gate here.
        "rally_structure": _tier(),
        "player_movement": _tier(_movement_reasons(probe, court_solved)),
        "ball_tracking": ball,
        # Judging consumes ball impacts, so it cannot outlive the tier that
        # produces them. Pointing at the ball tier rather than restating its
        # gates keeps one explanation of why, in one place. Built fresh rather
        # than aliased: sharing the dict would make a later edit to one tier
        # silently rewrite the other.
        "line_calls": _tier(
            [] if ball["enabled"] else [f"ball tracking off ({ball['reason']})"]
        ),
    }


def enabled_tiers(caps):
    """The names of the tiers that ran -- for logs and report assembly."""
    return [name for name in _TIERS if caps.get(name, {}).get("enabled")]
