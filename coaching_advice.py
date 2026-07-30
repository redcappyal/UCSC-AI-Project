"""Drill progressions for the per-player coaching report.

Every weakness the analytics can see maps to a four-stage progression:
solo, then partner drills, then conditioned games, then matchplay. That order
is deliberate — a shot has to be groovable alone before it survives a drill,
and it has to survive a drill before it holds under match pressure.

THRESHOLDS ARE UNCALIBRATED ESTIMATES. Nobody has yet measured what "too
high" or "too fast" actually looks like across real club-level clips in this
app's own units, so the constants below are starting points chosen from the
court geometry (tin 1.6 ft, service line 5.8 ft, out line 15.0 ft) rather
than from data. Tune them against labeled footage before trusting the advice
they trigger.
"""

# --- tunable thresholds ---------------------------------------------------
# Front-wall contact height, feet. Between these two bounds is "varied".
HEIGHT_HIGH_FT = 9.5        # above: floating, predictable, easy to volley
HEIGHT_LOW_FT = 4.5         # below: hugging the tin, high error risk

# Share of shots into each band of the 3x3 front-wall map, percent.
SIDE_RATE_LOW_PCT = 50.0    # zones 1,2,3,7,8,9 — width
HIGH_RATE_LOW_PCT = 8.0     # zones 1,4,7 — lob band, above ~10.6 ft
MID_RATE_LOW_PCT = 40.0     # zones 2,5,8 — driving height, the backbone of a rally
LOW_RATE_LOW_PCT = 15.0     # zones 3,6,9 — attacking band near the tin

# Ball pace into the front wall, mph.
SPEED_FAST_MPH = 55.0       # above: one-paced hitting
SPEED_SLOW_MPH = 28.0       # below: not enough pace to pressure

# Share of a player's lost rallies that were their own errors, percent.
UNFORCED_ERROR_PCT_HIGH = 50.0

# Below this many analyzed shots the rates are noise, so advice is withheld.
MIN_HITS_FOR_ADVICE = 6
# Advice items returned at most, highest priority first.
MAX_ADVICE_ITEMS = 4

# 3x3 grid: columns left (1-3), center (4-6), right (7-9); rows lob, normal,
# low within each column.
ZONE_NAMES = {
    1: "lob height on the left",
    2: "driving height on the left",
    3: "low on the left",
    4: "lob height through the middle",
    5: "driving height through the middle",
    6: "low through the middle",
    7: "lob height on the right",
    8: "driving height on the right",
    9: "low on the right",
}
WIDE_ZONES = {1, 2, 3, 7, 8, 9}
LOB_ZONES = {1, 4, 7}
MID_ZONES = {2, 5, 8}
LOW_ZONES = {3, 6, 9}


def _item(metric, issue, solo, drills, conditioned, matchplay, priority):
    """One weakness plus the four-stage progression that addresses it."""
    return {
        "metric": metric,
        "issue": issue,
        "priority": priority,
        "progression": [
            {"stage": "Solo", "text": solo},
            {"stage": "Drills", "text": drills},
            {"stage": "Conditioned games", "text": conditioned},
            {"stage": "Matchplay", "text": matchplay},
        ],
    }


def _zone_list(zones):
    numbers = sorted(
        int(zone["zone"]) for zone in zones or [] if zone.get("zone") is not None
    )
    return numbers


def _describe_zones(numbers, limit=3):
    shown = numbers[:limit]
    names = [f"zone {n} ({ZONE_NAMES.get(n, 'unknown area')})" for n in shown]
    if not names:
        return ""
    if len(names) == 1:
        text = names[0]
    else:
        text = ", ".join(names[:-1]) + " and " + names[-1]
    if len(numbers) > limit:
        text += f", plus {len(numbers) - limit} more"
    return text


def _missing_zone_advice(missing_numbers):
    """Unused wall areas. What fixes them depends on where they sit."""
    if not missing_numbers:
        return None
    described = _describe_zones(missing_numbers)
    missing_set = set(missing_numbers)

    if missing_set & LOW_ZONES:
        drills = (
            "Boast and drive, or drop and drive with a partner, so the short "
            "shot is played off a moving ball rather than a static feed."
        )
    elif missing_set & LOB_ZONES:
        drills = (
            "Boast and lob with a partner — the lob forces the height these "
            "zones need, and the boast makes you play it under stretch."
        )
    elif missing_set & WIDE_ZONES:
        drills = (
            "Rotating drives with a partner, holding each shot straight and "
            "tight rather than letting it drift back to the middle."
        )
    else:
        drills = (
            "Feed-and-hit with a partner, calling the target area before each "
            "shot so the aim is deliberate."
        )

    return _item(
        metric="missing_target_zones",
        issue=f"No shots reached {described}.",
        solo=(
            "Solo to the unused areas: pick one zone, hit twenty balls into it, "
            "then switch. Accuracy first, pace later."
        ),
        drills=drills,
        conditioned=(
            "Play a game where a rally only counts if the ball reaches one of "
            "your unused zones, so you are forced to find them under pressure."
        ),
        matchplay=(
            "Commit to using one of these areas early in each game, before the "
            "score gets tight and you fall back on familiar shots."
        ),
        priority=40,
    )


def _height_advice(avg_height_ft, high_rate, mid_rate, low_rate):
    """Height sits on four signals: the average, and the three band rates."""
    items = []

    if avg_height_ft is not None and avg_height_ft > HEIGHT_HIGH_FT:
        items.append(_item(
            metric="average_wall_height_ft",
            issue=(
                f"Average front-wall height was {avg_height_ft:.1f} ft — high "
                "enough to be predictable and easy to volley."
            ),
            solo=(
                "Solo kills, drops and short shots so the low ball is available "
                "when you want it."
            ),
            drills=(
                "Drop and drive with a partner: alternate a low short shot with "
                "a full-height drive so the contrast becomes automatic."
            ),
            conditioned=(
                "Play short game, which removes the option of lifting everything "
                "and forces the lower shot."
            ),
            matchplay=(
                "Mix lower shots in deliberately. Variety, not height, is what "
                "stops the opponent reading you."
            ),
            priority=70,
        ))

    if avg_height_ft is not None and avg_height_ft < HEIGHT_LOW_FT:
        items.append(_item(
            metric="average_wall_height_ft",
            issue=(
                f"Average front-wall height was {avg_height_ft:.1f} ft — close "
                "to the tin, so the margin for error is thin."
            ),
            solo=(
                "Solo lifting: hit up the front wall and let the ball carry to "
                "the back, feeling how much height length actually needs."
            ),
            drills=(
                "Boast and lob with a partner, so lifting is practiced under "
                "stretch rather than from a comfortable position."
            ),
            conditioned=(
                "Play a game where every shot must strike the front wall above "
                "the service line, or a length game to the back quarters."
            ),
            matchplay=(
                "Use height to buy time when you are under pressure, and keep "
                "the ball tight to the side wall as it drops."
            ),
            priority=80,
        ))

    if mid_rate is not None and mid_rate < MID_RATE_LOW_PCT:
        items.append(_item(
            metric="mid_target_rate",
            issue=(
                f"Only {mid_rate:.0f}% of shots were struck at driving height, "
                "so the rally was rarely built on length."
            ),
            solo=(
                "Rotating drives, aiming between the service line and the "
                "lob band, letting the ball run to the back corners."
            ),
            drills=(
                "Rotating drives with a partner, both players holding length "
                "before anyone is allowed to go short."
            ),
            conditioned=(
                "Play deep game — everything must land behind the service box, "
                "which rewards a solid driving height."
            ),
            matchplay=(
                "Build the rally with length first and open the court before "
                "attacking short."
            ),
            priority=60,
        ))

    # A player who is generally too low already gets the lifting card above;
    # a second one about the lob would just repeat it.
    too_low = avg_height_ft is not None and avg_height_ft < HEIGHT_LOW_FT
    if high_rate is not None and high_rate < HIGH_RATE_LOW_PCT and not too_low:
        items.append(_item(
            metric="high_target_rate",
            issue=(
                f"Only {high_rate:.0f}% of shots reached lob height, so there "
                "was little lifting to escape pressure or move them back."
            ),
            solo=(
                "Solo lobs from the front corners, sending the ball high and "
                "soft so it drops into the back."
            ),
            drills=(
                "Boast and lob with a partner: the boast pulls you to the front "
                "and the lob is the shot that gets you out of it."
            ),
            conditioned=(
                "Play a game where any ball taken in front of the service box "
                "must be lifted, so the lob becomes the automatic rescue."
            ),
            matchplay=(
                "Lift when you are stretched or out of position, instead of "
                "forcing a low shot from a poor base."
            ),
            priority=55,
        ))

    if low_rate is not None and low_rate < LOW_RATE_LOW_PCT:
        items.append(_item(
            metric="low_target_rate",
            issue=(
                f"Only {low_rate:.0f}% of shots attacked the bottom of the front "
                "wall, so there was little short threat."
            ),
            solo=(
                "Solo short shots — drops and kills — from both sides, working "
                "on a low, tight contact."
            ),
            drills=(
                "Boast and drive, then drop and drive with a partner, so the "
                "short shot is played off a real ball."
            ),
            conditioned=(
                "Play short game to build the hold, the deception and the "
                "quality the short ball needs."
            ),
            matchplay=(
                "Hold the ball a beat longer before going short, so the "
                "opponent commits before you release it."
            ),
            priority=50,
        ))

    return items


def _width_advice(side_rate):
    if side_rate is None or side_rate >= SIDE_RATE_LOW_PCT:
        return None
    return _item(
        metric="side_target_rate",
        issue=(
            f"Only {side_rate:.0f}% of shots went wide, so most of the game came "
            "back through the middle where the opponent is already standing."
        ),
        solo=(
            "Rotating drives solo, varying pace and height, keeping the ball "
            "straight instead of drifting to the middle."
        ),
        drills=(
            "Rotating drives with a partner, focusing on straight shots rather "
            "than constantly switching sides."
        ),
        conditioned=(
            "Play a straight-only game — no crosscourts — so width has to come "
            "from accuracy rather than changing direction."
        ),
        matchplay=(
            "Favour the straight drive. Change sides because it is the right "
            "ball, not out of habit."
        ),
        priority=65,
    )


def _pace_advice(avg_speed_mph):
    if avg_speed_mph is None:
        return None
    if avg_speed_mph > SPEED_FAST_MPH:
        return _item(
            metric="average_incoming_speed_mph",
            issue=(
                f"Average pace into the front wall was {avg_speed_mph:.0f} mph — "
                "hard hitting with little change of speed."
            ),
            solo=(
                "Solo lobs and slower length, using the height of the front wall "
                "instead of pace to move the ball."
            ),
            drills=(
                "Drives with a partner at deliberately mixed paces, calling hard "
                "or soft before each shot."
            ),
            conditioned=(
                "Play deep game, where hitting flat and hard mostly returns the "
                "ball to the opponent."
            ),
            matchplay=(
                "Change pace to break rhythm rather than trying to win every "
                "rally with speed."
            ),
            priority=45,
        )
    if avg_speed_mph < SPEED_SLOW_MPH:
        return _item(
            metric="average_incoming_speed_mph",
            issue=(
                f"Average pace into the front wall was {avg_speed_mph:.0f} mph — "
                "not enough to hurry the opponent."
            ),
            solo=(
                "Solo kills and stuns, plus deliberate pace changes, so a faster "
                "ball is there when you need it."
            ),
            drills=(
                "Hard-hitting rotating drives with a partner to build racket "
                "speed under a repeating pattern."
            ),
            conditioned=(
                "Play a game where a hard straight drive past the service box "
                "scores double, to reward committing to pace."
            ),
            matchplay=(
                "Use a hard length early in the rally to disrupt the opponent's "
                "rhythm, then vary from there."
            ),
            priority=35,
        )
    return None


def _error_advice(unforced, unforced_pct, total_errors):
    items = []
    if unforced_pct is not None and unforced_pct >= UNFORCED_ERROR_PCT_HIGH and unforced:
        items.append(_item(
            metric="unforced_error_percentage",
            issue=(
                f"{unforced_pct:.0f}% of lost rallies were unforced errors "
                f"({unforced} of {total_errors}) — more points were given away "
                "than taken by the opponent."
            ),
            solo=(
                "Solo the shots that broke down — short shots, kills, and higher "
                "shots — chasing accuracy rather than pace."
            ),
            drills=(
                "Drill with a bigger margin: aim for shot quality above the tin "
                "rather than the tightest possible line."
            ),
            conditioned=(
                "Play a game where an error on the tin or out of court gives the "
                "opponent two points, which prices in the risk."
            ),
            matchplay=(
                "Take the safer, higher ball when the position is poor, and only "
                "attack from balance."
            ),
            priority=90,
        ))
    return items


def player_advice(player_analytics):
    """-> {'items': [...], 'note': str|None, 'low_sample_note': str|None} for one player's report."""
    analytics = player_analytics or {}
    total = int(analytics.get("total_wall_hits") or 0)

    if total < MIN_HITS_FOR_ADVICE:
        return {
            "items": [],
            "note": (
                f"Only {total} front-wall "
                f"{'shot was' if total == 1 else 'shots were'} analyzed — too "
                "few to draw a pattern from. Track a longer clip for coaching "
                "advice."
            ),
            "low_sample_note": None,
        }

    items = []
    items.extend(_error_advice(
        int(analytics.get("unforced_errors") or 0),
        analytics.get("unforced_error_percentage"),
        int(analytics.get("total_errors") or 0),
    ))
    items.extend(_height_advice(
        analytics.get("average_wall_height_ft"),
        analytics.get("high_target_rate"),
        analytics.get("mid_target_rate"),
        analytics.get("low_target_rate"),
    ))
    width = _width_advice(analytics.get("side_target_rate"))
    if width:
        items.append(width)
    missing = _missing_zone_advice(_zone_list(analytics.get("missing_target_zones")))
    if missing:
        items.append(missing)
    pace = _pace_advice(analytics.get("average_incoming_speed_mph"))
    if pace:
        items.append(pace)

    items.sort(key=lambda entry: -entry["priority"])
    items = items[:MAX_ADVICE_ITEMS]

    # `note` is a result the page renders as body copy; `low_sample_note` is a
    # hedge about a number that IS shown, so it rides behind the UI's
    # provenance disclosure (DESIGN.md 8.23) instead.
    note = None
    low_sample_note = None
    if not items:
        note = (
            "No clear weakness stood out in this clip — shot height, width and "
            "pace were all inside the expected range."
        )
    elif total < 20:
        low_sample_note = f"Based on {total} front-wall shots, so read it as a pointer."
    return {"items": items, "note": note, "low_sample_note": low_sample_note}
