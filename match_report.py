"""report-v1: everything a run found, and everything it could not.

This is where the analysis ladder becomes something a player can read. It
assembles from what a run left on disk rather than recomputing anything, so a
report is cheap and a finished run can be re-read forever.

The hard requirement here is not completeness, it is **tolerance**. Runs
recorded before any of these tiers existed are still on disk, and they have no
probe, no capabilities and no rally timeline. Every new key is read with a
default, and a run missing all of them is flagged `legacy` with a reason --
because "this tier found nothing" and "this run predates the tier" collapse to
the same empty card otherwise, and only one of them is worth acting on.

The same principle runs through the whole file: an absent value is None, never
a zero or an empty list. Zero is a measurement.
"""

import json
from pathlib import Path

REPORT_SCHEMA = "report-v1"

# Ladder order, so a report always lists tiers bottom-up regardless of dict
# ordering. Mirrors capabilities._TIERS.
TIER_ORDER = ("rally_structure", "player_movement", "ball_tracking", "line_calls")

LEGACY_REASON = (
    "Analyzed before capability gating existed, so this run has no record of "
    "what it could or could not measure."
)


def tiers_enabled(job):
    """Names of the tiers a run actually ran, in ladder order.

    Empty for a legacy run -- which is honest. It ran the ball pipeline, but
    nothing recorded on what grounds, so claiming a tier was "enabled" would
    be inventing a decision nobody made.
    """
    caps = job.get("capabilities") or {}
    return [name for name in TIER_ORDER if (caps.get(name) or {}).get("enabled")]


def _video_facts(job):
    """The clip's own numbers, for the report header.

    Prefers the probe, which measured the file, and falls back to the job's
    own fps and frame window for runs made before probing existed.
    """
    probe = job.get("probe") or {}
    fps = probe.get("fps") or job.get("fps")

    duration = probe.get("duration_s")
    if duration is None:
        start, end = job.get("start_frame"), job.get("end_frame")
        if fps and isinstance(start, int) and isinstance(end, int) and end >= start:
            duration = (end - start + 1) / float(fps)

    return {
        "fps": fps,
        "width": probe.get("width"),
        "height": probe.get("height"),
        "duration_s": duration,
    }


def _shots(run_dir, job, coach_builder):
    """Ball-tier output, or None when that tier never ran.

    None rather than an empty shot list: a rally in which nobody hit anything
    and a run that never looked for hits are different, and the second is much
    more common.
    """
    caps = job.get("capabilities")
    if caps is not None and not (caps.get("ball_tracking") or {}).get("enabled"):
        return None

    detected_path = Path(run_dir) / "detected_hits.json"
    if not detected_path.exists():
        return None
    try:
        detected = json.loads(detected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    shots = {
        "hit_count": len(detected.get("hits") or []),
        "target_zones": job.get("target_zones"),
        "target_zones_by_player": job.get("target_zones_by_player"),
        "floor_zones": job.get("floor_zones"),
        "rallies": (job.get("player_assignment") or {}).get("rallies"),
        "players_v1": job.get("players_v1"),
        "coach": None,
        "coach_error": None,
    }

    if coach_builder is not None:
        try:
            shots["coach"] = coach_builder(detected)
        except Exception as error:
            # Coaching is the most fragile link in the chain -- it can reach an
            # LLM over the network. A report that 500s because the narration
            # failed would take the rally and movement tiers down with it, and
            # those are computed and correct.
            shots["coach_error"] = str(error)

    return shots


def build_report(run_dir, coach_builder=None):
    """Assemble report-v1 for a finished run.

    Raises FileNotFoundError when the run has no job.json, so the endpoint can
    404 rather than serving a report about nothing.
    """
    run_dir = Path(run_dir)
    job_path = run_dir / "job.json"
    if not job_path.exists():
        raise FileNotFoundError(f"no job.json in {run_dir}")

    job = json.loads(job_path.read_text(encoding="utf-8"))

    # Capabilities are the marker: every run that has them was analyzed after
    # the tiers existed, and every run without them predates all of it.
    legacy = job.get("capabilities") is None

    return {
        "schema": REPORT_SCHEMA,
        "run_id": job.get("run_id") or run_dir.name,
        "created_ms": int(run_dir.name) if run_dir.name.isdigit() else None,
        "status": job.get("status"),
        "legacy": legacy,
        "legacy_reason": LEGACY_REASON if legacy else None,
        "video": _video_facts(job),
        "probe": job.get("probe"),
        "capabilities": job.get("capabilities"),
        "tiers_enabled": tiers_enabled(job),
        "detection_coverage": job.get("detection_coverage"),
        "ball_backend": job.get("ball_backend"),
        "rally_timeline": job.get("rally_timeline"),
        "players_v2": job.get("players_v2"),
        "shots": _shots(run_dir, job, coach_builder),
    }
