import hashlib
import json
import math
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

import court_detect
import court_model
import match_report
from coaching_advice import player_advice
from media_probe import probe_video
from judge_call import (
    Point,
    clamp_wall_x_for_vertical_in,
    judge_ball,
    judge_margin_px,
    judge_serve_ball,
    load_ball_positions,
    load_calibration_lines,
    load_service_line,
    load_wall_corners,
    wall_diagram_coordinates,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT = Path(__file__).resolve().parent
APP_VERSION = "fps-wall-homography-2026-07-29-1"

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

# Ball detector: the default is ball_track_offline.BALL_DETECTOR_DEFAULT
# ("rfdetr", the hosted Roboflow RF-DETR), read through selected_detector() by
# every entry point. Do not re-pin it here with a setdefault -- that is what
# this line used to be, and it left two places owning one default. .env still
# overrides it; BALL_DETECTOR=local runs the committed WASB model instead.

# inference_engine sets the model-cache/metrics env defaults on import;
# import it (via job_runner) before anything touches the inference package.
from job_runner import (
    RUNS_DIR,
    UPLOADS_DIR,
    build_target_zone_summary,
    create_job,
    forget_job,
    get_job,
    is_serve_hit,
    request_cancel,
    start_tracking_job,
    update_job,
)
from inference_engine import DEFAULT_MODEL_ID, TRACKING_BACKEND

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

BY_HASH_DIR = UPLOADS_DIR / "by-hash"

BALL_POSITIONS_CACHE = {}
BALL_POSITIONS_LOCK = threading.Lock()
RUN_HITS_CACHE = {}
JUDGE_HIT_FRAME_TOLERANCE = 2
FRONT_WALL_OUT_HEIGHT_FT = 4.57 * 3.280839895
FRONT_WALL_TIN_HEIGHT_FT = 0.48 * 3.280839895
FRONT_WALL_SERVICE_HEIGHT_FT = 1.78 * 3.280839895
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_COACH_MODEL = "qwen3:8b"


def model_provenance():
    """What produced a run: model version, runtime, and app build.

    Recorded per-run because these are all environment-dependent — the same
    code on a teammate's laptop can pick a different device and a different
    ROBOFLOW_MODEL_ID, and a result that can't name its model can't be
    compared against another one.
    """
    return {
        "model_id": os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID),
        "tracking_backend": TRACKING_BACKEND,
        "device": os.environ.get("DEFAULT_DEVICE", "cpu"),
        "app_version": APP_VERSION,
    }


def error_response(message, status=400):
    return jsonify({"ok": False, "error": message}), status


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if fps else 0,
    }


def public_job(job):
    total_frames = max(1, int(job.get("total_frames", 1)))
    processed_frames = int(job.get("processed_frames", 0))
    progress = min(100.0, processed_frames / total_frames * 100)
    response = {
        "ok": True,
        "status": job.get("status", "queued"),
        "run_id": job.get("run_id"),
        "stage": job.get("stage"),
        "start_frame": job.get("start_frame"),
        "end_frame": job.get("end_frame"),
        "fps": job.get("fps"),
        "frame_stride": job.get("frame_stride", 1),
        "inference_width": job.get("inference_width", 0),
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "progress": progress,
        "message": job.get("message", ""),
    }

    for key in (
        "rows",
        "hits",
        "target_zones",
        "target_zones_by_player",
        "player_assignment",
        "players_v1",
        "user_player_number",
        "rallies",
        "floor_zones",
        "calibration_warning",
        "hits_error",
        "annotated_video_url",
        "csv_url",
        "error",
        # Analysis-tier state. The client renders capability cards from these,
        # so a key the worker writes but this whitelist omits is invisible to
        # every client. Listed ahead of the tasks that emit some of them --
        # only keys actually present are copied, so naming them early is inert.
        "probe",
        "capabilities",
        "detection_coverage",
        "rally_timeline",
        "players_v2",
    ):
        if key in job:
            response[key] = job[key]

    return response


def get_ball_positions(run_id, csv_path):
    mtime = csv_path.stat().st_mtime_ns
    with BALL_POSITIONS_LOCK:
        cached = BALL_POSITIONS_CACHE.get(run_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    positions = load_ball_positions(csv_path)
    with BALL_POSITIONS_LOCK:
        BALL_POSITIONS_CACHE[run_id] = (mtime, positions)
    return positions


def get_run_hits(run_id, run_dir):
    hits_path = run_dir / "detected_hits.json"
    if not hits_path.exists():
        return []

    mtime = hits_path.stat().st_mtime_ns
    with BALL_POSITIONS_LOCK:
        cached = RUN_HITS_CACHE.get(run_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    try:
        hits = json.loads(hits_path.read_text(encoding="utf-8")).get("hits", [])
    except (OSError, json.JSONDecodeError):
        hits = []

    with BALL_POSITIONS_LOCK:
        RUN_HITS_CACHE[run_id] = (mtime, hits)
    return hits


def find_run_hit_near_frame(run_id, run_dir, frame):
    best = None
    for hit in get_run_hits(run_id, run_dir):
        distance = abs(int(hit.get("frame", -10**9)) - frame)
        if distance <= JUDGE_HIT_FRAME_TOLERANCE and (best is None or distance < best[0]):
            best = (distance, hit)
    return best[1] if best else None


def find_hit_impact_near_frame(run_id, run_dir, frame):
    hit = find_run_hit_near_frame(run_id, run_dir, frame)
    return hit.get("impact") if hit else None


def front_wall_hits_from_payload(payload):
    hits = payload.get("hits", [])
    return [
        hit for hit in hits
        if hit.get("target_zone") is not None
        and hit.get("wall_diagram") is not None
        and hit.get("event_type") in (None, "wall", "unknown")
        and not is_serve_hit(hit)
    ]


def average(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def rounded(value, digits=1):
    return None if value is None else round(float(value), digits)


def wall_height_from_diagram_y(y):
    y = max(0.0, min(1.0, float(y)))
    return FRONT_WALL_OUT_HEIGHT_FT - y * (FRONT_WALL_OUT_HEIGHT_FT - FRONT_WALL_TIN_HEIGHT_FT)


def zone_lookup(zones):
    return {int(zone["zone"]): zone for zone in zones or [] if "zone" in zone}


def coaching_analytics_for_hits(hits, target_summary):
    target_zones = zone_lookup(target_summary.get("zones"))

    heights_ft = [
        wall_height_from_diagram_y(hit["wall_diagram"]["y"])
        for hit in hits
        if hit.get("wall_diagram") and hit["wall_diagram"].get("y") is not None
    ]
    speed_before_mph = [
        hit.get("velocity", {}).get("speed_before", {}).get("mph")
        for hit in hits
        if hit.get("velocity")
    ]
    speed_after_mph = [
        hit.get("velocity", {}).get("speed_after", {}).get("mph")
        for hit in hits
        if hit.get("velocity")
    ]
    speed_change_mph = [
        hit.get("velocity", {}).get("velocity_change", {}).get("mph")
        for hit in hits
        if hit.get("velocity")
    ]

    # 3x3 target grid: columns 1-3 left, 4-6 center, 7-9 right; within each
    # column the rows run lob -> normal (driving) -> low (attacking).
    def zone_total(*zones):
        return sum(int(target_zones.get(zone, {}).get("count", 0)) for zone in zones)

    total_wall_hits = int(target_summary.get("total_wall_hits") or len(hits))
    center_hits = zone_total(4, 5, 6)
    side_hits = zone_total(1, 2, 3, 7, 8, 9)
    high_hits = zone_total(1, 4, 7)     # lob band, above ~10.6 ft
    mid_hits = zone_total(2, 5, 8)      # driving height, ~10.6 ft to the service line
    low_hits = zone_total(3, 6, 9)      # service line down to the tin
    calls = [hit.get("call") for hit in hits]

    return {
        "total_wall_hits": total_wall_hits,
        "common_target_zones": target_summary.get("common_zones") or [],
        "missing_target_zones": target_summary.get("missing_zones") or [],
        "target_zone_percentages": [
            {
                "zone": int(zone.get("zone")),
                "count": int(zone.get("count", 0)),
                "percentage": rounded(zone.get("percentage", 0.0), 1),
            }
            for zone in (target_summary.get("zones") or [])
        ],
        "average_wall_height_ft": rounded(average(heights_ft), 1),
        "average_incoming_speed_mph": rounded(average(speed_before_mph), 1),
        "average_exit_speed_mph": rounded(average(speed_after_mph), 1),
        "average_velocity_change_mph": rounded(average(speed_change_mph), 1),
        "max_incoming_speed_mph": rounded(max(speed_before_mph), 1) if speed_before_mph else None,
        "center_target_rate": rounded(center_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "side_target_rate": rounded(side_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "low_target_rate": rounded(low_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "mid_target_rate": rounded(mid_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "high_target_rate": rounded(high_hits / total_wall_hits * 100, 1) if total_wall_hits else None,
        "in_count": sum(1 for call in calls if call == "IN"),
        "out_count": sum(1 for call in calls if call == "OUT"),
    }


def rally_duration_seconds(rally):
    """Return a saved or inferred rally duration, ignoring invalid timestamps."""
    saved_duration = rally.get("duration_seconds")
    if saved_duration is not None:
        try:
            duration = float(saved_duration)
        except (TypeError, ValueError):
            duration = None
        if duration is not None and math.isfinite(duration) and duration >= 0:
            return duration

    try:
        start_time = float(rally.get("start_time_seconds"))
        end_time = float(rally.get("end_time_seconds"))
    except (TypeError, ValueError):
        return None
    duration = end_time - start_time
    return duration if math.isfinite(duration) and duration >= 0 else None


def last_hit_reason_by_rally(front_wall_hits):
    """Map rally_number -> the judge reason of that rally's final hit."""
    last_hits = {}
    for hit in front_wall_hits or []:
        try:
            rally_number = int(hit.get("rally_number"))
        except (TypeError, ValueError):
            continue
        order = (
            float(hit.get("timestamp_seconds") or 0.0),
            int(hit.get("frame") or 0),
        )
        existing = last_hits.get(rally_number)
        if existing is None or order >= existing[0]:
            last_hits[rally_number] = (order, hit.get("reason"))
    return {number: reason for number, (_, reason) in last_hits.items()}


def player_error_metrics(rallies, player_number, front_wall_hits=None):
    """Summarize a player's lost-rally errors and won-rally durations.

    An OUT shot by the loser is an unforced error. When the winner made the
    final IN shot, the loser's failure to return it is a forced error. Rally
    duration is averaged over rallies won by this player so each player's
    value describes their own results.

    When hits are supplied, unforced errors split by the judge's per-hit
    reason: above the out line vs on/below the tin. Serve faults (below the
    service line) and rallies whose final hit is unmatched stay in the
    unforced total without a sub-bucket.
    """
    unforced_errors = 0
    unforced_out = 0
    unforced_tin = 0
    forced_errors = 0
    won_rally_durations = []
    last_reasons = last_hit_reason_by_rally(front_wall_hits)
    for rally in rallies or []:
        winner = int(rally.get("winner_player_number") or 0)
        if winner not in (1, 2):
            continue
        if winner == player_number:
            duration = rally_duration_seconds(rally)
            if duration is not None:
                won_rally_durations.append(duration)
            continue

        loser = 2 if winner == 1 else 1
        last_player = int(rally.get("last_player_number") or 0)
        if rally.get("last_call") == "OUT" and last_player == loser:
            unforced_errors += 1
            try:
                reason = last_reasons.get(int(rally.get("rally_number")))
            except (TypeError, ValueError):
                reason = None
            if reason == "above_or_on_top_line":
                unforced_out += 1
            elif reason == "below_or_on_bottom_line":
                unforced_tin += 1
        else:
            forced_errors += 1

    total_errors = unforced_errors + forced_errors
    return {
        "unforced_errors": unforced_errors,
        "unforced_errors_out": unforced_out,
        "unforced_errors_tin": unforced_tin,
        "forced_errors": forced_errors,
        "total_errors": total_errors,
        "average_rally_duration_seconds": rounded(average(won_rally_durations), 1),
        "unforced_error_percentage": (
            rounded(unforced_errors / total_errors * 100, 1)
            if total_errors
            else None
        ),
    }


def rally_number_for_hit(hit, numbered_rallies):
    """Match old hits without rally_number using saved rally bounds."""
    try:
        hit_rally_number = int(hit.get("rally_number"))
    except (TypeError, ValueError):
        hit_rally_number = None
    if hit_rally_number is not None:
        return hit_rally_number

    for value_key, start_key, end_key in (
        ("timestamp_seconds", "start_time_seconds", "end_time_seconds"),
        ("frame", "start_frame", "end_frame"),
    ):
        try:
            value = float(hit.get(value_key))
        except (TypeError, ValueError):
            continue
        for rally_number, rally in numbered_rallies:
            try:
                start = float(rally.get(start_key))
                end = float(rally.get(end_key))
            except (TypeError, ValueError):
                continue
            if start <= value <= end:
                return rally_number
    return None


def player_rally_outcome_analytics(front_wall_hits, rallies, player_number):
    """Split one player's shot metrics across rallies they won and lost."""
    numbered_rallies = []
    for index, rally in enumerate(rallies or [], start=1):
        try:
            rally_number = int(rally.get("rally_number") or index)
        except (TypeError, ValueError):
            rally_number = index
        numbered_rallies.append((rally_number, rally))

    player_hits_by_rally = {}
    for hit in front_wall_hits:
        if int(hit.get("player_number") or 0) != player_number:
            continue
        rally_number = rally_number_for_hit(hit, numbered_rallies)
        if rally_number is not None:
            player_hits_by_rally.setdefault(rally_number, []).append(hit)

    sections = {}
    for outcome, is_win in (("winning", True), ("losing", False)):
        section_rallies = []
        section_rally_numbers = set()
        for rally_number, rally in numbered_rallies:
            winner = int(rally.get("winner_player_number") or 0)
            if winner not in (1, 2):
                continue
            if (winner == player_number) == is_win:
                section_rallies.append(rally)
                section_rally_numbers.add(rally_number)

        section_hits = [
            hit
            for rally_number in section_rally_numbers
            for hit in player_hits_by_rally.get(rally_number, [])
        ]
        metrics = coaching_analytics_for_hits(
            section_hits,
            build_target_zone_summary(section_hits),
        )
        metrics.update({
            "outcome": outcome,
            "rally_count": len(section_rallies),
            "average_rally_duration_seconds": rounded(
                average(rally_duration_seconds(rally) for rally in section_rallies),
                1,
            ),
        })
        sections[outcome] = metrics
    return sections


def build_coaching_analytics(payload):
    floor_summary = payload.get("floor_zones") or {}
    rallies = (
        payload.get("rallies")
        or (payload.get("player_assignment") or {}).get("rallies")
        or []
    )
    front_wall_hits = front_wall_hits_from_payload(payload)

    target_summary = build_target_zone_summary(front_wall_hits)
    aggregate = coaching_analytics_for_hits(front_wall_hits, target_summary)
    players = []
    for player_number in (1, 2):
        player_hits = [
            hit
            for hit in front_wall_hits
            if int(hit.get("player_number") or 0) == player_number
        ]
        player_summary = build_target_zone_summary(player_hits)
        player_analytics = coaching_analytics_for_hits(player_hits, player_summary)
        player_analytics["player_number"] = player_number
        player_analytics["label"] = f"Player {player_number}"
        player_analytics.update(
            player_error_metrics(rallies, player_number, front_wall_hits)
        )
        player_analytics["rally_outcome_analytics"] = player_rally_outcome_analytics(
            front_wall_hits,
            rallies,
            player_number,
        )
        players.append(player_analytics)

    aggregate.update({
        "total_floor_bounces": int(floor_summary.get("total_floor_bounces") or 0),
        "common_floor_zones": floor_summary.get("common_zones") or [],
        "missing_floor_zones": floor_summary.get("missing_zones") or [],
        "average_wall_height_reference": {
            "tin_ft": round(FRONT_WALL_TIN_HEIGHT_FT, 1),
            "service_line_ft": round(FRONT_WALL_SERVICE_HEIGHT_FT, 1),
            "out_line_ft": round(FRONT_WALL_OUT_HEIGHT_FT, 1),
        },
        "players": players,
        "player_assignment": payload.get("player_assignment") or {
            "method": "legacy_global_alternation",
            "description": (
                "Odd-numbered front-wall contacts are Player 1; even-numbered "
                "front-wall contacts are Player 2."
            ),
        },
    })
    return aggregate


def local_coaching_feedback(analytics):
    if not analytics.get("total_wall_hits"):
        return (
            "No clear shot pattern was found yet. Review the labeled clips first and make sure "
            "the app is finding the main shots that hit the front wall."
        )

    notes = []
    players = [
        player for player in (analytics.get("players") or [])
        if player.get("total_wall_hits")
    ]
    for player in players[:2]:
        common = player.get("common_target_zones") or []
        center_rate = player.get("center_target_rate")
        side_rate = player.get("side_target_rate")
        avg_height = player.get("average_wall_height_ft")
        pieces = []
        if common:
            pieces.append(f"used zone {common[0]['zone']} most")
        if center_rate is not None:
            pieces.append(f"{center_rate:.0f}% middle")
        if side_rate is not None:
            pieces.append(f"{side_rate:.0f}% wide")
        if avg_height is not None:
            pieces.append(f"{avg_height:.1f} ft average height")
        if pieces:
            notes.append(f"{player['label']}: " + ", ".join(pieces) + ".")

    center_rate = analytics.get("center_target_rate")
    side_rate = analytics.get("side_target_rate")
    low_rate = analytics.get("low_target_rate")
    avg_height = analytics.get("average_wall_height_ft")
    avg_speed = analytics.get("average_incoming_speed_mph")
    common = analytics.get("common_target_zones") or []
    missing = analytics.get("missing_target_zones") or []

    if common:
        notes.append(
            f"Your most-used wall area was zone {common[0]['zone']} "
            f"({common[0]['percentage']:.0f}% of the shots analyzed)."
        )
    if center_rate is not None and center_rate >= 45:
        notes.append(
            f"{center_rate:.0f}% of your shots went through the middle of the wall. Mix in more "
            "width so the opponent has to move off the T."
        )
    elif side_rate is not None and side_rate >= 55:
        notes.append(
            f"{side_rate:.0f}% of your shots used the side areas. That is good for creating width "
            "and pulling the opponent away from the center."
        )
    if low_rate is not None and low_rate >= 45:
        notes.append(
            f"{low_rate:.0f}% of your shots hit lower on the front wall. That can be attacking, "
            "but it also raises error risk if you are not intentionally driving or dropping."
        )
    if avg_height is not None:
        if avg_height < FRONT_WALL_SERVICE_HEIGHT_FT:
            notes.append(
                f"Your typical front-wall contact height was {avg_height:.1f} ft, below the "
                "service line. That suggests a flatter, more attacking pattern."
            )
        else:
            notes.append(
                f"Your typical front-wall contact height was {avg_height:.1f} ft. Look for chances "
                "to vary height between safer length and lower attacking drives."
            )
    if avg_speed is not None:
        notes.append(f"Your average ball pace into the front wall was about {avg_speed:.1f} mph.")
    if missing:
        shown = ", ".join(str(zone["zone"]) for zone in missing[:3])
        notes.append(f"Across both players, zones {shown} were rarely used; those are good practice targets.")

    return " ".join(notes[:7])


def local_player_coaching_feedback(player_analytics):
    label = player_analytics.get("label") or f"Player {player_analytics.get('player_number', '')}".strip()
    total = int(player_analytics.get("total_wall_hits") or 0)
    if not total:
        return f"{label}: no reliable front-wall contacts were detected for this player."

    notes = []
    outcome_analytics = player_analytics.get("rally_outcome_analytics") or {}
    winning = outcome_analytics.get("winning") or {}
    losing = outcome_analytics.get("losing") or {}
    if winning.get("total_wall_hits") and losing.get("total_wall_hits"):
        winning_height = winning.get("average_wall_height_ft")
        losing_height = losing.get("average_wall_height_ft")
        if winning_height is not None and losing_height is not None:
            notes.append(
                f"{label}'s average shot height was {winning_height:.1f} ft in won rallies "
                f"versus {losing_height:.1f} ft in lost rallies."
            )

        usage_differences = []
        for metric, name in (
            ("center_target_rate", "middle usage"),
            ("side_target_rate", "wide usage"),
            ("low_target_rate", "low attacking rate"),
        ):
            winning_rate = winning.get(metric)
            losing_rate = losing.get(metric)
            if winning_rate is not None and losing_rate is not None:
                usage_differences.append((
                    abs(winning_rate - losing_rate),
                    name,
                    winning_rate,
                    losing_rate,
                ))
        if usage_differences:
            _, name, winning_rate, losing_rate = max(usage_differences)
            notes.append(
                f"{label}'s {name} was {winning_rate:.0f}% in won rallies versus "
                f"{losing_rate:.0f}% in lost rallies."
            )

    common = player_analytics.get("common_target_zones") or []
    center_rate = player_analytics.get("center_target_rate")
    side_rate = player_analytics.get("side_target_rate")
    low_rate = player_analytics.get("low_target_rate")
    avg_height = player_analytics.get("average_wall_height_ft")
    avg_speed = player_analytics.get("average_incoming_speed_mph")
    avg_rally_duration = player_analytics.get("average_rally_duration_seconds")
    missing = player_analytics.get("missing_target_zones") or []

    if common:
        top = common[0]
        notes.append(
            f"{label} used zone {top['zone']} most "
            f"({top.get('percentage', 0):.0f}% of their shots)."
        )
    if center_rate is not None and side_rate is not None:
        if center_rate >= 50:
            notes.append(
                f"{center_rate:.0f}% of {label}'s shots went through the middle. "
                "Mix in more width to move the opponent off the T."
            )
        else:
            notes.append(
                f"{side_rate:.0f}% of {label}'s shots went wide, which is good for "
                "using the side walls and making the opponent cover more court."
            )
    if low_rate is not None:
        notes.append(
            f"{label}'s low attacking rate was {low_rate:.0f}%. "
            "Low shots can pressure the opponent, but they raise tin risk if overused."
        )
    if avg_height is not None:
        notes.append(f"{label}'s typical front-wall contact height was {avg_height:.1f} ft.")
    if avg_speed is not None:
        notes.append(f"{label}'s average pace into the front wall was about {avg_speed:.1f} mph.")
    if avg_rally_duration is not None:
        notes.append(
            f"Rallies won by {label} averaged {avg_rally_duration:.1f} seconds."
        )
    if missing:
        shown = ", ".join(str(zone["zone"]) for zone in missing[:3])
        notes.append(f"{label} rarely used zones {shown}; those are useful practice targets.")

    return " ".join(notes[:5])


def extract_openai_text(data):
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()
    parts = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def openai_coach_response_format():
    player_schema = {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {"type": "string", "maxLength": 220},
                "minItems": 2,
                "maxItems": 2,
                "description": "Two concise, evidence-based observations for this player.",
            },
            "drill_name": {"type": "string", "maxLength": 80},
            "drill_instructions": {
                "type": "string",
                "maxLength": 300,
                "description": "A specific drill setup with repetitions or duration.",
            },
            "drill_goal": {
                "type": "string",
                "maxLength": 180,
                "description": "A measurable target for completing the drill successfully.",
            },
        },
        "required": [
            "observations",
            "drill_name",
            "drill_instructions",
            "drill_goal",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": "squash_coaching_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "players": {
                    "type": "object",
                    "properties": {
                        "1": player_schema,
                        "2": player_schema,
                    },
                    "required": ["1", "2"],
                    "additionalProperties": False,
                },
                "summary": {
                    "type": "string",
                    "maxLength": 360,
                    "description": (
                        "One concise sentence comparing the players; no repeated metrics."
                    ),
                },
            },
            # Put the bounded player sections first. Local models tend to
            # follow schema property order; this prevents an overlong summary
            # from consuming the entire generation before the useful report.
            "required": ["players", "summary"],
            "additionalProperties": False,
        },
    }


def parse_llm_coaching_report(text):
    if not text:
        return None
    try:
        report = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict) or not isinstance(report.get("summary"), str):
        return None

    players = report.get("players")
    if not isinstance(players, dict):
        return None
    for player_number in ("1", "2"):
        player = players.get(player_number)
        if not isinstance(player, dict):
            return None
        observations = player.get("observations")
        if (
            not isinstance(observations, list)
            or not observations
            or any(not isinstance(item, str) or not item.strip() for item in observations)
        ):
            return None
        for key in ("drill_name", "drill_instructions", "drill_goal"):
            if not isinstance(player.get(key), str) or not player[key].strip():
                return None
    return report


def format_llm_player_feedback(player_report):
    observations = "\n".join(
        f"• {observation.strip()}"
        for observation in player_report.get("observations", [])
        if observation.strip()
    )
    return (
        f"Feedback\n{observations}\n\n"
        f"Drill: {player_report['drill_name'].strip()}\n"
        f"{player_report['drill_instructions'].strip()}\n"
        f"Goal: {player_report['drill_goal'].strip()}"
    )


COACHING_LLM_PLAYER_METRICS = (
    "player_number",
    "total_wall_hits",
    "unforced_errors",
    "forced_errors",
    "total_errors",
    "unforced_error_percentage",
    "average_rally_duration_seconds",
    "average_wall_height_ft",
    "average_incoming_speed_mph",
    "average_exit_speed_mph",
    "average_velocity_change_mph",
    "max_incoming_speed_mph",
    "center_target_rate",
    "side_target_rate",
    "low_target_rate",
    "mid_target_rate",
    "high_target_rate",
    "in_count",
    "out_count",
)

COACHING_LLM_OUTCOME_METRICS = (
    "rally_count",
    "total_wall_hits",
    "average_rally_duration_seconds",
    "average_wall_height_ft",
    "average_incoming_speed_mph",
    "center_target_rate",
    "side_target_rate",
    "low_target_rate",
    "mid_target_rate",
    "high_target_rate",
)


def compact_coaching_analytics(analytics):
    """Keep only measurements that can support the requested coaching.

    The UI analytics object contains repeated nine-zone tables at the match,
    player, and won/lost-rally levels. Sending all of them encouraged qwen3 to
    restate the same percentages until it hit Ollama's output limit. This
    compact form retains the evidence while removing those repetitions.
    """

    compact_players = []
    for player in analytics.get("players", []) or []:
        if not isinstance(player, dict):
            continue
        compact = {
            key: player.get(key)
            for key in COACHING_LLM_PLAYER_METRICS
            if key in player
        }
        compact["target_zones_used"] = {
            str(zone.get("zone")): zone.get("percentage")
            for zone in player.get("target_zone_percentages", []) or []
            if isinstance(zone, dict) and (zone.get("count") or 0) > 0
        }
        compact["unused_target_zones"] = [
            zone.get("zone")
            for zone in player.get("missing_target_zones", []) or []
            if isinstance(zone, dict) and zone.get("zone") is not None
        ]
        outcomes = {}
        for outcome_name, outcome in (
            player.get("rally_outcome_analytics", {}) or {}
        ).items():
            if not isinstance(outcome, dict):
                continue
            if not (outcome.get("rally_count") or outcome.get("total_wall_hits")):
                continue
            outcomes[outcome_name] = {
                key: outcome.get(key)
                for key in COACHING_LLM_OUTCOME_METRICS
                if key in outcome
            }
        compact["rally_outcomes"] = outcomes
        compact_players.append(compact)
    return {
        "players": compact_players,
        "notes": {
            "total_wall_hits_means": "shots analyzed",
            "wall_height_means": "front-wall impact height",
            "speed_is_estimated": True,
        },
    }


def coaching_messages(analytics):
    coaching_analytics = compact_coaching_analytics(analytics)
    return [
        {
            "role": "system",
            "content": (
                "You are a practical squash coach. Analyze only the supplied automated "
                "match analytics and never invent shot types, technique, handedness, or "
                "court movement that the data does not measure. Give each player two "
                "concise observations, followed by one personalized drill with a clear "
                "setup, repetitions or duration, and a measurable goal. Cite numeric "
                "metrics when available. If a player's sample is small, say so plainly. "
                "Compare winning-rally and losing-rally analytics when both samples exist, "
                "and highlight useful differences without claiming they caused the result. "
                "Keep each player's observations about that player's own measurements; "
                "reserve cross-player comparisons for the one-sentence summary, and check "
                "the numeric direction of every comparison. "
                "Use 'shots analyzed' instead of 'front-wall hits'. Wall height means "
                "where the ball struck the front wall; pace is estimated ball speed "
                "around front-wall contact."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create the overall summary and separate Player 1 and Player 2 coaching "
                "reports from these analytics. The summary must be one sentence under "
                "45 words. Return exactly two observations per player and do not repeat "
                "the same metric:\n"
                + json.dumps(coaching_analytics, separators=(",", ":"))
            ),
        },
    ]


def configured_coach_provider():
    configured = os.getenv("COACH_LLM_PROVIDER", "").strip().lower()
    aliases = {
        "ollama": "ollama",
        "openai": "openai",
        "local": "local",
        "none": "local",
        "off": "local",
    }
    if configured:
        return aliases.get(configured, "invalid")
    return "openai" if os.getenv("OPENAI_API_KEY", "").strip() else "local"


def openai_coaching_feedback(analytics):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "missing_api_key"

    model = os.getenv("OPENAI_COACH_MODEL", "gpt-5-mini")
    body = json.dumps(
        {
            "model": model,
            "input": coaching_messages(analytics),
            "text": {
                "verbosity": "low",
                "format": openai_coach_response_format(),
            },
            "max_output_tokens": 850,
        }
    ).encode("utf-8")
    request_obj = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return None, "request_failed"

    text = extract_openai_text(data)
    if not text:
        return None, "empty_response"
    report = parse_llm_coaching_report(text)
    return (report, "ok") if report else (None, "invalid_response")


def ollama_coaching_feedback(analytics):
    model = os.getenv("OLLAMA_COACH_MODEL", DEFAULT_OLLAMA_COACH_MODEL).strip()
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).strip().rstrip("/")
    try:
        timeout = max(1.0, float(os.getenv("OLLAMA_COACH_TIMEOUT_SECONDS", "120")))
    except ValueError:
        timeout = 120.0
    try:
        max_tokens = max(
            400, int(os.getenv("OLLAMA_COACH_MAX_TOKENS", "1200"))
        )
    except ValueError:
        max_tokens = 1200

    body = json.dumps(
        {
            "model": model,
            "messages": coaching_messages(analytics),
            "stream": False,
            "think": False,
            "format": openai_coach_response_format()["schema"],
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
                "num_ctx": 8192,
            },
        }
    ).encode("utf-8")
    request_obj = urllib.request.Request(
        f"{base_url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return None, "ollama_model_missing" if error.code == 404 else "ollama_request_failed"
    except json.JSONDecodeError:
        return None, "invalid_response"
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, "ollama_unavailable"

    if not isinstance(data, dict):
        return None, "invalid_response"
    if data.get("done_reason") == "length":
        return None, "truncated_response"
    text = ((data.get("message") or {}).get("content") or "").strip()
    if not text:
        return None, "empty_response"
    report = parse_llm_coaching_report(text)
    return (report, "ok") if report else (None, "invalid_response")


def llm_coaching_feedback(analytics):
    provider = configured_coach_provider()
    if provider == "ollama":
        return ollama_coaching_feedback(analytics)
    if provider == "openai":
        return openai_coaching_feedback(analytics)
    if provider == "local":
        return None, "local_only"
    return None, "invalid_provider"


@app.get("/")
def index():
    response = send_file(ROOT / "index.html", max_age=0)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "version": APP_VERSION,
            "root": str(ROOT),
            "tracking_backend": TRACKING_BACKEND,
            "default_device": os.environ.get("DEFAULT_DEVICE"),
            "onnx_providers": os.environ.get("ONNXRUNTIME_EXECUTION_PROVIDERS"),
        }
    )


def video_path_for_id(video_id):
    video_id = secure_filename(str(video_id))
    if not video_id:
        return None

    matches = sorted(BY_HASH_DIR.glob(f"{video_id}.*"))
    return matches[0] if matches else None


@app.post("/api/upload")
def upload_video():
    video_file = request.files.get("video_file")
    if video_file is None or not video_file.filename:
        return error_response("No video file provided.")

    suffix = Path(secure_filename(video_file.filename)).suffix or ".mp4"
    BY_HASH_DIR.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    tmp_path = BY_HASH_DIR / f"upload-{int(time.time() * 1000)}-{threading.get_ident()}.tmp"
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = video_file.stream.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)

        video_id = hasher.hexdigest()
        final_path = BY_HASH_DIR / f"{video_id}{suffix}"
        if final_path.exists():
            tmp_path.unlink()
        else:
            os.replace(tmp_path, final_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    payload = {"ok": True, "video_id": video_id}
    try:
        info = video_info(final_path)
        payload.update(
            fps=info["fps"],
            frame_count=info["frame_count"],
            duration=info["duration"],
        )
    except ValueError:
        pass

    return jsonify(payload)


# --- chunked upload ---------------------------------------------------------
# /api/upload is a whole-file multipart POST capped at 2 GB -- roughly five
# minutes of our own 4K60 capture. A forty-minute match cannot be ingested at
# all, which makes "record a session and analyze it" false for real sessions,
# and nothing downstream can fix that.

# Largest single chunk accepted. Comfortably under MAX_CONTENT_LENGTH and small
# enough that a dropped connection costs one retry rather than the upload.
UPLOAD_CHUNK_MAX_BYTES = 32 * 1024 * 1024

UPLOAD_ID_BYTES = 16


def partial_dir():
    """Where in-progress uploads accumulate.

    Derived from BY_HASH_DIR rather than being a module-level constant so the
    `runs_dir` test fixture sandboxes it -- otherwise a test run writes real
    files into the developer's upload store, which is the exact shared state
    that fixture exists to remove.
    """
    return BY_HASH_DIR.parent / "partial"


def partial_path(upload_id):
    return partial_dir() / f"{upload_id}.part"


def partial_meta_path(upload_id):
    return partial_dir() / f"{upload_id}.json"


@app.post("/api/upload/init")
def upload_init():
    """Open a chunked upload. Returns the id every later call carries."""
    body = request.get_json(silent=True) or {}
    filename = secure_filename(str(body.get("filename") or "")) or "upload.mp4"
    # The suffix is captured now and carried to the assembled file because
    # video_path_for_id resolves ids by globbing `<id>.*`. An extensionless
    # assembly uploads and hashes correctly and is then unresolvable -- a
    # failure that only surfaces one screen later, at TRACK.
    suffix = Path(filename).suffix or ".mp4"

    upload_id = os.urandom(UPLOAD_ID_BYTES).hex()
    partial_dir().mkdir(parents=True, exist_ok=True)
    partial_path(upload_id).write_bytes(b"")
    partial_meta_path(upload_id).write_text(
        json.dumps({"suffix": suffix, "next_index": 0,
                    "declared_size": int(body.get("size") or 0)}),
        encoding="utf-8",
    )
    return jsonify({"ok": True, "upload_id": upload_id,
                    "chunk_max_bytes": UPLOAD_CHUNK_MAX_BYTES})


def _load_partial_meta(upload_id):
    meta_path = partial_meta_path(upload_id)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.post("/api/upload/chunk/<upload_id>")
def upload_chunk(upload_id):
    """Append one strictly-sequential chunk.

    Strict sequencing is the integrity guarantee. Accepting a gap would
    assemble a file with a hole in it, and that file still decodes, still
    analyses, and produces statistics about corrupted frames -- nothing
    downstream could tell. A repeat is refused for the same reason: a client
    retrying after a timeout must not double its bytes.
    """
    upload_id = secure_filename(upload_id)
    meta = _load_partial_meta(upload_id)
    if meta is None:
        return error_response("Upload was not found.", status=404)

    try:
        index = int(request.args.get("index", ""))
    except ValueError:
        return error_response("Chunk index must be a number.")

    chunk = request.get_data(cache=False)
    if len(chunk) > UPLOAD_CHUNK_MAX_BYTES:
        return error_response(
            f"Chunk exceeds {UPLOAD_CHUNK_MAX_BYTES} bytes.", status=413
        )

    if index != meta["next_index"]:
        return error_response(
            f"Expected chunk {meta['next_index']}, got {index}.", status=409
        )

    with partial_path(upload_id).open("ab") as out:
        out.write(chunk)

    meta["next_index"] = index + 1
    partial_meta_path(upload_id).write_text(json.dumps(meta), encoding="utf-8")
    return jsonify({"ok": True, "received_index": index,
                    "bytes": partial_path(upload_id).stat().st_size})


@app.post("/api/upload/complete/<upload_id>")
def upload_complete(upload_id):
    """Hash the assembly, move it into the by-hash store, answer like /api/upload.

    Same content-addressed store, so re-uploading a file already present costs
    a hash and nothing else.
    """
    upload_id = secure_filename(upload_id)
    meta = _load_partial_meta(upload_id)
    part_path = partial_path(upload_id)
    if meta is None or not part_path.exists():
        return error_response("Upload was not found.", status=404)

    hasher = hashlib.sha256()
    with part_path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)

    video_id = hasher.hexdigest()
    BY_HASH_DIR.mkdir(parents=True, exist_ok=True)
    final_path = BY_HASH_DIR / f"{video_id}{meta['suffix']}"
    try:
        if final_path.exists():
            part_path.unlink()
        else:
            os.replace(part_path, final_path)
    finally:
        # Abandoned partials of a forty-minute match are not small.
        part_path.unlink(missing_ok=True)
        partial_meta_path(upload_id).unlink(missing_ok=True)

    payload = {"ok": True, "video_id": video_id}
    try:
        info = video_info(final_path)
        payload.update(fps=info["fps"], frame_count=info["frame_count"],
                       duration=info["duration"])
    except ValueError:
        # Not decodable as video. The bytes are stored and addressable; the
        # caller finds out at track time, exactly as /api/upload behaves.
        pass

    return jsonify(payload)


@app.get("/api/court-model")
def get_court_model():
    """Court dimensions, landmarks, and wireframe for the calibration wizard."""
    return jsonify({"ok": True, **court_model.court_model_public()})


@app.get("/api/calibration/latest")
def latest_calibration():
    """Most recent run's calibration so a native client can reuse the court
    setup without redoing the wizard. Recency = calibration.json mtime, which
    tracks the last time /api/track accepted that calibration.

    An optional ?camera_id=<id> query param restricts eligibility to
    calibrations whose stored JSON has a matching top-level "camera_id" key;
    without it (including an empty ?camera_id=), every calibration is
    eligible (today's behavior)."""
    camera_id = request.args.get("camera_id") or None
    candidates = []
    if RUNS_DIR.exists():
        for path in RUNS_DIR.glob("*/calibration.json"):
            try:
                candidates.append(((path.stat().st_mtime_ns, path.parent.name), path))
            except OSError:
                continue
    candidates.sort(key=lambda entry: entry[0], reverse=True)

    for _key, path in candidates:
        try:
            calibration = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if camera_id is None:
                return error_response(
                    "Latest calibration could not be read.", status=500)
            continue
        if camera_id is not None and calibration.get("camera_id") != camera_id:
            continue
        saved_at = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
        return jsonify({
            "ok": True,
            "run_id": path.parent.name,
            "saved_at": saved_at,
            "calibration": calibration,
        })

    return error_response(
        "No saved calibration found. Run a calibrated analysis first.", status=404)


@app.post("/api/camera-check")
def camera_check():
    """Run the full camera solve on a candidate calibration and report health.

    Read-only wizard feedback: nothing is stored. Always 200 with a status
    field, mirroring solve_camera_model's never-raise contract.
    """
    payload = request.get_json(silent=True) or {}
    calibration = payload.get("calibration")
    if calibration is None:
        try:
            calibration = json.loads(payload.get("calibration_json") or "")
        except (json.JSONDecodeError, TypeError):
            return jsonify({"ok": True, "status": "invalid_json"})
    _, info = court_model.solve_camera_model(calibration)
    return jsonify({"ok": True, **info})


@app.post("/api/camera-model")
def camera_model():
    """Solve the full camera model for a calibration and return it as JSON.

    Same input contract and always-200 convention as /api/camera-check; adds
    the solved model (court_model.CameraModel.to_dict) under "camera_model"
    when the solve succeeds. Phones exchange these solved models at pairing.

    The solved frame size is echoed at the top level as well as inside the
    model: every px-valued field is only meaningful in that pixel space, and
    a client running a different capture resolution has to scale the model
    (court_model.scale_camera_model) before casting rays through it.
    """
    payload = request.get_json(silent=True) or {}
    calibration = payload.get("calibration")
    if calibration is None:
        try:
            calibration = json.loads(payload.get("calibration_json") or "")
        except (json.JSONDecodeError, TypeError):
            return jsonify({"ok": True, "status": "invalid_json"})
    model, info = court_model.solve_camera_model(calibration)
    response = {"ok": True, **info}
    if model is not None and info.get("status") == "ok":
        response["camera_model"] = model.to_dict()
        response["frame_width"] = model.frame_width
        response["frame_height"] = model.frame_height
    return jsonify(response)


MAX_DETECT_FRAMES = 8


@app.post("/api/detect-court")
def detect_court_endpoint():
    """Detect the court from a few frames of one fixed viewpoint.

    Read-only wizard feedback like /api/camera-check: nothing is stored, and
    the reply is always 200 with a status field so the client can fall back to
    the manual tap wizard on any failure.

    Frames arrive as JPEG bytes and are decoded with cv2.imdecode, which reads
    from memory -- so the CLAUDE.md warning about cv2.imread and non-ASCII
    paths does not apply here.
    """
    uploads = request.files.getlist("frames")
    if not uploads:
        return jsonify({"ok": True, "status": "no_frames",
                        "warnings": ["No frames were supplied."]})

    frames = []
    for upload in uploads[:MAX_DETECT_FRAMES]:
        buffer = np.frombuffer(upload.read(), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is not None and (not frames or image.shape == frames[0].shape):
            frames.append(image)

    if not frames:
        return jsonify({"ok": True, "status": "no_frames",
                        "warnings": ["No frame could be decoded."]})

    return jsonify(court_detect.detect_court(frames))


def validate_floor_calibration(calibration):
    """Return a warning string (and strip the floor plane) if it cannot be used.

    Floor mapping is additive: a bad floor plane must never fail the run or
    regress front-wall judging, so we drop it and surface a warning instead.
    """
    planes = calibration.get("planes")
    if not isinstance(planes, dict) or "floor" not in planes:
        return None

    try:
        floor_map = court_model.load_floor_calibration(calibration)
    except Exception:
        floor_map = None
    if floor_map is not None:
        return None

    planes.pop("floor", None)
    return "Floor calibration was invalid and has been ignored for this run."


@app.post("/api/track")
def track_clip():
    video_file = request.files.get("video_file")
    video_id = request.form.get("video_id", "").strip()
    calibration_text = request.form.get("calibration_json", "")

    if not video_id and (video_file is None or not video_file.filename):
        return error_response("Upload the source video before tracking.")

    try:
        calibration = json.loads(calibration_text)
    except json.JSONDecodeError:
        return error_response("Calibration JSON was invalid.")

    try:
        start_time = float(request.form.get("start_time", "0"))
        end_time = float(request.form.get("end_time", "0"))
        frame_stride = int(request.form.get("frame_stride", "1"))
        inference_width = int(request.form.get("inference_width", "960"))
    except ValueError:
        return error_response(
            "Clip start/end times, frame stride, and inference width must be numbers."
        )

    if end_time <= start_time:
        return error_response("Clip end must be after clip start.")

    if frame_stride < 1 or frame_stride > 10:
        return error_response("Frame stride must be between 1 and 10.")

    if inference_width not in {0, 640, 960, 1280}:
        return error_response("Inference width must be 0, 640, 960, or 1280.")

    run_id = str(int(time.time() * 1000))
    run_dir = RUNS_DIR / run_id

    if video_id:
        video_path = video_path_for_id(video_id)
        if video_path is None:
            return error_response("Uploaded video was not found. Upload it again.", status=404)
    else:
        upload_dir = UPLOADS_DIR / run_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_name = secure_filename(video_file.filename) or "source_video.mp4"
        video_path = upload_dir / video_name
        video_file.save(video_path)

    try:
        info = video_info(video_path)
    except ValueError as error:
        return error_response(str(error))

    start_frame = max(0, int(round(start_time * info["fps"])))
    end_frame = min(info["frame_count"] - 1, int(round(end_time * info["fps"])))
    if end_frame < start_frame:
        return error_response("Selected clip is outside the video duration.")

    calibration_warning = validate_floor_calibration(calibration)
    if calibration_warning:
        app.logger.warning("run %s: %s", run_id, calibration_warning)

    # Created only now, with every rejection path behind us. Made any earlier it
    # outlives the error that refused it — the 404 above returns without
    # cleaning up — and those empty runs then accumulate in the runs list.
    run_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = run_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    selected_frames = end_frame - start_frame + 1
    total_frames = (selected_frames + frame_stride - 1) // frame_stride
    extra_job_fields = {}
    if calibration_warning:
        extra_job_fields["calibration_warning"] = calibration_warning

    # This is the only place holding the file, so it is the only place that can
    # measure what the clip can support. The worker gates its tiers on the
    # result. A probe failure costs the capability *detail*, never the run:
    # a clip OpenCV cannot measure may still be one a player wants analysed,
    # and compute_capabilities already treats a missing probe conservatively.
    try:
        extra_job_fields["probe"] = probe_video(video_path)
    except Exception as error:
        app.logger.warning("run %s: could not probe video: %s", run_id, error)
    create_job(
        run_id,
        run_dir,
        status="queued",
        message="Queued tracking job.",
        video_path=str(video_path),
        start_frame=start_frame,
        end_frame=end_frame,
        fps=info["fps"],
        frame_stride=frame_stride,
        inference_width=inference_width,
        processed_frames=0,
        total_frames=total_frames,
        csv_url=f"/api/runs/{run_id}/ball_coordinates.csv",
        # Provenance: which model and runtime produced this run. Without it a
        # result can't be attributed to a model version, so an eval regression
        # can't be traced to the change that caused it.
        **model_provenance(),
        **extra_job_fields,
    )
    start_tracking_job(run_id)

    return jsonify(public_job(get_job(run_id)))


@app.get("/api/track/status/<run_id>")
def track_status(run_id):
    job = get_job(secure_filename(run_id))
    if job is None:
        return error_response("Tracking job was not found.", status=404)

    return jsonify(public_job(job))


@app.post("/api/track/cancel/<run_id>")
def track_cancel(run_id):
    """Abandon a run. Only one job holds the tracking semaphore, so a run the
    user backed out of has to be stopped for real or it blocks the next one."""
    safe_id = secure_filename(run_id)
    job = get_job(safe_id)
    if job is None:
        return error_response("Tracking job was not found.", status=404)

    if job.get("status") in {"queued", "running"}:
        request_cancel(safe_id)
    return jsonify({"ok": True, "run_id": safe_id, "status": job.get("status")})


@app.post("/api/judge")
def judge_frame():
    data = request.get_json(silent=True) or {}
    run_id = str(data.get("run_id", "")).strip()
    if not run_id:
        return error_response("Missing run_id.")

    try:
        frame = int(data.get("frame"))
    except (TypeError, ValueError):
        return error_response("Frame must be an integer.")

    run_dir = RUNS_DIR / secure_filename(run_id)
    csv_path = run_dir / "ball_coordinates.csv"
    calibration_path = run_dir / "calibration.json"

    if not csv_path.exists() or not calibration_path.exists():
        return error_response("Tracking result was not found.", status=404)

    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        top_line, bottom_line = load_calibration_lines(calibration)
        wall_corners = load_wall_corners(calibration)
        selected_hit = find_run_hit_near_frame(run_id, run_dir, frame)

        impact = selected_hit.get("impact") if selected_hit else None
        if impact is not None:
            ball = Point(float(impact["x"]), float(impact["y"]))
            source = "impact_estimate"
        else:
            ball = get_ball_positions(run_id, csv_path).get(frame)
            if ball is None:
                raise ValueError(f"No ball detection recorded for frame {frame}.")
            source = "detected_center"

        original_ball = ball
        ball, ball_x_clamped = clamp_wall_x_for_vertical_in(
            ball, top_line, bottom_line, wall_corners
        )
        call, reason, top_y, bottom_y = judge_ball(
            original_ball, top_line, bottom_line, wall_corners
        )
        standard_call = call
        margin_px = (
            judge_margin_px(ball, top_line, bottom_line, wall_corners)
            if call in ("IN", "OUT")
            else None
        )
        is_serve = bool(selected_hit and selected_hit.get("is_serve"))
        service_y = None
        if is_serve:
            service_line = load_service_line(calibration)
            call, reason, _, service_y = judge_serve_ball(
                ball, top_line, service_line, wall_corners
            )
            margin_px = (
                judge_margin_px(ball, top_line, service_line, wall_corners)
                if call in ("IN", "OUT")
                else None
            )
        diagram = wall_diagram_coordinates(
            ball,
            top_line,
            bottom_line,
            calibration.get("frame_width", 1),
            wall_corners,
        )
    except Exception as error:
        return error_response(str(error))

    return jsonify(
        {
            "ok": True,
            "frame": frame,
            "call": call,
            "standard_call": standard_call,
            "is_serve": is_serve,
            "reason": reason,
            "source": source,
            "ball_x_clamped": ball_x_clamped,
            "margin_px": margin_px,
            "ball": {"x": ball.x, "y": ball.y},
            "original_ball": (
                {"x": original_ball.x, "y": original_ball.y}
                if ball_x_clamped
                else None
            ),
            "top_y": top_y,
            "bottom_y": bottom_y,
            "service_y": service_y,
            "wall_diagram": {
                "x": diagram["x"],
                "y": diagram["y"],
                "x_span": diagram["x_span"],
                "x_reference": diagram.get("x_reference"),
                "y_reference": "0 is the out-line lower edge; 1 is the tin top edge",
            },
            "outside_line_span": (
                not wall_corners.contains_x(ball)
                if wall_corners is not None
                else (not top_line.contains_x(ball.x) or not bottom_line.contains_x(ball.x))
            ),
        }
    )


def coaching_analytics_for_run(run_id):
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return None, error_response("Run was not found.", status=404)

    hits_path = run_dir / "detected_hits.json"
    if not hits_path.exists():
        return None, error_response("Run analytics were not found.", status=404)

    try:
        payload = json.loads(hits_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, error_response("Run analytics could not be read.", status=500)

    return build_coaching_analytics(payload), None


def coaching_response_payload(analytics, llm_provider, llm_report=None, llm_status="local_only"):
    source = (
        "ollama"
        if llm_report and llm_provider == "ollama"
        else "llm" if llm_report
        else "local"
    )
    if llm_report:
        feedback = llm_report["summary"]
        player_feedback = {
            player_number: format_llm_player_feedback(player_report)
            for player_number, player_report in llm_report["players"].items()
        }
    else:
        feedback = local_coaching_feedback(analytics)
        player_feedback = {
            str(player.get("player_number")): local_player_coaching_feedback(player)
            for player in analytics.get("players", [])
            if player.get("player_number") is not None
        }

    # Drill progressions (solo -> drills -> conditioned games -> matchplay)
    # for whatever weaknesses each player's metrics expose. Derived from the
    # metrics, so they stand on their own whether or not an LLM answered.
    player_drills = {
        str(player.get("player_number")): player_advice(player)
        for player in analytics.get("players", [])
        if player.get("player_number") is not None
    }

    return {
        "ok": True,
        "analytics": analytics,
        "feedback": feedback,
        "feedback_source": source,
        "player_feedback": player_feedback,
        "player_feedback_source": source,
        "player_drills": player_drills,
        "llm_provider": llm_provider,
        "llm_status": llm_status,
    }


POOLED_ADVICE_SESSION_LIMIT = 8
POOLED_COACH_CACHE = {}
POOLED_COACH_CACHE_LOCK = threading.Lock()


def recent_runs_with_analytics(limit, identified_only=False):
    """Newest-first (run_id, payload) pairs for runs that have detected hits."""
    if not RUNS_DIR.is_dir():
        return []

    candidates = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name == "uploads":
            continue
        hits_path = run_dir / "detected_hits.json"
        if not hits_path.exists():
            continue
        created = (
            int(run_dir.name) / 1000.0
            if run_dir.name.isdigit()
            else hits_path.stat().st_mtime
        )
        candidates.append((created, run_dir.name, hits_path, run_dir / "job.json"))
    candidates.sort(reverse=True)

    loaded = []
    for created, run_id, hits_path, job_path in candidates:
        if len(loaded) >= limit:
            break
        try:
            payload = json.loads(hits_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            job = {}
        # A run with no front-wall contact contributes nothing but would still
        # burn one of the session slots.
        if not front_wall_hits_from_payload(payload):
            continue
        selected_player = job.get("user_player_number")
        if selected_player not in (1, 2):
            selected_player = None
        if identified_only and selected_player is None:
            continue
        player_names = ((job.get("players_v1") or {}).get("player_names") or {})
        loaded.append({
            "run_id": run_id,
            "created": created,
            "payload": payload,
            "user_player_number": selected_player,
            "player_names": player_names,
        })
    return loaded


def pooled_player_coaching(
    limit=POOLED_ADVICE_SESSION_LIMIT, identified_runs=None
):
    """Analytics built from several sessions at once.

    Pooling is by *player slot*, not by person: player_number comes from each
    run's own serve-alternation attribution, so Player 1 is whoever served
    first in that clip. The caller must surface that; see `pooling_note`.
    """
    runs = recent_runs_with_analytics(limit)
    if identified_runs is None:
        identified_runs = recent_runs_with_analytics(
            limit, identified_only=True
        )

    sessions = []
    hits_by_player = {1: [], 2: []}
    rallies = []
    selected_hits = []
    selected_sessions = []
    selected_error_metrics = []
    selected_name = None
    for run in runs:
        payload = run["payload"]
        run_hits = front_wall_hits_from_payload(payload)
        run_rallies = (
            payload.get("rallies")
            or (payload.get("player_assignment") or {}).get("rallies")
            or []
        )
        rallies.extend(run_rallies)
        for player_number in (1, 2):
            hits_by_player[player_number].extend(
                hit for hit in run_hits
                if int(hit.get("player_number") or 0) == player_number
            )
        sessions.append({
            "run_id": run["run_id"],
            "created": run["created"],
            "wall_hits": len(run_hits),
        })
    for run in identified_runs:
        payload = run["payload"]
        run_hits = front_wall_hits_from_payload(payload)
        run_rallies = (
            payload.get("rallies")
            or (payload.get("player_assignment") or {}).get("rallies")
            or []
        )
        selected_player = run.get("user_player_number")
        if selected_player in (1, 2):
            personal_hits = [
                hit for hit in run_hits
                if int(hit.get("player_number") or 0) == selected_player
            ]
            selected_hits.extend(personal_hits)
            selected_error_metrics.append(
                player_error_metrics(run_rallies, selected_player, run_hits)
            )
            track = "A" if selected_player == 1 else "B"
            name = run.get("player_names", {}).get(track)
            if name and selected_name is None:
                selected_name = name
            selected_sessions.append({
                "run_id": run["run_id"],
                "created": run["created"],
                "player_number": selected_player,
                "player_name": name,
                "wall_hits": len(personal_hits),
            })

    players = []
    for player_number in (1, 2):
        player_hits = hits_by_player[player_number]
        summary = build_target_zone_summary(player_hits)
        analytics = coaching_analytics_for_hits(player_hits, summary)
        analytics["player_number"] = player_number
        analytics["label"] = f"Player {player_number}"
        analytics.update(player_error_metrics(rallies, player_number))
        players.append({
            "player_number": player_number,
            "label": analytics["label"],
            "analytics": analytics,
        })

    personal_summary = build_target_zone_summary(selected_hits)
    personal_analytics = coaching_analytics_for_hits(
        selected_hits, personal_summary
    )
    personal_analytics["label"] = selected_name or "You"
    personal_analytics["player_number"] = None
    for key in (
        "unforced_errors",
        "unforced_errors_out",
        "unforced_errors_tin",
        "forced_errors",
        "total_errors",
    ):
        personal_analytics[key] = sum(
            int(metrics.get(key) or 0) for metrics in selected_error_metrics
        )
    personal_analytics["unforced_error_percentage"] = (
        rounded(
            personal_analytics["unforced_errors"]
            / personal_analytics["total_errors"] * 100,
            1,
        )
        if personal_analytics["total_errors"]
        else None
    )
    won_durations = [
        metrics.get("average_rally_duration_seconds")
        for metrics in selected_error_metrics
        if metrics.get("average_rally_duration_seconds") is not None
    ]
    personal_analytics["average_rally_duration_seconds"] = rounded(
        average(won_durations), 1
    )
    me = {
        "label": personal_analytics["label"],
        "session_count": len(selected_sessions),
        "sessions": selected_sessions,
        "analytics": personal_analytics,
    }

    return {
        "sessions": sessions,
        "session_count": len(sessions),
        "players": players,
        "me": me,
        "me_pooling_note": (
            f"Built from {len(selected_sessions)} identified "
            f"{'session' if len(selected_sessions) == 1 else 'sessions'}. "
            "Only matches where you selected your player are included."
        ),
        "pooling_note": (
            "Pooled across your last "
            f"{len(sessions)} {'session' if len(sessions) == 1 else 'sessions'} "
            "by player slot — Player 1 is whoever served first in each clip, so "
            "this assumes the same person served first every time."
        ),
    }


def multi_match_coach_schema():
    drill_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 80},
            "evidence": {"type": "string", "maxLength": 220},
            "setup": {"type": "string", "maxLength": 220},
            "work": {"type": "string", "maxLength": 320},
            "dose": {"type": "string", "maxLength": 120},
            "success_measure": {"type": "string", "maxLength": 180},
            "match_application": {"type": "string", "maxLength": 180},
        },
        "required": [
            "name",
            "evidence",
            "setup",
            "work",
            "dose",
            "success_measure",
            "match_application",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "headline": {"type": "string", "maxLength": 90},
            "summary": {"type": "string", "maxLength": 500},
            "trend_observations": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "minItems": 1,
                "maxItems": 3,
            },
            "drills": {
                "type": "array",
                "items": drill_schema,
                "minItems": 2,
                "maxItems": 2,
            },
            "next_match_focus": {"type": "string", "maxLength": 220},
        },
        "required": [
            "headline",
            "summary",
            "trend_observations",
            "drills",
            "next_match_focus",
        ],
        "additionalProperties": False,
    }


def parse_multi_match_coach_report(text):
    try:
        report = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    for key in ("headline", "summary", "next_match_focus"):
        if not isinstance(report.get(key), str) or not report[key].strip():
            return None
    trends = report.get("trend_observations")
    if (
        not isinstance(trends, list)
        or not trends
        or any(not isinstance(item, str) or not item.strip() for item in trends)
    ):
        return None
    drills = report.get("drills")
    if not isinstance(drills, list) or len(drills) != 2:
        return None
    drill_fields = (
        "name",
        "evidence",
        "setup",
        "work",
        "dose",
        "success_measure",
        "match_application",
    )
    for drill in drills:
        if not isinstance(drill, dict):
            return None
        if any(
            not isinstance(drill.get(key), str) or not drill[key].strip()
            for key in drill_fields
        ):
            return None
    return report


def identified_player_match_history(runs):
    """Compact identified-player analytics, ordered oldest to newest."""
    history = []
    for run in reversed(runs):
        player_number = run.get("user_player_number")
        analytics = build_coaching_analytics(run["payload"])
        player = next(
            (
                candidate
                for candidate in analytics.get("players", [])
                if candidate.get("player_number") == player_number
            ),
            None,
        )
        if not player:
            continue
        compact = compact_coaching_analytics({"players": [player]})["players"]
        if not compact:
            continue
        track = "A" if player_number == 1 else "B"
        history.append({
            "match_order": len(history) + 1,
            "run_id": run["run_id"],
            "created_unix_seconds": rounded(run["created"], 3),
            "player_name": run.get("player_names", {}).get(track),
            "metrics": compact[0],
        })
    return history


def multi_match_coaching_messages(history, pooled_analytics):
    pooled = compact_coaching_analytics(
        {"players": [pooled_analytics]}
    )["players"]
    evidence = {
        "ordering": "oldest_to_newest",
        "matches": history,
        "pooled_metrics": pooled[0] if pooled else {},
        "measurement_notes": {
            "total_wall_hits_means": "shots analyzed",
            "wall_height_means": "front-wall impact height",
            "speed_is_estimated": True,
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a practical squash coach reviewing one player's match "
                "history. The matches are supplied oldest to newest. Compare them "
                "in that order, prioritizing repeated weaknesses and the most "
                "recent evidence. Use only measured data; never invent technique, "
                "shot type, handedness, player movement, or causation. Cite concrete "
                "numbers and match order when useful. Give exactly two drills that "
                "a player can execute, each with setup, work, dosage, a measurable "
                "success test, and how to apply it in the next match. If samples are "
                "small or a metric is missing, state that limitation instead of "
                "guessing. Call front-wall contacts 'shots analyzed'."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this chronological match history. Explain what is improving, "
                "what keeps recurring, and the highest-value next action. Return the "
                "structured coaching report only:\n"
                + json.dumps(evidence, separators=(",", ":"))
            ),
        },
    ]


def ollama_multi_match_coaching_feedback(history, pooled_analytics):
    model = os.getenv("OLLAMA_COACH_MODEL", DEFAULT_OLLAMA_COACH_MODEL).strip()
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).strip().rstrip("/")
    try:
        timeout = max(
            1.0, float(os.getenv("OLLAMA_COACH_TIMEOUT_SECONDS", "120"))
        )
    except ValueError:
        timeout = 120.0
    try:
        max_tokens = max(
            700, int(os.getenv("OLLAMA_COACH_MAX_TOKENS", "1200"))
        )
    except ValueError:
        max_tokens = 1200

    messages = multi_match_coaching_messages(history, pooled_analytics)
    schema = multi_match_coach_schema()
    cache_payload = {
        "model": model,
        "base_url": base_url,
        "messages": messages,
        "schema": schema,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with POOLED_COACH_CACHE_LOCK:
        cached = POOLED_COACH_CACHE.get(cache_key)
    if cached is not None:
        return cached, "ok"

    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": schema,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens,
            "num_ctx": 16384,
        },
    }).encode("utf-8")
    request_obj = urllib.request.Request(
        f"{base_url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        status = (
            "ollama_model_missing"
            if error.code == 404
            else "ollama_request_failed"
        )
        return None, status
    except json.JSONDecodeError:
        return None, "invalid_response"
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, "ollama_unavailable"

    if not isinstance(data, dict):
        return None, "invalid_response"
    if data.get("done_reason") == "length":
        return None, "truncated_response"
    text = ((data.get("message") or {}).get("content") or "").strip()
    report = parse_multi_match_coach_report(text)
    if not report:
        return None, "invalid_response"
    with POOLED_COACH_CACHE_LOCK:
        if len(POOLED_COACH_CACHE) >= 16:
            POOLED_COACH_CACHE.pop(next(iter(POOLED_COACH_CACHE)))
        POOLED_COACH_CACHE[cache_key] = report
    return report, "ok"


@app.get("/api/coach/advice")
def coach_advice():
    """Ollama coaching over the user's identified matches in time order."""
    try:
        limit = int(request.args.get("sessions", POOLED_ADVICE_SESSION_LIMIT))
    except (TypeError, ValueError):
        return error_response("sessions must be a whole number.", status=400)
    if limit < 1 or limit > 50:
        return error_response("sessions must be between 1 and 50.", status=400)

    identified_runs = recent_runs_with_analytics(
        limit, identified_only=True
    )
    pooled = pooled_player_coaching(limit, identified_runs=identified_runs)
    history = identified_player_match_history(identified_runs)
    provider = configured_coach_provider()
    if not history:
        report, llm_status = None, "no_identified_sessions"
    elif provider != "ollama":
        report, llm_status = None, "ollama_not_configured"
    else:
        report, llm_status = ollama_multi_match_coaching_feedback(
            history, pooled["me"]["analytics"]
        )
    return jsonify({
        "ok": True,
        **pooled,
        "coach": report,
        "coach_match_order": [match["run_id"] for match in history],
        "coach_match_ordering": "oldest_to_newest",
        "llm_provider": provider,
        "llm_status": llm_status,
    })


@app.get("/api/runs/<run_id>/coach")
def coach_run(run_id):
    """Return deterministic analytics and local feedback without waiting."""
    analytics, error = coaching_analytics_for_run(run_id)
    if error is not None:
        return error

    llm_provider = configured_coach_provider()
    llm_status = "pending" if llm_provider in ("openai", "ollama") else (
        "local_only" if llm_provider == "local" else "invalid_provider"
    )
    return jsonify(coaching_response_payload(
        analytics,
        llm_provider,
        llm_status=llm_status,
    ))


@app.post("/api/runs/<run_id>/coach/llm")
def coach_run_llm(run_id):
    """Generate the slower LLM report after the local report is visible."""
    analytics, error = coaching_analytics_for_run(run_id)
    if error is not None:
        return error

    llm_provider = configured_coach_provider()
    llm_report, llm_status = llm_coaching_feedback(analytics)
    return jsonify(coaching_response_payload(
        analytics,
        llm_provider,
        llm_report=llm_report,
        llm_status=llm_status,
    ))


@app.get("/api/runs")
def list_runs():
    """Index of stored runs, newest first — the cheap facts a run list needs."""
    runs = []
    if RUNS_DIR.is_dir():
        for run_dir in RUNS_DIR.iterdir():
            if not run_dir.is_dir() or run_dir.name == "uploads":
                continue
            job_path = run_dir / "job.json"
            if not job_path.exists():
                continue
            try:
                job = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            fps = job.get("fps") or 0
            start = job.get("start_frame")
            end = job.get("end_frame")
            duration = None
            if fps and isinstance(start, int) and isinstance(end, int) and end >= start:
                duration = round((end - start + 1) / float(fps), 1)
            created = (
                int(run_dir.name) / 1000.0 if run_dir.name.isdigit() else job_path.stat().st_mtime
            )
            runs.append({
                "run_id": run_dir.name,
                "created": created,
                "duration_seconds": duration,
                "status": job.get("status"),
                "has_analytics": (run_dir / "detected_hits.json").exists(),
                "user_player_number": (
                    job.get("user_player_number")
                    if job.get("user_player_number") in (1, 2)
                    else None
                ),
                "player_names": (
                    (job.get("players_v1") or {}).get("player_names")
                    or {"A": None, "B": None}
                ),
                # Which analysis tiers this run actually ran. Empty for runs
                # made before capability gating -- a list where every row looks
                # identical is not a list worth reading, and "we don't know"
                # has to be visible as its own answer.
                "tiers_enabled": match_report.tiers_enabled(job),
            })
    runs.sort(key=lambda entry: entry["created"], reverse=True)
    return jsonify({"ok": True, "runs": runs})


@app.delete("/api/runs/<run_id>")
def delete_run(run_id):
    """Permanently remove one completed analysis, never its shared upload."""
    safe_run_id = secure_filename(run_id)
    if not safe_run_id or safe_run_id != run_id:
        return error_response("Run ID is invalid.", status=400)

    run_dir = RUNS_DIR / safe_run_id
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    job = get_job(safe_run_id)
    if job and job.get("status") in {"queued", "running"}:
        return error_response(
            "A session cannot be deleted while analysis is running.",
            status=409,
        )

    try:
        # video_path normally points into ui_runs/uploads/by-hash. Removing
        # only run_dir preserves that shared source for other analyses.
        shutil.rmtree(run_dir)
    except OSError as error:
        return error_response(f"Run could not be deleted: {error}", status=500)

    forget_job(safe_run_id)
    with BALL_POSITIONS_LOCK:
        BALL_POSITIONS_CACHE.pop(safe_run_id, None)
        RUN_HITS_CACHE.pop(safe_run_id, None)
    return jsonify({"ok": True, "run_id": safe_run_id})


@app.get("/api/runs/<run_id>/report")
def run_report(run_id):
    """report-v1 for one run: every tier it ran, and why it skipped the rest."""
    run_dir = RUNS_DIR / secure_filename(run_id)
    try:
        report = match_report.build_report(
            run_dir, coach_builder=report_coach_builder
        )
    except FileNotFoundError:
        return error_response("Run was not found.", status=404)
    except (OSError, json.JSONDecodeError) as error:
        return error_response(f"Run could not be read: {error}", status=500)

    return jsonify({"ok": True, "report": report})


def report_coach_builder(detected):
    """Coaching analytics for a report, from an already-loaded detected_hits.

    Deliberately the derived analytics only, never the LLM narration: a report
    is fetched on every view, and narration is a network call with a cost and a
    latency nobody asked for when they opened a page. The Coach tab still asks
    for the narrated version explicitly.
    """
    return build_coaching_analytics(detected)


@app.get("/api/runs/<run_id>/<path:filename>")
def run_file(run_id, filename):
    run_dir = RUNS_DIR / secure_filename(run_id)
    return send_from_directory(run_dir, filename, as_attachment=False)


@app.get("/api/runs/<run_id>/source_video")
def run_source_video(run_id):
    """Stream a run's source clip so the UI can rehydrate after a page refresh.

    The clip lives outside the run dir (uploads/by-hash or uploads/<run_id>),
    so resolve it through the persisted job rather than run_file above.
    send_file(conditional=True) honors Range requests, which <video> seeking
    needs.
    """
    job = get_job(secure_filename(run_id))
    if job is None:
        return error_response("Run was not found.", status=404)

    video_path = job.get("video_path")
    if not video_path or not Path(video_path).exists():
        return error_response("Source clip is no longer on the server.", status=404)

    return send_file(Path(video_path), conditional=True)


@app.get("/api/dev/index-mtime")
def index_mtime():
    """Dev live-reload probe: the UI polls this and reloads the tab on change."""
    try:
        mtime = (ROOT / "index.html").stat().st_mtime_ns
    except OSError:
        mtime = 0
    return jsonify({"ok": True, "mtime": mtime, "debug": bool(app.debug)})


GROUND_TRUTH_TYPES = {"wall", "racket", "floor", "side_wall"}


@app.post("/api/runs/<run_id>/ground_truth")
def save_ground_truth(run_id):
    """Persist user-labeled bounce events for a run (the eval ground truth)."""
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    data = request.get_json(silent=True) or {}
    events = data.get("events")
    if not isinstance(events, list):
        return error_response("Body must include an events list.")

    cleaned = []
    for event in events:
        try:
            frame = int(event["frame"])
            kind = str(event["type"])
        except (KeyError, TypeError, ValueError):
            return error_response("Each event needs an integer frame and a type.")
        if kind not in GROUND_TRUTH_TYPES:
            return error_response(f"Unknown event type: {kind}")
        cleaned.append({"frame": frame, "type": kind})
    cleaned.sort(key=lambda event: event["frame"])

    payload = {"tolerance_frames": 1, "events": cleaned}
    (run_dir / "ground_truth.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return jsonify({"ok": True, "count": len(cleaned)})


PLAYER_NAME_MAX_CHARS = 40
PLAYER_PROFILE_SCHEMA = "player-profile-v1"


@app.post("/api/runs/<run_id>/me")
def save_user_player(run_id):
    """Identify which attributed player represents the app's user.

    The choice is stored per run because Player 1 means "served first in this
    clip", not one stable person across every match.
    """
    safe_run_id = secure_filename(run_id)
    if not safe_run_id or safe_run_id != run_id:
        return error_response("Run ID is invalid.", status=400)
    run_dir = RUNS_DIR / safe_run_id
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Body must be a JSON object.")
    try:
        player_number = int(data.get("player_number"))
    except (TypeError, ValueError):
        return error_response("player_number must be 1 or 2.")
    if player_number not in (1, 2):
        return error_response("player_number must be 1 or 2.")

    payload = {
        "schema": PLAYER_PROFILE_SCHEMA,
        "player_number": player_number,
        "track": "A" if player_number == 1 else "B",
        "selected_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    try:
        (run_dir / "player_profile.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except OSError as error:
        return error_response(
            f"Player selection could not be saved: {error}", status=500
        )
    update_job(safe_run_id, user_player_number=player_number)
    return jsonify({"ok": True, **payload})


@app.post("/api/runs/<run_id>/players")
def save_player_names(run_id):
    """Post-hoc naming: map anonymous tracks A/B to typed names. Pure run
    metadata — analysis never re-runs (spec §4.5)."""
    # Sanitize once and reuse everywhere below. get_job/update_job build their
    # own RUNS_DIR/<run_id>/job.json path from whatever id they're given
    # (job_runner.get_job, job_runner.update_job) -- passing the raw run_id
    # there while the directory check above uses the sanitized one means a
    # run_id needing sanitization resolves to two different paths: the run
    # dir exists (found via the sanitized path), but get_job(raw) can't find
    # job.json and returns None, so the players_v1 update below silently
    # no-ops. Worse, if that raw id were ever passed to update_job directly,
    # its rehydrate-from-disk miss would create a permanent empty in-memory
    # JOBS[raw_run_id] stub under a key nothing else ever looks up.
    run_id = secure_filename(run_id)
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    # request.get_json(silent=True) returns None for a missing/invalid-JSON
    # body. Checking this before the old `or {}` fallback matters: `or {}`
    # would turn that None into an empty dict, which then sails through the
    # isinstance(dict) check below and silently clears both names (every
    # `data.get(track)` misses -> None) even though the request was
    # malformed, not an intentional clear. An explicit `{"A": null, "B":
    # null}` body still clears both names -- that's real and stays.
    data = request.get_json(silent=True)
    if data is None:
        return error_response("Body must be a JSON object.")
    if not isinstance(data, dict):
        return error_response("Body must be a JSON object.")
    if any(key not in ("A", "B") for key in data):
        return error_response("Only tracks A and B can be named.")

    names = {}
    for track in ("A", "B"):
        value = data.get(track)
        if value is None:
            names[track] = None
            continue
        if not isinstance(value, str):
            return error_response("Names must be strings or null.")
        value = value.strip()
        if not value or len(value) > PLAYER_NAME_MAX_CHARS:
            return error_response(
                f"Names must be 1-{PLAYER_NAME_MAX_CHARS} characters."
            )
        names[track] = value

    (run_dir / "player_names.json").write_text(
        json.dumps(names, indent=2), encoding="utf-8"
    )
    job = get_job(run_id)
    if job and isinstance(job.get("players_v1"), dict):
        players_v1 = dict(job["players_v1"])
        players_v1["player_names"] = names
        update_job(run_id, players_v1=players_v1)
    return jsonify({"ok": True, "player_names": names})


CORRECTION_SCHEMA_VERSION = "corrections-v2"
CORRECTION_TYPES = GROUND_TRUTH_TYPES | {"none"}
CORRECTION_CALLS = {"IN", "OUT"}


def parse_ball_point(value):
    """-> {x, y} floats, or raise ValueError. None passes through."""
    if value is None:
        return None
    try:
        return {"x": float(value["x"]), "y": float(value["y"])}
    except (KeyError, TypeError, ValueError):
        raise ValueError("ball must be {x, y} numbers")


def parse_corrected(data):
    """Validate the human half of a correction -> dict, or raise ValueError.

    type drives which other fields are legal: call is wall-only, and a
    'none' (detector false positive) carries no position or timing —
    there is no bounce to locate."""
    hit_type = str(data.get("type", "")).lower()
    if hit_type not in CORRECTION_TYPES:
        raise ValueError(
            "corrected.type must be one of " + ", ".join(sorted(CORRECTION_TYPES))
        )

    call = data.get("call")
    if hit_type == "wall":
        call = None if call is None else str(call).upper()
        if call not in CORRECTION_CALLS:
            raise ValueError("corrected.call must be IN or OUT for wall hits.")
    elif call is not None:
        raise ValueError("corrected.call applies to front-wall corrections only.")

    if hit_type == "none":
        if data.get("ball") is not None:
            raise ValueError("a 'none' correction cannot carry a ball position.")
        return {"type": hit_type, "call": None, "ball": None,
                "frame_is_bounce": None, "frame": None}

    ball = parse_ball_point(data.get("ball"))
    if ball is None:
        raise ValueError("corrected.ball {x, y} is required unless type is none.")
    frame_is_bounce = data.get("frame_is_bounce")
    if not isinstance(frame_is_bounce, bool):
        raise ValueError("corrected.frame_is_bounce must be true or false.")
    corrected_frame = data.get("frame")
    if frame_is_bounce:
        if corrected_frame is not None:
            raise ValueError("corrected.frame is only valid when "
                             "frame_is_bounce is false.")
    else:
        try:
            corrected_frame = int(corrected_frame)
        except (TypeError, ValueError):
            raise ValueError("corrected.frame (int) is required when "
                             "frame_is_bounce is false.")
    return {"type": hit_type, "call": call, "ball": ball,
            "frame_is_bounce": frame_is_bounce, "frame": corrected_frame}


def parse_predicted(data):
    """Normalize the label-time model snapshot -> dict, or raise ValueError.
    Every field is nullable; this input is programmatic, so malformed
    values are client bugs and rejected rather than coerced to null."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("predicted must be an object.")
    hit_type = data.get("type")
    if hit_type is not None:
        hit_type = str(hit_type).lower()
        if hit_type not in CORRECTION_TYPES - {"none"}:
            raise ValueError("predicted.type must be a detector hit type.")
    call = data.get("call")
    if call is not None:
        call = str(call).upper()
        if call not in CORRECTION_CALLS | {"UNKNOWN"}:
            raise ValueError("predicted.call must be IN, OUT, or UNKNOWN.")
    margin = data.get("margin_px")
    if margin is not None:
        try:
            margin = float(margin)
        except (TypeError, ValueError):
            raise ValueError("predicted.margin_px must be a number.")
    source = data.get("source")
    return {"type": hit_type, "call": call,
            "source": None if source is None else str(source),
            "margin_px": margin, "ball": parse_ball_point(data.get("ball"))}


def correction_agreement(corrected, predicted):
    """Server-derived agreement flags; null where comparison is undefined."""
    return {
        "type": (None if predicted["type"] is None
                 else predicted["type"] == corrected["type"]),
        "call": (predicted["call"] == corrected["call"]
                 if corrected["type"] == "wall" and corrected["call"]
                 and predicted["call"] in CORRECTION_CALLS else None),
        "frame": corrected["frame_is_bounce"],
    }


@app.get("/api/runs/<run_id>/corrections")
def get_corrections(run_id):
    """The UI's load path: an empty list (not a 404) when nothing is recorded
    yet, so every fresh run doesn't log a console error."""
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)
    try:
        data = json.loads((run_dir / "corrections.json").read_text(encoding="utf-8"))
        corrections = data.get("corrections", [])
        schema_version = data.get("schema_version")
    except (OSError, json.JSONDecodeError):
        corrections = []
        schema_version = None
    return jsonify({"ok": True, "schema_version": schema_version,
                    "corrections": corrections})


@app.post("/api/runs/<run_id>/corrections")
def save_correction(run_id):
    """Record a human bounce correction: hit type, ball position, and bounce
    timing (plus IN/OUT for wall hits). One correction per frame, latest
    wins; corrected null removes the frame's entry (undo). corrections.json
    is the raw feed for the eval set, so each entry keeps the model's
    label-time snapshot alongside the human values."""
    run_dir = RUNS_DIR / secure_filename(run_id)
    if not run_dir.is_dir():
        return error_response("Run was not found.", status=404)

    data = request.get_json(silent=True) or {}
    try:
        frame = int(data["frame"])
    except (KeyError, TypeError, ValueError):
        return error_response("Correction needs an integer frame.")

    corrected_data = data.get("corrected")
    entry = None
    if corrected_data is not None:
        if not isinstance(corrected_data, dict):
            return error_response("corrected must be an object or null to undo.")
        try:
            corrected = parse_corrected(corrected_data)
            predicted = parse_predicted(data.get("predicted"))
        except ValueError as error:
            return error_response(str(error))
        entry = {
            "frame": frame,
            "corrected": corrected,
            "predicted": predicted,
            "agrees": correction_agreement(corrected, predicted),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "app_version": APP_VERSION,
        }

    corrections_path = run_dir / "corrections.json"
    try:
        existing = json.loads(corrections_path.read_text(encoding="utf-8"))
        corrections = existing.get("corrections", [])
    except (OSError, json.JSONDecodeError):
        corrections = []
    corrections = [c for c in corrections if c.get("frame") != frame]
    if entry is not None:
        corrections.append(entry)

    corrections.sort(key=lambda c: c.get("frame", 0))
    corrections_path.write_text(
        json.dumps({"schema_version": CORRECTION_SCHEMA_VERSION,
                    "corrections": corrections}, indent=2),
        encoding="utf-8",
    )
    return jsonify({"ok": True, "count": len(corrections), "correction": entry})


@app.post("/api/label_runs")
def create_label_run():
    """A run directory for labeling only: video reference and frame range,
    no tracking. The id is deterministic per video hash so every labeling
    session for the same clip lands in one place, and a later tracking or
    training pass can locate the source video from label_run.json."""
    data = request.get_json(silent=True) or {}
    video_id = secure_filename(str(data.get("video_id", "")).strip())
    if not video_id:
        return error_response("Missing video_id.")
    matches = sorted(BY_HASH_DIR.glob(f"{video_id}.*"))
    if not matches:
        return error_response("Uploaded video was not found. Upload it again.", status=404)
    try:
        info = video_info(matches[0])
    except ValueError as error:
        return error_response(str(error))

    run_id = f"label-{video_id[:12]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "video_path": str(matches[0]),
        "fps": info["fps"],
        "start_frame": 0,
        "end_frame": max(0, int(info["frame_count"]) - 1),
        "label_only": True,
    }
    (run_dir / "label_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return jsonify(meta)


@app.get("/fonts/<path:filename>")
def font_file(filename):
    return send_from_directory(ROOT / "fonts", filename, max_age=86400)


@app.get("/design-lab/<path:filename>")
def design_lab_file(filename):
    """Serve the design-lab prototypes same-origin so they can call the API."""
    return send_from_directory(ROOT / "design-lab", filename)


if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    host = os.getenv("HOST", "127.0.0.1")
    # Not 5000: macOS AirPlay Receiver commonly owns it and can intercept
    # `localhost` with an AirTunes 403 even while Flask is bound on IPv4.
    # 5188 is the number README, CLAUDE.md, and the /verify skill all cite.
    port = int(os.getenv("PORT", "5188"))
    print(f"Starting SquashAnalytics {APP_VERSION} from {ROOT}")
    print(f"TRACKING_BACKEND={TRACKING_BACKEND} DEFAULT_DEVICE={os.environ.get('DEFAULT_DEVICE')}")
    print(f"ONNXRUNTIME_EXECUTION_PROVIDERS={os.environ.get('ONNXRUNTIME_EXECUTION_PROVIDERS')}")
    print(f"Open http://127.0.0.1:{port}/ on this Mac.")
    if host == "127.0.0.1":
        print(f"For phone access, restart with: HOST=0.0.0.0 PORT={port} python app.py")
    app.run(host=host, port=port, debug=False)
