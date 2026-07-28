import hashlib
import json
import math
import os
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
from coaching_advice import player_advice
from judge_call import (
    Point,
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
APP_VERSION = "wall-corner-calibration-2026-07-20-1"

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

# inference_engine sets the model-cache/metrics env defaults on import;
# import it (via job_runner) before anything touches the inference package.
from job_runner import (
    RUNS_DIR,
    UPLOADS_DIR,
    build_target_zone_summary,
    create_job,
    get_job,
    is_serve_hit,
    request_cancel,
    start_tracking_job,
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
        "rallies",
        "floor_zones",
        "calibration_warning",
        "hits_error",
        "annotated_video_url",
        "csv_url",
        "error",
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


def player_error_metrics(rallies, player_number):
    """Summarize a player's lost-rally errors and won-rally durations.

    An OUT shot by the loser is an unforced error. When the winner made the
    final IN shot, the loser's failure to return it is a forced error. Rally
    duration is averaged over rallies won by this player so each player's
    value describes their own results.
    """
    unforced_errors = 0
    forced_errors = 0
    won_rally_durations = []
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
        else:
            forced_errors += 1

    total_errors = unforced_errors + forced_errors
    return {
        "unforced_errors": unforced_errors,
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
        player_analytics.update(player_error_metrics(rallies, player_number))
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
                "items": {"type": "string"},
                "description": "Two concise, evidence-based observations for this player.",
            },
            "drill_name": {"type": "string"},
            "drill_instructions": {
                "type": "string",
                "description": "A specific drill setup with repetitions or duration.",
            },
            "drill_goal": {
                "type": "string",
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
                "summary": {
                    "type": "string",
                    "description": "A short overall summary comparing the two players.",
                },
                "players": {
                    "type": "object",
                    "properties": {
                        "1": player_schema,
                        "2": player_schema,
                    },
                    "required": ["1", "2"],
                    "additionalProperties": False,
                },
            },
            "required": ["summary", "players"],
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


def coaching_messages(analytics):
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
                "Use 'shots analyzed' instead of 'front-wall hits'. Wall height means "
                "where the ball struck the front wall; pace is estimated ball speed "
                "around front-wall contact."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create the overall summary and separate Player 1 and Player 2 coaching "
                "reports from these analytics:\n"
                + json.dumps(analytics, indent=2)
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

    body = json.dumps(
        {
            "model": model,
            "messages": coaching_messages(analytics),
            "stream": False,
            "think": False,
            "format": openai_coach_response_format()["schema"],
            "options": {
                "temperature": 0.2,
                "num_predict": 850,
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
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None, "ollama_unavailable"

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
        frame_stride = int(request.form.get("frame_stride", "4"))
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

        call, reason, top_y, bottom_y = judge_ball(ball, top_line, bottom_line, wall_corners)
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
            "margin_px": margin_px,
            "ball": {"x": ball.x, "y": ball.y},
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
            })
    runs.sort(key=lambda entry: entry["created"], reverse=True)
    return jsonify({"ok": True, "runs": runs})


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
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SquashAnalytics {APP_VERSION} from {ROOT}")
    print(f"TRACKING_BACKEND={TRACKING_BACKEND} DEFAULT_DEVICE={os.environ.get('DEFAULT_DEVICE')}")
    print(f"ONNXRUNTIME_EXECUTION_PROVIDERS={os.environ.get('ONNXRUNTIME_EXECUTION_PROVIDERS')}")
    print(f"Open http://127.0.0.1:{port}/ on this Mac.")
    if host == "127.0.0.1":
        print(f"For phone access, restart with: HOST=0.0.0.0 PORT={port} python app.py")
    app.run(host=host, port=port, debug=False)
