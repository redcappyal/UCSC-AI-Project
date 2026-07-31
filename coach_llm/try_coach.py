"""Compare the live coaching prompt against the crosscourt-coach Modelfile.

Both sides get the same run's analytics, the same JSON schema and the same
decoding options, so the only variable is the prompt:

  app    -- app.py's coaching_messages(), the prompt that ships today
  model  -- no system message in the request, so the Modelfile SYSTEM applies

Usage:
    .venv/bin/python coach_llm/try_coach.py <run_id> [--model crosscourt-coach]

Iterate on coach_llm/Modelfile, `ollama create crosscourt-coach -f
coach_llm/Modelfile`, rerun this, and when the right column wins, move that
prompt into app.py:923 deliberately.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    DEFAULT_OLLAMA_URL,
    build_coaching_analytics,
    coaching_messages,
    compact_coaching_analytics,
    openai_coach_response_format,
    parse_llm_coaching_report,
)


def load_analytics(run_id):
    run_dir = ROOT / "ui_runs" / run_id
    detected = json.loads((run_dir / "detected_hits.json").read_text(encoding="utf-8"))
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    # A run writes these to job.json; build_coaching_analytics reads them off the
    # payload it is handed, the way the report endpoint hands them over.
    for key in ("floor_zones", "player_assignment", "rallies"):
        detected.setdefault(key, job.get(key))
    return build_coaching_analytics(detected)


def ask(model, messages, base_url):
    """One /api/chat call with the app's own schema and decoding options."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": openai_coach_response_format()["schema"],
        "options": {"temperature": 0.1, "num_predict": 1200, "num_ctx": 8192},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return None, f"http_{error.code}", time.time() - start
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return None, f"unreachable ({error})", time.time() - start

    elapsed = time.time() - start
    if data.get("done_reason") == "length":
        return None, "truncated (raise OLLAMA_COACH_MAX_TOKENS)", elapsed
    text = ((data.get("message") or {}).get("content") or "").strip()
    report = parse_llm_coaching_report(text) if text else None
    return report, ("ok" if report else "invalid_response"), elapsed


def show(title, report, status, elapsed):
    print(f"\n{'=' * 62}\n{title}   [{status}, {elapsed:.1f}s]\n{'=' * 62}")
    if not report:
        return
    print(f"summary: {report.get('summary')}")
    for number, player in sorted((report.get("players") or {}).items()):
        print(f"\n  Player {number}")
        for observation in player.get("observations") or []:
            print(f"    - {observation}")
        print(f"    drill: {player.get('drill_name')}")
        print(f"      how: {player.get('drill_instructions')}")
        print(f"     goal: {player.get('drill_goal')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--model", default="crosscourt-coach")
    parser.add_argument("--app-model", default="qwen3:4b")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args()

    analytics = load_analytics(args.run_id)
    payload = json.dumps(compact_coaching_analytics(analytics), separators=(",", ":"))
    print(f"run {args.run_id}: {len(payload)} chars of analytics (~{len(payload) // 4} tokens)")

    app_messages = coaching_messages(analytics)
    show(
        f"app prompt (coaching_messages) on {args.app_model}",
        *ask(args.app_model, app_messages, args.base_url),
    )

    # No system message: the Modelfile's SYSTEM block is what governs here.
    show(
        f"Modelfile prompt on {args.model}",
        *ask(args.model, [app_messages[-1]], args.base_url),
    )


if __name__ == "__main__":
    main()
