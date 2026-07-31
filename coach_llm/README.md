# coach_llm — prompt iteration for the Ollama coach

The Coach tab's narration comes from a local Ollama model (`COACH_LLM_PROVIDER=ollama`).
This directory is where that prompt gets *tuned*; it is not where it ships from.

## Which prompt actually runs

`app.py`'s `coaching_messages()` (app.py:923) sends its own `system` message, and
`ollama_coaching_feedback()` sends its own `options`. Both **override** anything in a
Modelfile. So:

| Surface | Prompt that governs |
|---|---|
| The app (Coach tab, `/api/runs/<id>/coach/llm`) | `coaching_messages()` in app.py:923 |
| `ollama run crosscourt-coach`, and `try_coach.py`'s right-hand column | `SYSTEM` in `Modelfile` |

The Modelfile exists so a prompt can be rewritten and re-tested for free, without
touching a module that has a test suite behind it. When a Modelfile prompt clearly
beats the shipping one, move it into `coaching_messages()` on purpose.

## Setup

```bash
ollama create crosscourt-coach -f coach_llm/Modelfile
```

`.env` points the app at the base model:

```dotenv
COACH_LLM_PROVIDER=ollama
OLLAMA_COACH_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

**`qwen3:4b`, not the `qwen3:8b` in README.md.** The 8B build is ~5.2 GB against 8 GB of
unified memory on this Mac, and it shares that memory with the ball detector. The 4B at
Q4 is ~2.5 GB and answers a full two-player report in 14–27 s.

## Comparing prompts

```bash
.venv/bin/python coach_llm/try_coach.py 1785451989787
```

Runs one stored run's real analytics through both prompts with the same schema, the same
`temperature`/`num_ctx`/`num_predict`, and `think: false`, then prints them side by side.

## What a 4B model still gets wrong

Measured on run `1785451989787` (9 shots analyzed per player), against the shipping
prompt, which already forbids most of this:

- **Invents technique.** "indicating reduced power transfer" — the pipeline sees ball
  position, not power transfer.
- **Invents setup.** "from 10 feet away" appears in drills; no distance is in the data.
- **Writes goals in mph.** "Increase average exit speed to 8.0 mph" is not something a
  player can practise toward.
- **Praises.** "showing exceptional consistency" breaks DESIGN.md §14 (no praise), off a
  9-shot sample.
- **Skips the small-sample hedge** even when the prompt demands it.

The Modelfile prompt fixes the drill vocabulary (real squash drills, goals in shots
landed) but does not fix the last two. Constraints a 4B model reliably ignores are better
enforced in Python than asked for in a prompt — `coaching_advice.py` already computes
`low_sample_note` deterministically, and the same approach suits the rest.
