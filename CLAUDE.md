# CLAUDE.md

Squash Line Calling — a single-file mobile web app (`index.html`) backed by a Flask
pipeline (`app.py`, `job_runner.py`, `inference_engine.py`) that watches a squash rally
through a fixed phone camera and calls the ball IN or OUT.

## Design

**For all UI and front-end work, strictly follow the rules in [DESIGN.md](DESIGN.md).**

DESIGN.md is the single source of truth for design tokens (colors, type, spacing, radii,
motion), the component library, per-screen blueprints, and the hard "never do" list. Read
it before writing or changing any HTML/CSS/JS that renders UI. If a change requires
deviating from it, update DESIGN.md deliberately in the same change — never drift
silently. Verify UI changes in both themes at a phone viewport (use the `/verify` skill).
