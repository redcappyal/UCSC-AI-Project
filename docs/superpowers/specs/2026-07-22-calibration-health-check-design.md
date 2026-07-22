# Calibration Health Check + Floor-Handoff Sigma Gate — Design

Date: 2026-07-22. Approved by Ian (conversation).

## Motivation

The first wizard run after the camera mount is installed will be the first-ever
15-correspondence camera solve on real data. Today the solve runs only at judge time
behind `fusion_3d`; a bad calibration is discovered after the capture session is over.
This feature surfaces solve health live in the wizard so a bad tap is fixed at the court.
Also builds the deferred floor-handoff sigma gate (floor analogue of the wall snap gate).

## Part 1 — Live calibration health check

### Solver: labeled per-point residuals (court_model.py)

- When `solve_camera_model` assembles correspondences, each carries a label:
  `floor:<landmark_id>`, `line:out_left`, `line:out_right`, `line:tin_left`,
  `line:tin_right`, `corner:top_left`, `corner:bottom_right`, etc.
- `info` gains `per_point: [{label, residual_px}]` sorted worst-first, populated on
  success AND on `high_residual` rejection (the case where it matters most).
- Pure addition — existing callers (`job_runner.py:790`) and `camera_warning` unchanged.

### Endpoint: POST /api/camera-check (app.py)

- Body: `calibration_json` — same payload `buildJson()` sends to `/api/track`.
- Runs `solve_camera_model`; returns the info dict as JSON. Always HTTP 200 with a
  `status` field (`ok`, `insufficient_points`, `implausible_geometry`, `high_residual`,
  `no_frame_size`, `init_failed`, ...). Garbage input → 200 with an error status,
  mirroring the solver's never-raise philosophy. Read-only; nothing stored.

### Wizard UI (index.html)

- On any calibration state change (tap, snap refine, undo, corner tap), debounce
  ~400 ms then POST current `buildJson()` to `/api/camera-check`; stale-response token
  discards out-of-order replies.
- Health strip in the wizard, three states:
  - **Pending** (neutral, not an error): "Camera: needs 4+ floor points" — partial
    state is normal mid-wizard.
  - **OK**: "Camera OK · rms 1.4px · 15 pts".
  - **Failed**: gate reason in plain words, e.g. "Camera solve failed — taps don't
    agree (median 9.2px). Worst: front-wall left corner".
- Worst 1–2 offenders get the existing `warned` marker styling on the overlay.
- Follows DESIGN.md; verified both themes at phone viewport.

## Part 2 — Floor-handoff sigma gate (event_engine.py)

- New config `floor_snap_max_sigmas: 3.0` alongside `wall_snap_max_sigmas`.
- In the `judge_hits` floor branch (currently ungated at ~:660): trust a 3D floor
  contact only when its floor-plane distance ≤ `floor_snap_max_sigmas × sigma`,
  sigma derived from arc RMS exactly as the wall branch does. Beyond the gate, fall
  back to the existing 2D path (same behavior as no-3D).

## Testing

- pytest additions: per-point labels present/sorted (test_camera_model.py);
  `/api/camera-check` via Flask test client — good, degraded, garbage payloads;
  floor gate accept/reject with `tests/synthetic3d.py` scenes (test_event_engine.py).
- End-to-end rehearsal: headless-Chromium wizard pass on existing footage exercising
  the live check — the first 15-point solve dry-run on real data.

## Out of scope

- CoreML retry, 1e8 conditioning-threshold validation (blocked on calibrated footage),
  server-side calibration store, corpus rebuild / A-B rerun (blocked on mount).
