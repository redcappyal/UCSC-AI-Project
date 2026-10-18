# Judge run-through checklist — Congressional App Challenge

Written 2026-10-17 before the walk-through, scored during it. The run used the web app
at phone width (375 px) in the desktop app's built-in browser, Flask served from this
worktree on Windows, one upload of `SquashAnalytics.mp4` (5:12, 1080p60), calibration
on the frame at 0:55, and a 10 s analysis window (0:50–1:00, 601 frames, hosted RF-DETR).
Nothing here is about detection accuracy, which `/eval` owns.

Legend: `[x]` pass · `[!]` fail or needs work, see Findings · `[ ]` not checked.

## A. First launch (empty app)

- [x] A1 Dashboard boots instantly with zero console errors (favicon 404 excepted).
- [!] A2 Fits 375 px with no horizontal scroll, but `body.phase-load main` centers the
      content vertically, leaving a ~115 px dead band under the header on the empty
      Dashboard and still ~100 px once Coach Notes exist. See F8.
- [x] A3 Empty states are guidance, not blank pages, on all four roots.
- [x] A4 Theme toggle flips both ways, persists in localStorage, both themes legible on
      Dashboard, Analysis, Progress (not screenshot-verified on the review page).
- [x] A5 Nav dock: four icons, active one highlighted, each lands on the right root.
- [x] A6 DEV row ("Label mode") is hidden under `?shell=1` (iOS). In a plain browser
      judges will see it. Decide: keep for the demo or hide behind a query flag. See F9.
- [ ] A7 Backend-down banner: not exercised.

## B. Upload → calibrate

- [x] B1 40 MB clip loads to Pick frame in ~0.7 s; filmstrip renders; no freeze.
- [x] B2 Pick frame: instruction readable, "Use" pill and title fit at 375 px.
- [!] B3 Court auto-detect declined honestly both times ("Could not find the front
      wall's out line, service line, tin, and floor line together. Tap the lines
      instead."). First call took 56 s (cold model load); second took 7.5 s. The
      overlay shows no time estimate. See F7.
- [!] B4 Tap wizard: out line and service line fit in one tap each (RMS 0.5–0.9 px).
      The tin is the fragile step on this footage: its red line is thin and dark on
      a white board, and the warm wood floor is close enough in hue that a tap a few
      px low floods the floor ("Fill leaked off the line"). Took five attempts and a
      lower tolerance. See F6.
- [x] B4b Review step draws fitted edges and explains span snapping. Wall corners step
      accepts four taps and reports "Wall bounds will be used…".
- [!] B4c Floor map step needs 4+ of 8 landmarks with long labels; I skipped it, which
      the app allows, and the match page then reports "Movement not measured: no
      solved court". For a judge demo this step must be pre-done. See F11.
- [x] B5 Clip: default window is ±2 s around the calibration frame; handles work;
      Analyze pill enabled.

## C. Analysis run

- [!] C1 Progress: the screen read "Analyzing 0%" for about 2 min while the server was
      at "Loading ball detector…" then "Person detector"; frames then ran fast (601
      frames in 45 s). The server's stage message never reaches the Analyzing screen.
      "Processing complete0%" flashes at the end. See F3.
- [x] C2 Hosted RF-DETR reachable with the `.env` key; no errors in the server log.
- [!] C3 Call pane: three wall hits (IN, OUT by 18 px, IN), bar tap seeks, magnifier
      inset shows the ball. But the verdict box prints raw provenance:
      "detected ball center (1239.0, 200.6) · between_out_and_service_lines". See F4.
- [x] C4 Player 1 / Player 2 panes render; header shows the clip name; Back exits.
      Panes are honest about thin data ("Only 0 front-wall shots were analyzed…").
- [!] C5 Coach: header reads "Local feedback · LLM disabled" and the text is the
      no-pattern template. Cause: this machine's `.env` has `COACH_LLM_PROVIDER=local`
      (which means no LLM) and the documented `OLLAMA_COACH_MODEL=qwen3:8b` is not
      installed in Ollama here (qwen3.5:4b and qwen2.5:3b are). See F2.

## D. After the run (populated app)

- [x] D1 Dashboard: Sessions 1, Unforced errors 33 %, ring 1/5, Coach Notes rail, hero
      note hidden. A stale "Loaded SquashAnalytics.mp4." status line sits under the
      button. See F10.
- [!] D2 Analysis lists "1 session" (singular, good) but the card title truncates to
      "Today, 20…" at 375 px, and the match page says "No rallies found (clip has no
      audio track…)" while the Call pane says "3 rallies". The match page also shows
      the raw run id ("run 1792293211606"). See F4, F5.
- [!] D3 Training → Your coach still says "Tell us which player is you" after "This is
      me" was saved; `/api/coach/advice` reports `no_identified_sessions` because the
      session has no usable shots. Progress, meanwhile, says "Your first identified
      session is saved." The two screens disagree about the same state. See F5.
- [x] D4 Progress with one session shows the honest "one more match" state.
- [x] D5 Reload restores the review session and keeps the run list; no re-prompts.

## E. Submission polish

- [!] E1 Static `<h1>` in index.html still says "Squash Line Calling" before JS swaps
      it to "CrossCourt"; `<title>` is right. Server banner says "Open … on this Mac."
      on Windows. See F9.
- [x] E2 No archived-feature copy (stereo/fusion/challenge) in the UI; only comments.
- [!] E3 README paths are Mac-only (`.venv/bin/python`); Windows needs
      `.venv\Scripts\python.exe`, and a worktree needs a venv junction + `.env` copy.
- [!] E4 No saved demo calibration exists. Every run depends on tapping lines live.
      And `SquashAnalytics.mp4` is angled footage of a court with a thin tin line,
      not the fixed back-wall mount the app is designed for. See F1.
- [!] E5 Coach provider and model are documented in README but the running config on
      this box contradicts it (F2).

## Findings, ranked by what a judge would notice

1. **F1 Demo footage and calibration (blocking).** Record or pick a fixed-mount clip
   straight at the front wall with a clearly painted tin, calibrate it once including
   the floor map, and keep that calibration so a live demo never depends on tapping
   lines under time pressure. On the current clip the stats pages are mostly "—" and
   "0 s" because only three contacts were found in 10 s, which makes the product look
   empty.
2. **F2 Coach LLM is off on this machine.** Set `COACH_LLM_PROVIDER=ollama` and
   `OLLAMA_COACH_MODEL` to an installed model (or pull `qwen3:8b`). Then rename the
   user-facing "LLM disabled" badge to plain words.
3. **F3 Two minutes of "Analyzing 0%".** Surface the job's stage message (Loading
   ball detector / Coarse pass / Judging) on the Analyzing screen, and pre-warm the
   detector at server start so the first run is not a 2-minute wait. Fix the
   "Processing complete0%" flash.
4. **F4 Developer text on user screens.** Verdict provenance
   (`between_out_and_service_lines`, pixel coordinates) and the raw run id belong
   behind the §8.23 "why" tap, not in body copy.
5. **F5 Screens disagree.** Rally count (Call pane 3 vs match page 0) and identity
   state (Progress saved vs Training asks again). Pick one source of truth per fact.
6. **F6 Tin tap is fragile** on thin tin lines; add a hint ("tap the thin red line,
   not the board") or seed the tin from the out line and wall-corner geometry.
7. **F7 Auto-detect cold start** of 56 s with no estimate; pre-warm or say "first
   run takes about a minute".
8. **F8 Dashboard dead band** from vertical centering; top-align `p-load` or let the
   Coach Notes rail fill the space.
9. **F9 Names and copy:** static `<h1>` "Squash Line Calling"; "on this Mac" server
   line; decide whether the DEV row shows in the browser demo.
10. **F10 Stale "Loaded <file>." status** persists on the Dashboard after a run.
11. **F11 Floor map** is eight long-label landmarks; the demo calibration should
    include it so Movement and zone stats appear.
12. **F12 Card title truncation** "Today, 20…" at 375 px on the Analysis list.

## Environment notes from this run (Windows)

- Worktrees ship with no usable venv; this one had a partial `.venv` (no torch, no
  onnxruntime). Replaced with a junction to the main checkout's `.venv` and copied
  `.env` from the main checkout. The old folder is left at `.venv-partial/`.
- Ollama is installed (`C:\Users\alann\AppData\Local\Programs\Ollama`) and starts on
  demand; models present: qwen3.5:4b-q4_K_M, qwen2.5:3b.
- Run artifacts for this walk-through: `ui_runs/1792293211606/`.
