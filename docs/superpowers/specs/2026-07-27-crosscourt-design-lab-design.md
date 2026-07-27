# CrossCourt Design Lab — TennisIQ-inspired prototype (design)

Date: 2026-07-27
Status: approved by Ian (pending spec review)

## Purpose

A standalone design exploration — **not** a change to the shipping app. We translate a
TennisIQ-style mobile design (light theme, chartreuse accent, soft white cards) into
CrossCourt's squash training-feedback-loop product and see how it carries the product
vision. Nothing in `index.html` or `DESIGN.md` changes. A decision about adopting any of
this into the real app happens *after* the lab exists, as a separate deliberate step.

## Scope

- One self-contained file: `design-lab/crosscourt-lab.html`.
  - Inline CSS and JS, no CDN or remote assets of any kind, no font downloads
    (system sans stack). Fake data lives inline as JS objects.
  - Renders a 390 px-wide phone frame centered on the page so the browser preview at
    desktop width still shows a faithful phone layout.
  - Light theme only (faithful to the inspiration).
- Four screens behind a working 4-tab bottom nav, order: **Dashboard · Analysis ·
  Training · Progress**. Instant view swaps.
- Every implied affordance does something visible but shallow (expand, tab switch,
  toast); no dead taps.

Out of scope: dark theme, real data, any wiring to the Flask pipeline, DESIGN.md
changes, iOS app changes.

## Visual language (extracted from the inspiration)

- **Canvas:** soft cool gray `#F2F3F5` app background.
- **Cards:** pure white, border-radius 20–24 px, no borders; depth from one very soft
  shadow (e.g. `0 1px 3px rgba(20,22,25,.06)`), stacked with ~12 px gaps.
- **Ink:** primary near-black `#17181A`; secondary gray `#8A8F98`.
- **Accent:** chartreuse `#D9F24F`, always with dark ink on top. Used for: primary
  buttons, active nav/segment states, highlighted chart elements, Start chips, play
  buttons, gauge/score fills.
- **Inverse hero:** near-black (`#17181A`-ish) card with white text; accent reserved for
  the ring/badge and small highlights inside it.
- **Severity chips:** low-saturation tinted chips — soft green (Good), soft amber
  (Average), soft red (Poor). Difficulty tags reuse the same tint family (Easy/Medium/
  Hard).
- **Type:** system sans (`-apple-system, …`); big 700-weight tabular numerals for stats
  (28–32 px hero values with small dim suffixes like `/100`); 12–13 px sentence-case
  labels at 500; no uppercase-paragraph telemetry styling — this is the consumer-warm
  counterpoint to the shipping app.
- **Shape grammar:** rounded-square icon tiles (~36 px, light gray fill) leading list
  rows; pill buttons; dark inset stat tiles inside the hero card.
- **Charts:** Whoop-style smooth trend lines (see Progress) and a segmented arc gauge
  for scores where used in condensed report chips.

## Screens

### 1. Dashboard — "what should I work on next", one glance

Deliberately minimal; no ambition beyond one screenful.

1. Header: CrossCourt wordmark + notification bell (visual only).
2. **Dark hero card — headline recommendation** from the overall player report, in
   coaching voice, e.g. "Focus: straight-drive depth — drives landing short of the
   service box in 3 of your last 4 sessions." Small trend badge (e.g. `IN% 78 ▲3`).
3. Button pair: accent **Start Recording** pill (camera icon) + white **Upload Video**
   pill. Both toast in the lab.
4. **Next up:** the top 1–2 recommended drills as compact rows (icon tile, name,
   duration, Start chip). Start deep-links to the Training tab.

### 2. Analysis — past clips, front door to the existing pipeline

1. Header: title + clip count meta.
2. Clip list, one white card per clip: thumbnail placeholder (inline SVG/gradient — no
   image files), date, duration, and a state tag: `ANALYZED` (with score chip, e.g. 84)
   or `NOT ANALYZED`.
3. Tap a card → expands in place (animated height, one card open at a time).
   - **Unanalyzed clip, expanded:** meta rows + full-width accent **Analyze** button.
     In the real app this hands off into the already-implemented judge/track pipeline;
     in the lab it's a visual stub (brief fake progress → flips the clip to analyzed).
   - **Analyzed clip, expanded — the condensed report:** compact score row (Overall +
     Accuracy / Placement / Consistency chips), top 2–3 feedback lines with severity
     chips, small ghost **Re-analyze** button. There is no separate full-screen report
     in the lab.

Sub-score grounding: Accuracy is IN%-derived (real today), Placement is target-zone
spread (near-real), Consistency is aspirational (rally-length/error blend). Overall
session score is aspirational. North-star content, grounded where possible.

### 3. Training — the full menu of drills

1. Dark **program card**: "Intermediate Program · Week 3 of 8", progress ring in accent.
2. **Today Missions:** 2 rows with icon tiles + Start chips.
3. **Exercises** list: icon tile/thumbnail, name, difficulty tag, duration + sets,
   accent circular play button.

Squash vocabulary throughout: straight-drive depth, boast accuracy, ghosting,
serve/return targets, tin margin. Domain terms per the shipping app (out line, tin,
service line) where they appear.

### 4. Progress — cross-session trends, Whoop-style

The feedback-loop page: same measurables tracked across sessions.

1. **Range selector:** segmented control with exactly `1W · 1M · 3M · 6M` (Whoop's
   grammar). Switching ranges re-renders the charts from inline fake data.
2. **"Since last session" delta strip:** 2–3 compact tiles (IN% ▲3 · Session score ▲5 ·
   Rallies —).
3. **Trend cards (Whoop grammar):** each metric gets a card with:
   - a smooth SVG line (Catmull-Rom/spline smoothing) with a soft area fill under it;
   - a shaded horizontal **baseline band** (the player's typical range) with a dotted
     average line;
   - the latest data point emphasized (accent dot + value label);
   - current value large at the card top-left, delta vs. baseline beside it.
   - Metrics: Session score (primary), Front-wall IN%, Active minutes.
4. **Personal-best callout row** (e.g. "Best IN% to date — 84%, Jul 22").

No bar charts on this page; bars were the earlier draft and were replaced by
Whoop-style lines at Ian's request.

## Interaction summary

- Bottom nav pill, 4 tabs, instant swaps; active tab = accent capsule.
- Analysis cards: accordion expand, one open at a time.
- Analyze button: fake determinate progress, then state flips to analyzed.
- Start chips on Dashboard → switch to Training tab.
- Everything else visible-but-stub (toast or no-op with pressed feedback — but any
  control that looks primary must respond visibly).

## Non-goals / guardrails

- This file must not import from or modify the shipping app.
- If any of this later graduates into `index.html`, that work re-enters through
  DESIGN.md deliberately (new tokens/components documented there first) — the lab's
  style is *not* automatically sanctioned for the app.

## Success criteria

- All four screens render correctly in the browser pane at 390 px width.
- The flow reads as CrossCourt (squash vocabulary, real measurables where possible),
  not as a tennis app reskin.
- Ian can look at it and make the adopt/blend/reject call for the real app.
