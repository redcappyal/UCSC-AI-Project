# DESIGN.md — Squash Line Calling

Design system and UI rulebook for the Squash Line Calling app (`index.html`, single-file
mobile web app). This document is the **single source of truth for all UI and front-end
work**. Any agent or human touching the UI must follow it exactly. When code and this
document disagree, fix the code (or update this document deliberately in the same PR —
never silently drift).

The aesthetic in one sentence: **a referee's instrument with a broadcast-sports finish** —
OLED-black stage, one loud accent, neon functional markers over film strips, giant tabular
numerals, capsule controls, and a floating liquid-glass tab bar.

---

## 0. TL;DR — the 12 binding rules

1. **Dark-first, OLED black.** Page background is pure `#000` (dark). Every element must
   also work in the light theme via tokens — never hardcode a themed color.
2. **One accent.** Yellow `--accent-bg: #ffd60a` with black text is the *only* accent.
   Never introduce a second accent; every other color must carry a semantic meaning
   defined in §5.
3. **Color = meaning.** Green/red are reserved for IN/OUT verdicts. Cyan/lime are reserved
   for fitted calibration edges. Neon marker hues are reserved for event types. Never use
   these decoratively, and never let color be the only carrier of a meaning — always pair
   with a text label or icon.
4. **Capsules and cards only.** Controls are pill-shaped (`border-radius:999px`);
   containers use the radius scale in §4.4. No sharp-cornered UI.
5. **Chakra Petch, three weights.** 400 / 600 / 700, self-hosted from `/fonts`. Never load
   a font (or anything else) from a CDN — the app must work offline on court.
6. **44 pt minimum touch targets** (Apple HIG). `min-height:44px` and ≥44 px effective
   width on every tappable control; primary actions are 48 px.
7. **Uppercase controls, sentence-case guidance.** Button/step labels are uppercase with
   `letter-spacing:.05em`; instructions and status text are sentence case in `--dim`.
8. **Tabular numerals everywhere numbers move.** `font-variant-numeric:tabular-nums` on
   timecodes, counters, stats, and readouts.
9. **No layout shift.** Reserve space for anything that appears/disappears (see the
   verdict box pattern, §8.7). Showing a result must never push the page around.
10. **Direct manipulation, video-editor grammar.** Timelines are film strips that slide
    under a **center-fixed white playhead**; trimming uses yellow handles; markers are
    opaque neon bars with a dark hairline and soft glow.
11. **Respect the shell.** Every screen is a phase `<section>` inside the fixed
    header / stage / main / nav-pill shell (§3). Don't invent new navigation chrome.
12. **Verify both themes at 390 × 844** (iPhone-class viewport) before calling UI work
    done. Use the project's `/verify` skill.

---

## 1. Product character & principles

The app watches a squash rally through a fixed phone camera and calls the ball IN or OUT.
It is used **courtside, one-handed, mid-game, often in bad light and with no network**.
Design decisions follow from that:

- **The verdict is the hero.** Everything funnels into one legible answer (IN / OUT / event
  class). The verdict gets the biggest type on screen (28 px+) and a full-bleed color fill.
  Everything else defers to it (HIG: *deference*).
- **Glanceable, then precise.** First read in < 1 s from arm's length (big numerals, high
  contrast, one accent). Precision tools (frame steppers, zoom, labeling) are available but
  visually quiet until needed.
- **Trust through evidence, not decoration.** Show the frame, the fitted lines, the
  trajectory markers — never a spinner where evidence could be shown. Progress is honest
  (real counts: `frames 132/300`, fps, ETA — see §13).
- **Fast beats fancy.** Lesson from SwingVision user research (Figma board): laggy/buggy
  calibration and slow AI features are the #1 complaint. Prefer instant state swaps over
  transitions; animation is for ambient status (pulse) and delight (theme toggle) only.
- **Few stats, well chosen.** Users wanted "simple statistics clearly displayed" — target
  zones, bounce maps, percentages. Big numbers, small labels, no chartjunk.
- **One thing per screen.** Each phase asks exactly one question ("Use this frame?",
  "Looks right?", "Track ball"). The single primary action lives in the header pill.

---

## 2. Aesthetic direction (mood-board translation)

The mood board (sports play-by-play, F1 fantasy dashboard, training-load apps, iOS video
editors) translates into these concrete motifs — all already in the codebase:

| Mood-board motif | Our implementation |
|---|---|
| OLED-black dashboards with one hot accent | `--bg:#000`, yellow accent capsules |
| Giant stat numerals (`121 PTS`, `41.1`) | Verdict 28 px, zone percentages 26 px, 700 weight, tabular |
| Play-by-play event feed with OUT chips | Hit timeline markers + verdict states |
| Video-editor trim UI (yellow handles, filmstrip, white playhead) | `.clipEditor`, `.clipHandle`, center-fixed `#clipCursor` |
| Segmented duration controls (`15/30/60 sec`, `7D/30D/3M`) | `.engineSeg` segmented control pattern |
| Bottom toolbar of icon+label actions | `#navPill` liquid-glass section tab bar |
| Zone/heat charts on a literal court | `.targetCourt` front-wall chart, `#floorMapSvg` bounce map |
| Squared "telemetry" typeface | Chakra Petch |

Tone target: **broadcast sports-tech** (F1/fitness screens), *not* consumer-cute. No
gradients-for-fun, no glassmorphism outside the nav pill, no illustration style. The only
"physical" rendering allowed is the miniature squash court (§8.10), which is deliberately
literal (plaster wall, wood floor, red court lines).

---

## 3. Platform & app shell

Single-file mobile web app (`index.html`) served by Flask, designed to feel like a native
iOS app in Safari (add-to-home-screen capable).

### 3.1 Document-level requirements

- `<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">`
- `<meta name="theme-color" content="#000000">`
- `color-scheme` set on `:root` per theme; theme stored in `localStorage('slc-theme')`,
  defaulting to `prefers-color-scheme`, applied as `data-theme` on `<html>` **before
  first paint** (inline head script — keep it there to avoid theme flash).
- `body{height:100dvh; overflow:hidden}` — the app is a fixed shell, not a scrolling page.
  Individual panels may scroll internally if they must; the shell never does.
- `-webkit-tap-highlight-color:transparent` — we own press feedback (§11).

### 3.2 Shell anatomy (do not restructure)

```
<header>        fixed-height top bar: back chevron · step label · theme toggle · action pill
#instr          one-line contextual instruction strip (dim), directly under header
#stage          flex-growing black canvas area: video frame, overlays, zoom controls
<main>          the current phase <section> (controls, timelines, cards)
#navPill        floating liquid-glass section tab bar, bottom-center (section roots only)
```

- Header: `min-height:56px`, padding `8px 14px`, gap 10. Step label = uppercase 16/700,
  `letter-spacing:.05em`, ellipsizes.
- `<main>` padding: `12px 14px calc(70px + env(safe-area-inset-bottom))` — the bottom
  clearance keeps content above the nav pill. Never remove the safe-area term.
- Nav pill sits at `bottom:calc(12px + env(safe-area-inset-bottom))`.
- **Shell embed:** when loaded inside the native iOS shell the URL hash carries
  `shell=1`, which adds `body.shell-embed` and hides `#navPill` — the app's own
  tab bar owns section navigation there. Everything else renders unchanged;
  `.callTabs` (review/challenge pill) stays visible since it has no native twin.
- Z-index ladder: stage overlays `5–6` · error banner `30` · nav pill `40`. New modal
  surfaces (if ever needed) start at `50`. Do not exceed without updating this table.

### 3.3 Phases, not pages

Each screen is a `<section id="p-…">` inside `<main>`, toggled with `.hidden`. The current
phases: `p-load`, `p-record`, `p-frame`, `p-tap`, `p-review`, `p-tap-floor`, `p-clip`,
`p-analyze`, `p-track`, `p-label`, `p-target`, and the roadmap phases `p-live`,
`p-matches`, `p-coach`, `p-stats`, `p-shot-bot`, `p-sharing` (blueprints in §16). To add a screen,
add a section and follow §17 — never add a second header, tab bar, or routing chrome
beyond the §8.3 nav pill.

### 3.4 The proxied-primary pattern

Each phase's primary action button **stays in the section's DOM** as the behavioral source
of truth (click handler + disabled state) but is hidden with `.proxied` and mirrored into
the header pill `#hdrAction`. Follow this pattern for every new phase: one primary action,
proxied to the header. Secondary actions stay inline in the section.

### 3.5 Known platform gotchas (keep these workarounds)

- `#vid` is kept renderable at 2 px / `opacity:0` — `display:none` breaks canvas capture
  on some iOS versions.
- Film strips set `touch-action:none` and implement their own gesture handling; keep
  `cursor:grab` for desktop.
- View Transitions for the root are disabled (`animation:none`) so theme switching is an
  instant swap with only the sun/moon icon animating.

---

## 4. Design tokens

Tokens live in the `:root` block of `index.html` and are the **only** place themed values
may be defined. To use a new color/size, add a token first, then reference it. Current
canonical set:

### 4.1 Color tokens — dark (default)

```css
:root{ color-scheme:dark;
  --bg:#000;            /* page + letterbox background (OLED black) */
  --surface:#1c1c1e;    /* raised containers, secondary buttons (iOS systemGray6-dark) */
  --line:#26262a;       /* hairlines, borders, tertiary fills, pressed state */
  --dim:#98989f;        /* secondary text, inactive controls */
  --text:#fff;          /* primary text */
  --accent-bg:#ffd60a;  /* THE accent (iOS yellow-dark). Always with --accent-text */
  --accent-text:#000;
  --strip-bg:#000;      /* filmstrip + stage wells */
  --tick:rgba(255,255,255,.12);
  --out:#35e0ff;        /* fitted OUT-line edge (calibration) — cyan */
  --tin:#b4ff3a;        /* fitted TIN edge (calibration) — lime */
  --in:#2ecc5e;         /* IN verdict fill  (verdicts ONLY) */
  --outcall:#e03a2f;    /* OUT verdict fill (verdicts ONLY) */
  --mk-racket:#22d3ee;  /* timeline marker: racket hit */
  --mk-floor:#ffb020;   /* timeline marker: floor bounce */
  --mk-side:#c77dff;    /* timeline marker: side wall */
  --mk-unknown:#c7c7cc; /* timeline marker: unclassified */
}
```

### 4.2 Color tokens — light overrides

```css
:root[data-theme="light"]{ color-scheme:light;
  --bg:#f5f5f7; --surface:#e9e9ee; --line:#e0e0e5; --dim:#6e6e76; --text:#000;
  --accent-bg:#ffd60a; --accent-text:#000;      /* accent is theme-invariant */
  --strip-bg:#e3e3e8; --tick:rgba(0,0,0,.14);
  --out:#007da6; --tin:#557a00;                 /* darkened for contrast on light */
}
```

Rules: verdict (`--in`/`--outcall`), marker hues, and the accent are theme-invariant.
Any token that renders on `--bg`/`--surface` needs a light override that keeps ≥ 4.5:1
contrast for text and ≥ 3:1 for UI shapes.

### 4.3 Canvas / overlay palette (JS-drawn, non-token by necessity)

Values drawn on the video canvas and SVG overlays. These are fixed — reuse, don't invent:

| Value | Meaning |
|---|---|
| `#3ddc84` | confirmed / done / detected-ball green |
| `#f5c518` | candidate / warning gold (also floor-wizard "warned" mark) |
| `#ffc828` | floor-bounce dots on court maps |
| `#ff5252` | error / rejected red |
| `#9aa0a6` | neutral gray annotation |
| `#f5f5f5` | playhead white (always with `0 0 0 1px rgba(0,0,0,.45)` hairline) |
| `rgba(0,0,0,.55–.62)` | scrims (analyzing overlay, trim shades) |
| `rgba(255,214,10,.9)` | accent-colored canvas overlay (matches `--accent-bg`) |

### 4.4 Radius scale

| Token (use literally) | Use |
|---|---|
| `999px` | all buttons, pills, chips, segmented containers, progress bars |
| `14px` | verdict box |
| `12px` | error banner, text/number inputs |
| `10px` | progress box |
| `8px` | film strips, cards, court containers, floor diagram |
| `6px` | trim selection frame |
| `4px` | tiny tags (court text), viewport indicators |

### 4.5 Spacing scale

Use only: **2, 4, 6, 8, 10, 12, 14, 18, 24**.
Defaults: section stack gap `10`; page gutter `14`; in-card padding `12px 14px`;
row gap `10`; chip gap `6`.

### 4.6 Type, weight, motion tokens

See §6 (type scale) and §10 (motion table). Weights available: **400, 600, 700** only
(three self-hosted woff2 files, `font-display:swap`, system-stack fallback).

---

## 5. Color system — roles and law

### 5.1 Neutral stack

`--bg` (page) → `--strip-bg` (media wells) → `--surface` (raised) → `--line` (hairline /
pressed) → `--dim` (secondary ink) → `--text` (primary ink). Depth comes from these fills
and 1 px `--line` borders — **not** from drop shadows. The only shadows allowed: the nav
pill's ambient shadow, marker/playhead hairlines+glows, and the court miniature's internal
shadows.

### 5.2 Semantic families (never cross the streams)

| Family | Tokens | Where it may appear |
|---|---|---|
| **Accent / action** | `--accent-bg` + `--accent-text` | primary buttons, active nav/chip/segment states, trim handles, sliders, progress fill, active wizard mark |
| **Verdict** | `--in` (green), `--outcall` (red) | verdict box fills, IN/OUT timeline markers, verdict text. Nothing else is ever green/red. |
| **Calibration edges** | `--out` (cyan), `--tin` (lime) | fitted line overlays on the frame + inline references to them (`#instr b.out/.tin`) |
| **Event markers** | `--mk-racket` cyan, `--mk-floor` amber, `--mk-side` purple, `--mk-unknown` gray | timeline bars + legend dots + label buttons. Single source of truth for both. |
| **Status (canvas)** | §4.3 greens/golds/reds | JS-drawn annotations only |

The distinction between **cyan/lime (where the lines are)** and **green/red (what the call
is)** is intentional and load-bearing. Never "simplify" them into one family.

### 5.3 Contrast requirements

- Body/primary text: ≥ 4.5:1 against its fill (white on `#1c1c1e` ✓, black on yellow ✓).
- `--dim` is for secondary information only — never for values the user must read to act
  (timecode values, stats render in `--text`).
- Anything drawn over video thumbnails needs a dark hairline (`0 0 0 1px rgba(0,0,0,.45)`)
  or text-shadow — video content is unpredictable.
- Large verdict text on `--outcall` red is white; on `--in` green is near-black `#03230c`.

---

## 6. Typography

**Family:** `'Chakra Petch', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
sans-serif`. Self-hosted at `/fonts/chakra-petch-{400,600,700}.woff2`. Never add another
family or weight; never fetch fonts remotely.

**Base:** 16 px / 1.4 line-height on `body`.

### 6.1 Type scale (roles, not free sizes)

| Role | Spec | Used by |
|---|---|---|
| Verdict word | 28 / 700, uppercase, tracking `.04em` | `.verdict strong` |
| Stat numeral | 26 / 700, tabular | `.targetPct` |
| Overlay status | 22 / 700, uppercase, tracking `.14em` | `.analyzePulse` |
| Page title | 21 / 700, centered | `h2` |
| Card title | 18 / 700 | `.targetHead strong` |
| Input value | 17 / 400, tabular | `input[type=number]` |
| Step label | 16 / 700, uppercase, tracking `.05em` | `#stepLabel` |
| Body | 16 / 400 | default |
| Control / instruction | 15 / 600–700 (controls) · 15 / 400 dim (guidance) | buttons, `.instruction`, `#floorPrompt` |
| Meta / status / readout | 14 / 400 dim; values 600 `--text`, tabular | `.status`, `.stat`, `.tlReadout`, `.sliderlabel` |
| Caption / small control | 13 / 600 | `button.small`, `.nudgeLabel`, `.targetMeta strong` |
| Chip / tag | 12 / 600 | correction chips, `.courtText` |
| Micro / debug | 11 / 600, uppercase, tracking `.06em` | `.debugEngineRow`, `.engineSeg` |

### 6.2 Rules

- **Uppercase = interactive or telemetric** (buttons, step labels, overlay status).
  **Sentence case = guidance** (instructions, prompts, statuses). Never uppercase a
  paragraph.
- `font-variant-numeric:tabular-nums` on every number that updates in place.
- Big-number pattern (mood board): huge 700-weight numeral + small dim label *below or
  beside*, never a big label.
- Letter-spacing only per the scale above; never negative tracking.

---

## 7. Layout

- **Stack layout:** each phase section is `display:flex; flex-direction:column; gap:10px`.
  Order: instruction → media/strip(s) → readout → controls → secondary actions.
- **Rows:** `.row{display:flex; gap:10px}` with equal-flex children for paired buttons.
- **Grids:** 2-column only (`.labelGrid`, `.targetMeta` — `grid-template-columns:1fr 1fr;
  gap:10px`). Phone-first; no breakpoint system — the app targets one handheld viewport.
- **Full-bleed stage, gutter content:** the stage/canvas runs edge-to-edge; everything in
  `<main>` respects the 14 px gutter.
- **Reserved heights:** any element whose content appears/disappears reserves its space
  (`.verdict{min-height:78px}`, `.status{min-height:1.2em}`). New dynamic elements must do
  the same — zero cumulative layout shift is the bar.
- **One primary action per phase**, proxied to the header pill (§3.4). If a phase seems to
  need two primaries, split the phase.

---

## 8. Component library

Recipes below are the canonical implementations — copy the pattern, don't fork it.
All buttons inherit: uppercase, `letter-spacing:.05em`, `border-radius:999px`, flex-center
content, `cursor:pointer`, `:disabled{opacity:.4}` (never a different disabled color).

### 8.1 Buttons

| Variant | Spec | Use |
|---|---|---|
| **Primary** (`button.primary`, `label.filebtn`) | yellow fill, black text, 15/700, `min-height:48px`, full-width | the phase's main action (usually proxied) |
| **Secondary** (default `button`) | `--surface` fill, `--text`, 15/600, `min-height:48px`, full-width | alternate actions (`Target zones`) |
| **Small** (`button.small`) | width:auto, `min-height:44px`, 13/600, padding `8px 14px` | inline utilities (`Skip landmark`, `Dismiss`) |
| **Header pill** (`.pill`) | yellow, width:auto, `min-height:40px`, padding `0 18px`, 15/700, `:disabled{opacity:.35}` | header action only |
| **Chip** (`.correctionRow button`) | width:auto, `min-height:34px`, 12/600, `1px --line` border, transparent; `.active` = yellow fill + 700 | dense multi-choice rows (corrections) |
| **Segmented** (`.engineSeg`) | pill container `1px --line`, transparent segments 11/600 dim, dividers `--line`; `.on` = `--line` fill, `--text` | small mode toggles (debug engine) |
| **Stepper** (`.stepper button`) | 44×44 transparent circle, 26/400 glyph (−/+), `:active{background:var(--line)}`; groups divided by `1px --line`; center `.stepUnit` 13 dim label (`1 s`, `1 fr`) | frame/second nudging |
| **Play** (`.playBtn`) | stepper-style circle with 22×22 stroke SVG | transport |
| **Icon-only** (`#hdrBack`, `#themeBtn`, zoom) | 44×44 transparent; stage-floating ones get `text-shadow:0 1px 3px rgba(0,0,0,.9)` instead of a fill | chrome |

Do: give every icon-only button an `aria-label`. Don't: mix variants in one row, invent
in-between sizes, or put two primary-styled buttons on screen at once.

### 8.2 Header

Back chevron (`‹`, 30/400, 44 px) — step label (ellipsizing) — theme toggle (animated
sun/moon SVG, §10) — action pill. Hidden elements use `.hidden`, layout never reflows
around them (label flexes).

### 8.3 Nav pill (liquid glass — the only glass in the app)

```css
#navPill{position:fixed; left:50%; bottom:calc(12px + env(safe-area-inset-bottom));
  transform:translateX(-50%); z-index:40; display:flex; gap:4px; padding:5px;
  border-radius:999px;
  background:color-mix(in srgb, var(--surface) 62%, transparent);
  backdrop-filter:blur(20px) saturate(1.5);
  border:1px solid color-mix(in srgb, var(--text) 14%, transparent);
  box-shadow:0 12px 32px rgba(0,0,0,.4)}
```

Items: icon (17×17) + 13 px label, `min-height:44px`, dim → `.active` yellow capsule.
The pill is the app's **section tab bar** — exactly 3 items: **Play · Matches · Coach**,
the top-level sections (not judge/label modes; Label mode lives in the Play dev row,
§8.15). Visible only on the three section roots (`p-load`, `p-matches`, `p-coach`);
hidden inside flows and sub-pages. Back chevron is hidden on section roots (they are
siblings — the pill switches between them) and shown everywhere else. The glass recipe
above is unchanged.

### 8.4 Inputs

- Number: `--surface` fill, radius 12, 17 px tabular, `min-height:48px`, no border.
- Range: native, `accent-color:var(--accent-bg)`, 44 px hit area, paired `.sliderlabel`
  (14 dim label + 600 tabular value `output`).
- File: hidden input inside `label.filebtn`.

### 8.5 Progress (`.progressbox`)

Bordered box (radius 10) containing `.stat` rows (14 px, dim label / value right) and a
10 px pill track with yellow fill; width transitions `.18s ease`. Indeterminate = 35 %
yellow segment sweeping via `slidebar` 1.1 s ease-in-out infinite. Always prefer real
numbers (frames, fps, ETA) over indeterminate when known.

### 8.6 Error banner (`#errBanner`)

Fixed top (z 30), `--surface` fill, radius 12, 15/700 message + small "Dismiss" button in
`--line`. No red fills for errors (red = OUT verdict); errors are calm surface + bold text.

### 8.7 Verdict box (`.verdict`) — reserved-height pattern

Radius 14, centered, `min-height:78px`, 13 px caption + 28 px `strong` word. Four states:
`.in` (green fill, `#03230c` ink) · `.out` (red fill, white ink) · `.neutral` (surface +
border, for event classifications) · `.blank` (transparent + dashed `--line` border, dim —
placeholder so the box always occupies identical space). Never show/hide the box itself.

### 8.8 Film strips & timelines (the app's signature)

Three-tier pattern, video-editor grammar:

1. **Overview rail** (`.clipOverview`, 44 px): whole-clip thumbnails, radius 8, `1px
   --line` border, `--strip-bg` well; carries a 2 px `--dim` viewport rectangle and a 2 px
   white cursor.
2. **Editor strip** (`.clipEditor` 96 px / `.hitTimeline` 88 px): zoomable filmstrip that
   **slides under a center-fixed playhead** — a 3 px `#f5f5f5` bar with dark hairline; the
   playhead itself never moves. `touch-action:none; cursor:grab`.
3. **Readout** (`.tlReadout`): centered 14 px dim, tabular values in 600 `--text`.

**Trim:** darkened shades `rgba(0,0,0,.62)` outside the selection; 3 px yellow selection
frame (radius 6); 14 px yellow side handles with a dark center grip, `cursor:ew-resize`;
off-view handles drop to `.35` opacity. Nudge rows pair a 78 px label column with stepper
groups.

**Markers** (`.hitBar`): 36 px hit area (18 px `.mini`) around a 6 px (4 px mini) opaque
neon bar, radius 3, colored by `--mk`:

```css
.hitBar::before{width:6px; border-radius:3px; background:var(--mk);
  box-shadow:0 0 0 1px rgba(0,0,0,.45),
    0 0 8px 0 color-mix(in srgb, var(--mk) 60%, transparent)}
.hitBar.selected::before{box-shadow:0 0 0 2px var(--text),
  0 0 12px 1px color-mix(in srgb, var(--mk,#fff) 85%, transparent)}
```

Verdict markers use `--in`/`--outcall`; event markers use the `--mk-*` hues; selection =
2 px white ring + stronger glow. Any new timeline annotation must follow this exact
finish (opaque core + dark hairline + soft same-hue glow) so it sits crisply over
thumbnails next to the white playhead.

### 8.9 Cards (`.targetZones` pattern)

`--surface` fill, `1px --line` border, radius 8, header row (`12px 14px` padding, bottom
hairline, 18/700 title + 13 dim meta right), body content, and an optional 2-column
`.targetMeta` footer (13 dim `strong` labels above 15 px values).

### 8.10 Court visualizations (the sanctioned "literal" art)

- **Front-wall target chart** (`.targetCourt`): perspective miniature of a court —
  plaster wall, wood floor, red `#d83a2e` court lines, white label tags — with an absolute
  positioned zone grid: cells outlined `rgba(216,58,46,.78)`, `rgba(245,197,24,.10)` fill,
  13/900 zone number + 26/900 tabular percentage in dark blue-gray inks.
- **Floor bounce map** (`#floorMapSvg`): flat SVG court, `--dim` lines (stroke .28),
  `#f5c518` bounce dots with dark stroke.
- **Floor wizard diagram** (`#floorDiagram`, 118 px): `--surface` card; marks progress
  through landmarks — dim → `active` yellow (pulsing radius) → `done` `#3ddc84` →
  `warned` `#f5c518`.

These are the only components allowed to use non-token "physical" colors, and their
palette is fixed. Extend zones/dots; don't restyle the court.

### 8.11 Legend dots

10 px circles (`.dot`) using the exact marker/verdict tokens, 9 px right margin, inline
with 15 px labels — the legend and the timeline must always agree because they share
tokens.

### 8.12 Stage overlays

- **Analyzing:** full-stage scrim `rgba(0,0,0,.55)` + "ANALYZING…" 22/700 uppercase
  pulsing `.35→1` opacity, 1.4 s. Pattern for any in-place canvas work.
- **Zoom controls:** top-right borderless white glyph buttons with heavy text-shadow —
  floating chrome over video never gets a fill.

### 8.13 Feature cards (hub pages)

`.featureCard` — the tappable card variant: a `<button>` on the §8.9 card recipe.
`--surface` fill, `1px --line` border, radius 8, full width, `min-height:56px`, padding
`12px 14px`, flex row, gap 10, text-align left. Contents, left → right:

- 24 px line icon (§9 grammar) in `--dim`;
- title 16/700 **normal case** + one-line 13/400 `--dim` description (cards are content,
  not controls — the uppercase rule applies only to the tag);
- right-aligned phase tag: 12/600 uppercase `--dim` (`PHASE 4`, `SOON`).

`:active` = instant `--line` fill (0 ms, §10). The whole card is one target, ≥ 44 pt.

Feature cards live on **hub pages** (currently the Coach hub, `p-coach`). Hubs carry no
guidance copy — the cards are the page. They navigate to sub-pages via `setPhase()` — this is
sanctioned drill-down navigation, **not** nav chrome: the nav pill stays a section tab
bar (§8.3) and §3.3/§18 still hold.

### 8.14 Placeholder pages

`.placeholderHero` — the §13 `.blank` dashed treatment scaled to a page: dashed
`1px --line` border, radius 8, `min-height:180px`, centered column containing the
feature's line icon at 40 px in `--dim`, then `COMING SOON!` — 16/700, uppercase,
tracking `.05em`, `--dim` (the one sanctioned exclamation mark, §14). Drawn inline
(SVG + text); no image files, no network (§0.5).

Below the hero, a §8.9 card titled "Planned" lists capabilities as rows: a leading
chip tag + a 15 px sentence-case label. Chip tags are 12/600 uppercase, radius 999px:
`CORE` = `--line` fill, `--text`; `LATER` = transparent, `1px dashed --line`, `--dim`.
Chips are informational only (not interactive), so they are exempt from the 44 px rule.

Placeholder phases have **no primary action** — the header shows the step label plus a
back chevron on sub-pages (like `p-label`). `p-matches` is a section root: no chevron;
the nav pill switches away from it (§8.3).

### 8.15 Hero action cards (Play screen)

`.heroCard` — the Play screen's primary actions. Radius 8, full width,
`min-height:72px`, padding 14px, flex row, gap 12: 28 px line icon (§9) · column of
title 16/700 **normal case** + 13/400 description. One ≥ 48 px target each (these are
primaries). Two variants:

- **Accent** — `--accent-bg` fill, all ink `--accent-text`: the *single loudest* action
  ("Judge a clip" — it *is* the file input, `label.filebtn` recast around the hidden
  `<input type=file>`). Exactly one accent card per screen, ever.
- **Surface** — `--surface` fill, `1px --line` border, `--text` title, `--dim`
  icon/description: every other card. A working surface card ("Record a clip", opens
  `p-record`) carries no tag; a future one carries the right-aligned `SOON` tag
  12/600 uppercase `--dim` ("Live match"). Tag presence is what separates "works
  today" from "coming" — never a second accent.

Cards sit under a 12/600 uppercase `--dim` "PLAY" heading, order: Judge a clip ·
Record a clip · Live match.

**Dev row:** bottom of Play, after a `1px --line` hairline: `DEV` micro-label (11/600
uppercase, `--dim`) + a row of `button.small` utilities — "Debug targets" and the
"Label mode" toggle (`.active` = accent fill + 700, like correction chips).

### 8.16 Record screen components (`p-record`)

- **Live preview** (`#camVid`): a `<video>` absolutely filling the stage,
  `object-fit:contain` on the `--strip-bg` well, z-index 4 (under the §3.2 overlay
  ladder). Visible only in the record phase; the canvas stays beneath it.
- **REC readout** (`.recRow`, reserved 24 px): 10 px `.recDot` + `#recClock` `m:ss.t`
  14/600 tabular. Idle: dot `--line`, clock `--dim`. Recording: dot **accent yellow**
  with an opacity-only 1.2 s pulse (reduced-motion wrapped), clock `--text`. The dot is
  yellow because **red belongs to OUT verdicts** (§5.2) — never a red record dot.
- **Primary**: Record ↔ Stop, proxied (§3.4). Secondary full-width "Calibrate court"
  (disabled while recording) + a `.status` calibration line ("Not calibrated" /
  "Calibrated · lines + wall corners + floor map").
- **Recordings list**: a §8.9 card ("Recordings" + count/total-size meta). Rows
  (`.recItem`, hairline-separated) are two lines: top line = 15/600 tabular date,
  optional `CAL` tag (12/600 uppercase, dashed `--line` capsule, `--dim` — marks an
  attached on-site calibration), right-aligned 13 dim duration + size; second line =
  chip-style actions **Judge · Save · Delete** (34 px visual, `1px --line` border,
  12/600, like correction chips). Delete arms on first tap — label flips to
  "Confirm" with the accent fill (`.arm`) and disarms after ~2.6 s; no modal.
- Empty state: dim sentence-case `.recEmpty` row ("No recordings yet.").
- Storage is on-device only: OPFS `recordings/` folder (blob + JSON sidecar carrying
  the calibration) with an IndexedDB fallback; no server round-trip.

---

## 9. Iconography

Inline SVG only (no icon fonts, no image files): 24×24 viewBox, `fill:none`,
`stroke:currentColor`, `stroke-width:2`, round caps/joins; solid dots via small filled
circles. Rendered at 17–24 px. Decorative SVGs get `aria-hidden="true"`; the owning
button carries the `aria-label`. Match the existing set (target reticle, tag, sun/moon,
play/pause) in weight and simplicity — HIG/SF-Symbols-like line style, two shapes max.

---

## 10. Motion

| Token | Value | Use |
|---|---|---|
| Press | 0 ms | `:active` fill swap on steppers/buttons — instant |
| Micro | `.18s ease` | progress width, small property changes |
| Gentle | `.35s cubic-bezier(0,0,0,1)` | theme-icon mask slide |
| Expressive | `.5s cubic-bezier(.25,0,.3,1)` (+ overshoot `cubic-bezier(.5,1.25,.75,1.25)` for beams) | theme sun/moon only |
| Ambient | 1.1–1.4 s ease-in-out infinite | `analyzePulse`, `slidebar`, `floorPulse` |

Rules:

- Screens/phases swap **instantly** — no page transitions (speed is a feature; SwingVision
  lag is the anti-goal).
- Animate only: honest progress, ambient "working" pulses, the theme toggle, and press
  feedback. Never animate layout position of content the user is reading.
- Every *new* nonessential animation must be wrapped in
  `@media (prefers-reduced-motion: no-preference)`; keep ambient pulses opacity-based
  (safe) rather than positional.
- Nothing between 500 ms and ambient — if it needs 800 ms, it's a progress state, not an
  animation.

---

## 11. Interaction rules

- **Touch targets:** ≥ 44×44 px effective (48 px for primaries); dense chips (34 px
  visual) must still clear ~44 px including gaps — never go denser than the correction
  chips.
- **Press feedback:** momentary `--line` fill on `:active` (steppers) or the button's
  natural fill darkening under opacity; no resting hover states — hover may enhance but
  never gate anything (touch-first).
- **Direct manipulation first:** scrub strips, drag handles, tap the frame; buttons are
  the fallback for precision (±1 s / ±1 fr steppers mirror every drag interaction).
- **Tap semantics on choices:** tapping the model's own call confirms it; tapping another
  option corrects; tapping the highlighted one undoes (correction chips). Reuse this
  confirm/correct/undo grammar for any human-feedback UI.
- **Gesture ownership:** interactive strips declare `touch-action:none` and must not
  fight page gestures; everything else leaves default touch behavior alone.
- **Disabled ≠ hidden:** keep actions visible-but-disabled (`opacity:.4`) when they'll
  become available in this phase; hide (`.hidden`) only what belongs to another phase.

---

## 12. Data-viz & stats

- Percentages: integer + `%`, 26/700 tabular on zone cells; dim 13 px labels.
- Timecodes: `m:ss.t` or frame counts, tabular, values in `--text` 600 inside dim
  sentences.
- Heat/zone intensity: vary **fill alpha of one hue** (gold on the wall chart), never a
  rainbow ramp.
- Every chart pairs with a text summary (`Most used`, `Untouched zones`) — the chart is
  never the only representation (accessibility + glanceability).
- No gridlines, axes, or legends beyond what decodes the data; the court itself is the
  axis system.

---

## 13. States & feedback

| State | Pattern |
|---|---|
| Empty / first-run | Centered phase (`body.phase-load`), hero action cards (§8.15) — no guidance copy |
| Working (known progress) | `.progressbox` with real stats (frames, fps, ETA) + determinate bar |
| Working (unknown) | Indeterminate bar or stage scrim + pulsing uppercase label |
| Inline status | `.status` line (14 dim), reserved height; `.warn` = 700 `--text` (still no red), `.ok` = `--text` |
| Error | `#errBanner` top banner: bold message + Dismiss. Recoverable, calm, specific |
| Result | Verdict box state change within reserved space |
| Placeholder | Dashed-border `.blank` treatment; full-page placeholders use `.placeholderHero` (§8.14) |

Copy for statuses is specific and actionable ("Tap the two ends of the out line", not
"Error"). Numbers over adjectives ("132/300 frames" over "working…").

---

## 14. Voice & copy

**Referee's voice: calm, terse, factual.**

- Verdicts and telemetry: uppercase single words (IN, OUT, ANALYZING…).
- Buttons: verb-first, ≤ 3 words ("Track ball", "Use this frame", "Judge frame").
- Instructions: one sentence, present tense, name what the user sees ("Load a clip from
  this phone to begin."). Colored keywords (`b.out`, `b.tin`) when referring to fitted
  lines.
- Domain terms exactly: *out line, tin, service line, front/side wall, floor, rally,
  bounce* (never "boundary", "net", etc.).
- No exclamation marks, no praise ("Great!"), no anthropomorphism. The app states facts.
  Single sanctioned exception: the "COMING SOON!" hero text on roadmap placeholder pages
  (§8.14) — nowhere else.

---

## 15. Accessibility checklist (every UI change)

- [ ] Text ≥ 4.5:1, UI shapes ≥ 3:1 against their fill, **in both themes**
- [ ] Meaning never color-only: verdicts/markers always have a text label or legend
- [ ] Icon-only buttons have `aria-label`; decorative SVG `aria-hidden="true"`
- [ ] Targets ≥ 44 px; primary 48 px
- [ ] Tabular numerals on updating numbers; reserved space (no CLS)
- [ ] New animation respects `prefers-reduced-motion`
- [ ] Works one-handed: primary action reachable in header pill; nav pill bottom-center
- [ ] Overlays on video carry hairline/shadow separation

---

## 16. Screen blueprints (current phases)

Each phase: header shows step label + proxied primary; `#instr` gives the one-line hint.

| Phase | Purpose | Body (top→bottom) | Primary (proxied) |
|---|---|---|---|
| `p-load` | Play section root — get a clip | "PLAY" heading · hero cards (§8.15): Judge a clip (accent, file input) / Record a clip (surface, working) / Live match (surface, SOON) · dev row | — (no chevron; section root) |
| `p-record` | Record rallies + on-site calibration | stage = live camera preview · REC readout · Calibrate court + calibration status · Recordings card (§8.16) | "Record" ↔ "Stop" |
| `p-frame` | Pick a clean calibration frame | overview rail · editor strip w/ playhead · readout · transport+steppers | "Use this frame" |
| `p-tap` | Tap out line & tin on frame | stage-driven; clear-selection small button | "Looks right" (disabled until 2 taps) |
| `p-review` | Approve fitted lines (cyan/lime on stage) | minimal; evidence is the stage | "Use these lines" |
| `p-tap-floor` | Floor calibration wizard | `.floorRow`: diagram (progress marks) + prompt/side actions · skip-all / save-profile | "Use floor map" |
| `p-clip` | Trim rally clip | debug engine seg (right-aligned) · overview · trim editor (yellow handles) · transport+readout row · start/end nudge steppers · frame summary | "Track ball" |
| `p-analyze` | Honest processing | `.progressbox` stats + bar (+ stage ANALYZING pulse) | — (auto-advances) |
| `p-track` | Review track, judge calls | scrub hint lives in the header `#instr` line (detection failures replace it, `.warn`) · overview w/ marker minis · hit timeline (neon bars, center playhead) · readout · transport — **Review pane:** frame input + "Target zones" row · verdict box; **Challenge pane:** type dropdown (In/Out folded into the front-wall options) · Bounce / Not-bounce toggle (`.corrSeg`) · panes switched by a floating Review \| Challenge pill (`.callTabs`, same liquid-glass style as `#navPill`, fixed bottom-center) | "Judge frame" |
| `p-label` | Human bounce labeling | overview · label timeline · transport+zoom · 2-col type grid (dot+label) · delete (destructive = plain secondary, disabled until selection) | — |
| `p-target` | Stats: targets & bounces | Front-wall targets card (court chart + meta) · Floor bounces card (SVG map + meta) | — |
| `p-matches` | Matches section root (placeholder) | placeholder hero · Planned card (§8.14) | — (no chevron; section root) |
| `p-coach` | Coach section root (hub) | three feature cards (§8.13) — no hero, no copy | — (no chevron; section root) |
| `p-live` | Placeholder: live match | placeholder hero · Planned card (§8.14) | — (back chevron only) |
| `p-stats` | Placeholder: stats + trends | placeholder hero · Planned card (§8.14) | — (back chevron only) |
| `p-shot-bot` | Placeholder: shot selection | placeholder hero · Planned card (§8.14) | — (back chevron only) |
| `p-sharing` | Placeholder: your coach (coaching platform) | placeholder hero · Planned card (§8.14) | — (back chevron only) |

`p-load` is the **Play section root**: hero action cards (§8.15) + dev row replace the
old load-button stack. Sub-page back routes: `p-record` / `p-live` → Play; `p-stats` /
`p-shot-bot` / `p-sharing` → Coach. The calibration wizard (`p-tap` → … → `p-tap-floor`)
serves two flows: entered from `p-frame` it exits to `p-clip`; entered from `p-record`
("Calibrate court", on a frame frozen from the live camera) it exits back to `p-record`
and the result rides along with each recording. The nav pill is the section tab bar and appears on the three
section roots only (§8.3).

Production surfaces (section roots and placeholder pages) carry **no `#instr` guidance
copy** — the UI leads; onboarding will teach, later. The `#instr` strip remains for the
working tool flows (calibration, clip, call), whose instructions are operational, not
explanatory. The Play hint line (`#loadHint`) is empty in judge mode (reserved height,
no CLS) and speaks only for the dev-row label mode. A healthy backend is silent —
`#loadStatus` only reports problems.

Mode switch Judge ↔ Label lives in the Play dev-row toggle (§8.15). The call page's
Review ↔ Challenge switch is a second instance of the same liquid-glass pill
(`.callTabs` shares `#navPill`'s rules); only one pill is ever on screen at a time.

---

## 17. Extending the system (roadmap: live match mode, auto editor, stats/AI coaching)

1. **New screen** = new phase `<section id="p-…">` + step-label entry + proxied primary.
   Reuse §8 components; a new component requires a new subsection here first.
2. **New color** = new `:root` token with a light override and a §5.2 family assignment.
   PR must state which family it joins. Hex literals outside tokens are allowed only in
   the canvas palette (§4.3) and court miniature (§8.10).
3. **Live-mode surfaces** (future): verdict box and marker grammar scale up — a live call
   is a full-stage verdict-colored flash + uppercase word, not a new visual language.
   Stats/coaching screens are §8.9 cards + §12 rules stacked in `<main>`.
4. **CSS lives in the `<style>` block of `index.html`**, grouped by component with the
   existing terse `/* purpose */` comments; JS constants that mirror tokens (e.g. `CFG`
   hexes) must be updated in lockstep with `:root`.
5. When in doubt, find the closest existing screen in §16 and copy its structure.

---

## 18. Never do

- No new fonts, weights, icon sets, or CDN/remote assets of any kind.
- No second accent; no green/red outside verdicts; no cyan/lime outside calibration.
- No drop shadows on cards/buttons; no glass outside the nav pill; no decorative gradients
  outside the court miniature.
- No spinners where real progress or evidence is possible; no fake progress.
- No layout shift from appearing content; no moving playhead (the strip moves).
- No page-transition animations; nothing animated longer than 500 ms except ambient
  pulses.
- No hover-only affordances; no touch targets under 44 px; no removal of safe-area math.
- No scrolling app shell; no second header/tab-bar/nav chrome.
- No sentence-case buttons, no uppercase paragraphs, no exclamation marks.
- No proportional figures in updating numbers.

---

## 19. Pre-merge design checklist

- [ ] Both themes checked at 390 × 844 (and one small phone, e.g. 375 × 667)
- [ ] §15 accessibility list passes
- [ ] Only tokens / sanctioned palettes used; spacing & radii from scale
- [ ] One primary action, proxied; header/nav chrome untouched
- [ ] Reserved-height rule holds (toggle every dynamic state and watch for shift)
- [ ] Copy follows §14; domain terms correct
- [ ] `/verify` skill run: app drives end-to-end, screenshots captured in both themes
