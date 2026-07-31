# DESIGN.md — CrossCourt

Design system and UI rulebook for the CrossCourt app (`index.html`, single-file
mobile web app). This document is the **single source of truth for all UI and front-end
work**. Any agent or human touching the UI must follow it exactly. When code and this
document disagree, fix the code (or update this document deliberately in the same PR —
never silently drift).

**2026-07-28: the design-lab language was adopted.** The visual system below is the one
prototyped in `design-lab/crosscourt-lab.html` (TennisIQ-derived) and ported into the
app wholesale: soft canvas, white cards on soft shadows, one lime accent, ink-dark hero
cards, and an icon-only dark dock. The pre-port system (OLED black, yellow accent,
Chakra Petch, liquid glass) is history — do not reintroduce it.

The aesthetic in one sentence: **a training companion with a modern sports-app finish** —
soft canvas, floating white cards, one lime accent, ink-dark hero panels, neon functional
markers over film strips, giant tabular numerals, capsule controls, and a floating dark
icon dock.

---

## 0. TL;DR — the 12 binding rules

1. **Two first-class themes.** Default follows the system; light is the reference look
   (`--bg:#F1F2F4` canvas, white cards), dark is its ink twin (`--bg:#0E0F11`). Every
   element must work in both via tokens — never hardcode a themed color.
2. **One accent.** Lime `--accent-bg: #D9F04A` with ink text is the *only* accent.
   Never introduce a second accent; every other color must carry a semantic meaning
   defined in §5.
3. **Color = meaning.** Green/red are reserved for IN/OUT verdicts. Cyan/lime-yellow are
   reserved for fitted calibration edges. Neon marker hues are reserved for event types.
   Never use these decoratively, and never let color be the only carrier of a meaning —
   always pair with a text label or icon.
4. **Capsules and soft cards only.** Controls are pill-shaped (`border-radius:999px`);
   containers use the radius scale in §4.4 (24 px cards) with the §4 shadow tokens.
   No sharp-cornered UI.
5. **System type stack.** `-apple-system / SF Pro Text` with weights 400–800; no webfonts,
   no CDN assets of any kind — the app must work offline on court.
6. **44 pt minimum touch targets** (Apple HIG). `min-height:44px` and ≥44 px effective
   width on every tappable control; primary actions are 46–48 px.
7. **Sentence-case controls, tight tracking.** Buttons and titles are sentence case with
   `letter-spacing:-.01em`; uppercase survives only in micro-labels, chips/tags, and
   telemetric overlays (ANALYZING…, state chips), with positive tracking.
8. **Tabular numerals everywhere numbers move.** `font-variant-numeric:tabular-nums` on
   timecodes, counters, stats, and readouts.
9. **No layout shift.** Reserve space for anything that appears/disappears (see the
   verdict box pattern, §8.7). Showing a result must never push the page around.
10. **Direct manipulation, video-editor grammar.** Timelines are film strips that slide
    under a **center-fixed white playhead**; trimming uses accent handles; markers are
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
  "Looks right?", "Analyze"). The single primary action lives in the header pill.

---

## 2. Aesthetic direction (mood-board translation)

The reference is the design lab (`design-lab/crosscourt-lab.html`, TennisIQ-derived
training-app language). Its motifs, all now in the codebase:

| Lab motif | Our implementation |
|---|---|
| Soft canvas, white cards on soft shadows | `--bg` canvas + `.card`/`.targetZones` with `--shadow-card` |
| One lime accent on ink | `--accent-bg:#D9F04A` capsules, active dock circle, trim handles |
| Ink-dark hero panel with stat tiles + progress ring | `.hero` + `.statgrid`/`.stattile` + `.ringbadge` |
| Giant stat numerals (`68%`, `84`) | Verdict 28 px, stat tiles 22 px, 700 weight, tabular |
| Tinted status chips (good/avg/poor) | `.chip.good/.avg/.poor` on `--tint-*` pairs |
| Video-editor trim UI (accent handles, filmstrip, white playhead) | `.clipEditor`, `.clipHandle`, center-fixed `#clipCursor` |
| Segmented range controls (`1W/1M/3M/6M`) | `.seg` segmented control, `.corrSeg` two-way segment (§8.18, currently uninstantiated) |
| Floating dark icon dock | `#navPill` section dock (§8.3) |
| Session cards with stat tiles + feedback rows | `.clipcard` head rows on the Analysis root; the stat tiles and feedback live on `p-match` behind them (§8.20) |
| Zone/heat charts on a literal court | `.targetCourt` front-wall chart, `#floorMapSvg` bounce map |

Tone target: **modern training companion** (TennisIQ-class sports apps) — warm but not
consumer-cute. No gradients-for-fun (run-thumb gradients on Analysis cards are the
sanctioned exception), no glassmorphism anywhere, no illustration style. The only
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
- **The theme is one setting for the whole app, not per page.** The store is the
  origin's, so a flip has to reach pages that are already loaded: `syncThemeFromStorage`
  re-applies it on the `storage` event (a second browser tab) and on
  `visibilitychange`/`pageshow` (a section webview coming back on screen in the native
  shell, where each of the four tabs is its own webview and the boot script above ran
  once, long ago). Syncing never replays the §10 wipe — that transition is anchored to
  the button that was pressed. Any future cross-page setting needs the same two hooks.
- `body{height:100dvh; overflow:hidden}` — the app is a fixed shell, not a scrolling page.
  Individual panels may scroll internally if they must; the shell never does.
- `body{touch-action:manipulation}` — no double-tap smart zoom, anywhere. WebKit (and the
  native shell's webviews, which honor the viewport meta) otherwise swallows a fast second
  tap on a control — the −/+ steppers especially — as a zoom gesture instead of a second
  click. Panning/scrolling is unaffected, and the stage + strips keep their stricter
  `touch-action:none`. Don't relax this per-element; a native app never double-tap zooms.
- **A content page that can outgrow the viewport must be given an internal
  scroller.** `main` is `flex:0 0 auto; overflow:hidden`, so a tall page does not
  clip-and-scroll — it grows straight past the bottom of a body that never
  scrolls, putting the overflow permanently out of reach. Section pages carry
  `body.phase-page` (`flex:1 1 auto; min-height:0; overflow-y:auto`); reports use
  `body.phase-target` and the Call page `body.phase-track`. Any new stage-less
  phase needs one of these, or its lower content silently becomes unreachable.
- **Every internal scroller pins `overflow-x`.** CSS promotes the other axis from
  `visible` to `auto` whenever one axis is `auto`, so `overflow-y:auto` alone hands
  the page a horizontal scrollbar the moment any child pokes past the 14 px gutter
  (§7). That is how the Dashboard picked up a 14 px side-scroll that dragged the
  whole page — the Coach Notes rail's negative side margin, ported from the lab,
  where no intermediate scroller clips it. The rail's own `overflow-x:auto` is
  untouched: a rail may scroll sideways, a page may not.
- **A page that fixes `main`'s height must also floor the stage.**
  `body.phase-track` sizes `main` from `--track-main-height`, measured from the
  call controls; `preserveTrackStageHeight()` clamps that measurement to
  `innerHeight − (header + #reviewSeg + #instr) − TRACK_MIN_STAGE_PX` (400 px)
  and never below `TRACK_MIN_MAIN_PX` (240 px). Unclamped it is the same trap
  reached from the other side: the taller Call pane measured past the viewport,
  `#stage` collapsed to 0, and the tail of the page went off a body that never
  scrolls. Any new chrome added between the header and `main` joins that
  subtraction, and the clamp re-runs on `resize`.
- `-webkit-tap-highlight-color:transparent` — we own press feedback (§11).

### 3.2 Shell anatomy (do not restructure)

```
<header>        fixed-height top bar: back chevron · step label · theme toggle · action pill
#reviewSeg      match-review pane switcher (review phases only — §8.21)
#instr          one-line contextual instruction strip (dim), directly under the header
#stage          flex-growing black canvas area: video frame, overlays, zoom controls
<main>          the current phase <section> (controls, timelines, cards)
#navPill        floating dark icon dock, bottom-center (section roots only)
```

- Header: `min-height:56px`, padding `calc(8px + env(safe-area-inset-top)) 16px 8px`,
  gap 10. Step label = 17/700, `letter-spacing:-.02em`, ellipsizes. The top
  safe-area term keeps the header below the notch/Dynamic Island when the page
  runs full-bleed in the native shell (viewport-fit=cover); it resolves to 0 in
  a normal browser. `#errBanner` carries the same term on its `top`. Never
  remove either safe-area term.
- `<main>` padding: `12px 14px calc(92px + env(safe-area-inset-bottom))` — the bottom
  clearance keeps content above the nav dock. Never remove the safe-area term.
- Nav dock sits at `bottom:calc(14px + env(safe-area-inset-bottom))`.
- **Shell embed:** when loaded inside the native iOS shell the URL carries
  `?shell=1`, which adds `body.shell-embed` and hides `#navPill` — the app's own
  tab bar owns section navigation there. Because that tab bar owns it, each
  shell webview is **pinned to the section it booted as** (`SHELL_TAB`, from the
  `#tab=` fragment, carried across hash-stripping reloads in per-webview
  `sessionStorage`): `setPhase` rewrites any section-*root* target to this
  webview's own root, so a flow that exits cross-section on the web — the
  analyze-a-clip review exits to Analysis, roadmap pages back out to Dashboard
  or Training — lands back on the tab it started from instead of stranding the
  native tab on another section (the pill that would recover it is hidden
  here). Phases below root are never rewritten, and the `#run=` review sheet
  has no `#tab=` so it is never pinned. Session restore participates:
  `tryRestoreSession`'s "still at rest" guards compare against the pinned root
  (`SHELL_TAB || 'load'`), so the `openRunReview` reload still rehydrates the
  full review on a pinned tab instead of being vetoed by a rest phase that is
  no longer `load`. Restore failures on a pinned non-Dashboard tab report
  through the §8.6 banner — `#loadStatus` sits inside the hidden `p-load`
  section there, and the "load the same video" recovery copy assumes a file
  input those tabs never show. It also hides `.devOnly` (the Dashboard
  Dev row, §8.15): inside the app the page is the product, and the only reader of
  that row is someone sitting at the Mac. Everything else renders unchanged;
  Since the Challenge dock was archived there is no second dock left to reconcile.
  The flag is a **query** parameter, not a hash one: `openRunReview` strips the
  hash before `location.reload()`, so a hash-borne flag was dropped on that
  reload (and on a WKWebView process-restore) and both docks ended up on screen.
  The hash form is still read, for links minted before that move.
- Z-index ladder: stage overlays `5–6` · error banner `30` · nav pill `40`. New modal
  surfaces (if ever needed) start at `50`. Do not exceed without updating this table.

**Native shell (iOS).** The native client substitutes `NavigationStack` and the
system back button for the header chevron and the proxied primary (§3.4), which
are web-shell mechanisms. The phase inventory and the §16 blueprints are shared;
only the chrome that moves between phases differs per client. The native tab bar
carries **the same four items as the web dock, in the same order** — Dashboard ·
Analysis · Training · Progress, webviews onto those section roots with
`?shell=1` — drawn in the platform's own chrome (system tab bar) rather than as a
copy of the dark dock. It is **icon-only**, like the dock: `tabItem` takes a bare
`Image`, because a `Label` there always draws its title on iPhone, and the name
rides along as the image's accessibility label — the native counterpart of the
dock buttons' `aria-label`. The two menus are one menu with two renderings;
changing the item set means changing both. The native record screen
still has no tab (hidden 2026-07-28 until the auto-calibrating capture flow earns
it back). One piece of native-only chrome
floats over the webviews: the server-settings gear, bottom-trailing above the tab
bar, because a fresh install must be able to point the app at a pipeline.

### 3.3 Phases, not pages

Each screen is a `<section id="p-…">` inside `<main>`, toggled with `.hidden`. The current
phases: `p-load`, `p-record`, `p-frame`, `p-tap`, `p-review`, `p-tap-floor`, `p-clip`,
`p-analyze`, `p-track`, `p-player1-report`, `p-player2-report`, `p-label`,
the section roots `p-matches` (Analysis), `p-coach` (Training), `p-progress`, the
match analysis page `p-match`, and the
roadmap placeholder `p-live` (blueprints in §16).
To add a screen, add a section and follow §17 — never add a second header, tab bar, or
routing chrome beyond the §8.3 nav dock.

**One page may span several phases.** `p-track`, `p-player1-report` and
`p-player2-report` are the three panes of the match review page (§16), not three steps:
a `REVIEW_PHASES` constant groups them, the `#reviewSeg` switcher (§8.21) moves between
them, the header carries the match date instead of a step number, and Back exits to
Analysis from any pane. They stay separate `S.phase` values only because ~25 call sites
key off `S.phase === 'track'` — the grouping is a routing fact, not a licence to spread
one screen across phases whenever it is convenient.

### 3.4 The proxied-primary pattern

Each phase's primary action button **stays in the section's DOM** as the behavioral source
of truth (click handler + disabled state) but is hidden with `.proxied` and mirrored into
the header pill `#hdrAction`. Follow this pattern for every new phase: one primary action,
proxied to the header. Secondary actions stay inline in the section.

**The pill must not crowd the title out.** `#hdrAction` is sized by its own text and
`#stepLabel` takes what is left, so a long pill label is paid for by the title's ellipsis —
"Use calibration" beside "Confirm court" truncated it to "Confirm co…" at 390 px, and every
calibration step clipped at 375. A `STEP_META` action may therefore carry a **`short`**: the
pill wears `short`, and `label` becomes its `aria-label`. Nothing is lost — the proxied
original is hidden, so the pill is the only thing a screen reader reads, and it still reads
the full phrase.

Every calibration step means the same thing (accept what is on screen and go on) and the
title beside it already names the object, so all of them wear **`Use`**. A step whose verb
is *not* accept-and-continue keeps its own word: `Record`, `Analyze`, `Judge`. When adding a
phase, check the title still fits at **375 px** — that is the support floor, and the title is
the thing that must survive.

### 3.5 Known platform gotchas (keep these workarounds)

- `#vid` is kept renderable at 2 px / `opacity:0` — `display:none` breaks canvas capture
  on some iOS versions.
- Film strips set `touch-action:none` and implement their own gesture handling; keep
  `cursor:grab` for desktop.
- The root View Transition's *default* cross-fade is disabled (`animation:none` on
  `::view-transition-old/new(root)`); the §10 wipe replaces it with a scripted
  `clip-path` circle on `::view-transition-new(root)`.
- **Size that circle in percentages, never pixels.** `clip-path` resolves against the
  `::view-transition-new(root)` box, and that box is not reliably viewport CSS pixels —
  WebKit sizes the root snapshot in *device* pixels. A px circle therefore landed at
  1/dpr of its coordinates (emitting from mid-screen instead of the button) and grew to
  1/dpr of its radius, so on a 2x display the wipe stalled at ~81 % coverage and the rest
  snapped in one frame. Percentages resolve against whatever that box is, and it is
  always a uniform scale of the viewport. A percentage `<shape-radius>` resolves against
  `hypot(w, h) / sqrt(2)`.

---

## 4. Design tokens

Tokens live in the `:root` block of `index.html` and are the **only** place themed values
may be defined. To use a new color/size, add a token first, then reference it. Current
canonical set:

### 4.1 Color tokens — dark (default)

```css
:root{ color-scheme:dark;
  --bg:#0E0F11;         /* page canvas (ink) */
  --surface:#1B1C1F;    /* cards, raised containers, secondary buttons */
  --line:#26272B;       /* hairlines, borders, tertiary fills */
  --dim:#9A9DA5;        /* secondary text, inactive controls */
  --text:#F4F5F7;       /* primary text */
  --accent-bg:#D9F04A;  /* THE accent (lab lime). Always with --accent-text */
  --accent-text:#141517;
  --accent-soft:#2E321C;/* soft accent fill for icon circles */
  --hero:#141517; --hero-text:#fff;  /* ink hero card — same in both themes */
  --tile:#26272B;       /* stat tiles inside the hero */
  --tint-good-bg:#20301F; --tint-good-fg:#8FD98A;   /* status chips (data, not verdicts) */
  --tint-avg-bg:#332B14;  --tint-avg-fg:#E5B95C;
  --tint-poor-bg:#361D19; --tint-poor-fg:#EF8A76;
  --shadow-card:0 1px 2px rgba(0,0,0,.25), 0 10px 30px rgba(0,0,0,.35);
  --shadow-btn:0 1px 2px rgba(0,0,0,.3), 0 4px 14px rgba(0,0,0,.35);
  --seg-bg:#26272B;     /* segmented-control track */
  --nav-bg:#1B1C1F;     /* dark dock (nav + call tabs) — same in both themes */
  --strip-bg:#000;      /* filmstrip + stage wells */
  --tick:rgba(255,255,255,.12);
  --out:#35e0ff;        /* fitted OUT-line edge (calibration) — cyan */
  --service:#ff9f43;    /* fitted SERVICE-line edge (calibration) — amber */
  --tin:#b4ff3a;        /* fitted TIN edge (calibration) — lime-yellow */
  --in:#2ecc5e;         /* IN verdict fill  (verdicts ONLY) */
  --outcall:#e03a2f;    /* OUT verdict fill (verdicts ONLY) */
  --mk-racket:#22d3ee;  /* timeline marker: racket hit */
  --mk-floor:#ffb020;   /* timeline marker: floor bounce */
  --mk-side:#c77dff;    /* timeline marker: side wall */
  --mk-unknown:#c7c7cc; /* timeline marker: unclassified */
}
```

### 4.2 Color tokens — light overrides (the reference look)

```css
:root[data-theme="light"]{ color-scheme:light;
  --bg:#F1F2F4; --surface:#FFFFFF; --line:#EFF0F3; --dim:#8E9199; --text:#141517;
  --accent-bg:#D9F04A; --accent-text:#141517;   /* accent is theme-invariant */
  --accent-soft:#ECF3C6;
  --hero:#141517; --hero-text:#fff; --tile:#26272B;   /* hero stays ink in light */
  --tint-good-bg:#E7F4E9; --tint-good-fg:#2E7D3E;
  --tint-avg-bg:#FCF3DC;  --tint-avg-fg:#A06E00;
  --tint-poor-bg:#FBE7E4; --tint-poor-fg:#B3402F;
  --shadow-card:0 1px 2px rgba(16,17,20,.03), 0 10px 30px rgba(16,17,20,.05);
  --shadow-btn:0 1px 2px rgba(16,17,20,.04), 0 4px 14px rgba(16,17,20,.06);
  --seg-bg:#E7E8EC; --nav-bg:#1B1C1F;
  --strip-bg:#E3E4E8; --tick:rgba(0,0,0,.14);
  --out:#007da6; --service:#a84f00; --tin:#557a00;  /* darkened for contrast on light */
}
```

Rules: verdict (`--in`/`--outcall`), marker hues, the accent, and the hero/dock inks
(`--hero`, `--tile`, `--nav-bg`) are theme-invariant. Any token that renders on
`--bg`/`--surface` needs a light override that keeps ≥ 4.5:1 contrast for text and
≥ 3:1 for UI shapes.

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
| `rgba(217,240,74,.9)` / `#D9F04A` | accent-colored canvas overlay (matches `--accent-bg`) |

### 4.4 Radius scale

| Token (use literally) | Use |
|---|---|
| `999px` | all buttons, pills, chips, segmented containers, progress bars |
| `50%` | icon circles, dock buttons, play buttons |
| `24px` | page-level cards (`.card`, `.targetZones`, hero, placeholder hero) |
| `18px` | secondary cards (feature cards, delta tiles, rail cards, verdict box, error banner, progress box) |
| `14px` | inner tiles (stat tiles, `.scol`, `.coachMetric`, thumbs), inputs, selects |
| `12px` | wall-corner list chips |
| `8px` | film strips, floor diagram, player crops |
| `6px` | trim selection frame |
| `4px` | tiny tags (court text), viewport indicators |

### 4.5 Spacing scale

Use only: **2, 4, 6, 8, 10, 12, 14, 18, 24**.
Defaults: section stack gap `10`; page gutter `14`; in-card padding `12px 14px`;
row gap `10`; chip gap `6`.

### 4.6 Type, weight, motion tokens

See §6 (type scale) and §10 (motion table). System stack; weights **400, 500, 600, 700,
800** (500 for quiet tile labels, 800 for metric numerals only).

---

## 5. Color system — roles and law

### 5.1 Neutral stack

`--bg` (page) → `--strip-bg` (media wells) → `--surface` (raised) → `--line` (hairline /
pressed) → `--dim` (secondary ink) → `--text` (primary ink). Depth comes from these fills
plus the two shadow tokens: `--shadow-card` on cards and `--shadow-btn` on raised
controls — always the tokens, never ad-hoc shadows. `1px --line` borders mark *inner*
tiles and ghost buttons, not page-level cards. Other sanctioned shadows:
the dock's ambient shadow, marker/playhead hairlines+glows, and the court miniature's
internal shadows.

### 5.2 Semantic families (never cross the streams)

| Family | Tokens | Where it may appear |
|---|---|---|
| **Accent / action** | `--accent-bg` + `--accent-text` | primary buttons, active nav/chip/segment states, trim handles, sliders, progress fill, active wizard mark |
| **Verdict** | `--in` (green), `--outcall` (red) | verdict box fills, IN/OUT timeline markers, and verdict text — `OUT` and the tin fault `DOWN` share `--outcall`, since both end the rally the same way. Nothing else is ever green/red. |
| **Calibration edges** | `--out` (cyan), `--service` (amber), `--tin` (lime) | fitted line overlays on the frame + inline references to them (`#instr b.out/.service/.tin`) |
| **Event markers** | `--mk-racket` cyan, `--mk-floor` amber, `--mk-side` purple, `--mk-unknown` gray | timeline bars + legend dots + label buttons. Single source of truth for both. `--mk-unknown` also renders an unclassified or no-call event — not a verdict, so it stays in this family, not the Verdict one. |
| **Status (canvas)** | §4.3 greens/golds/reds | JS-drawn annotations only |
| **Quality tints** | `--tint-good/avg/poor` bg+fg pairs | `.chip` status chips and trend deltas — *data quality*, not verdicts. Deliberately soft pastel pairs so they can never be mistaken for the loud verdict fills. |

The distinction between **calibration hues (where the lines are)** and **green/red (what
the call is)** is intentional and load-bearing. Never "simplify" them into one family.
`--service` amber and `--mk-floor` amber are close in hue but never share a surface — one
lives on the calibration frame, the other on the timeline; keep it that way.

Won/lost, correct/incorrect and other **data** splits are not verdicts. Separate them with
the solid-vs-dashed border grammar (§8.14, §13) plus an explicit label — never by
borrowing `--in`/`--outcall` (§8.17 is the worked example).

**How sure we are is also a data split.** Attribution provenance (§8.22 — whether a
rally's shot labels were observed, repaired, assumed, or contradicted) is data quality,
not a call and not a calibration edge: solid-vs-dashed carries *not confirmed*, and the
quality tints carry the one contradicted state. It may never borrow `--in`/`--outcall`
**or** the calibration hues (`--out`/`--service`/`--tin`). A draft flagged the conflict
state with `--out` cyan, on the one page that also draws real calls and real fitted
edges — where a borrowed hue stops meaning "we are unsure" and starts asserting
something about the ball.

### 5.3 Contrast requirements

- Body/primary text: ≥ 4.5:1 against its fill (`--text` on `--surface` ✓, ink on lime ✓).
- `--dim` is for secondary information only — never for values the user must read to act
  (timecode values, stats render in `--text`).
- Anything drawn over video thumbnails needs a dark hairline (`0 0 0 1px rgba(0,0,0,.45)`)
  or text-shadow — video content is unpredictable.
- Large verdict text on `--outcall` red is white; on `--in` green is near-black `#03230c`.

---

## 6. Typography

**Family:** `-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto,
sans-serif` with `-webkit-font-smoothing:antialiased`. No webfonts — never fetch fonts
remotely.

**Base:** 17 px / 1.4 line-height on `body`.

### 6.1 Type scale (roles, not free sizes)

Raised one notch on 2026-07-30 — the previous ladder bottomed out at 11–12 px, which
is where most of the app's copy actually lived, and it read cramped on a phone held at
arm's length on court. Sizes below are the *whole* ladder; move it as a ladder, never a
rule at a time.

| Role | Spec | Used by |
|---|---|---|
| Verdict word | 32 / 700, uppercase, tracking `.04em` | `.verdict strong` |
| Stat numeral | 28 / 700, tabular | `.trend .val` |
| Tile numeral | 22–24 / 700, tabular | `.stattile .sv`, `.scol .sv`, `.coachMetric span` |
| Overlay status | 24 / 700, uppercase, tracking `.14em` | `.analyzePulse` |
| View title | 22 / 700, tracking `-.02em` | `h2` — base rule, **no current user**: screen titles live in `#stepLabel` (§16) |
| Step label | 18 / 700, tracking `-.02em` | `#stepLabel` |
| Input value | 18 / 400, tabular | `input[type=number]` |
| Body | 17 / 400 | default |
| Card title | 16 / 600, tracking `-.01em` | `.cardtitle`, `.targetHead strong`, `.hubHead` |
| Instruction / guidance | 16 / 400 dim | `.instruction`, `#floorPrompt` |
| Control | 15 / 600, tracking `-.01em` | buttons, `.pill`, row names |
| Meta / status / readout | 15 / 400 dim; values 600 `--text`, tabular | `.status`, `.stat`, `.tlReadout`, `.sliderlabel` |
| Caption / small control | 14 / 600 | `button.small`, `.nudgeLabel`, `.fbrow .t` |
| Sub-meta | 13 / 400–600 dim | `.metaline`, descriptions, `.targetHead span` |
| Chip / tag | 11–13 / 600–700, uppercase, tracking `.04em` | `.chip`, `.statechip`, `.fcTag` |
| Micro / debug | 12 / 600, uppercase, tracking `.06em` | `.devHead`, tile labels (500, no caps) |

**Exempt: lettering inside the court diagram** (`.targetPct` 26, `.targetZoneNum` 13,
`.courtText` 12 — §8.10). Those sit in a locked-aspect box and are sized to the
diagram, not to this ladder; scaling them with it clips the percentages and slides the
line chips over the zone numbers. When the ladder moves, they stay.

### 6.2 Rules

- **Sentence case everywhere words act** (buttons, titles, controls) with tight negative
  tracking (`-.01em`/`-.02em`). **Uppercase survives only in micro-labels, chips/tags,
  and telemetric overlays** (ANALYZING…, IN/OUT verdict words, state chips), always with
  positive tracking. Never uppercase a paragraph.
- `font-variant-numeric:tabular-nums` on every number that updates in place.
- Big-number pattern (lab): huge 700–800-weight numeral + small dim label *above or
  beside*, never a big label.
- Letter-spacing only per the scale above.

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
  (`.verdict{min-height:86px}`, `.status{min-height:1.2em}`). New dynamic elements must do
  the same — zero cumulative layout shift is the bar.
- **One primary action per phase**, proxied to the header pill (§3.4). If a phase seems to
  need two primaries, split the phase.

---

## 8. Component library

Recipes below are the canonical implementations — copy the pattern, don't fork it.
All buttons inherit: sentence case, `letter-spacing:-.01em`, `border-radius:999px`,
flex-center content, `cursor:pointer`, `:disabled{opacity:.4}` (never a different
disabled color), `:active{filter:brightness(.97)}`.

### 8.1 Buttons

| Variant | Spec | Use |
|---|---|---|
| **Primary** (`button.primary`, `label.filebtn`) | lime fill, ink text, 14/600, `min-height:46px`, full-width, no shadow | the phase's main action (usually proxied) |
| **Secondary** (default `button`) | `--surface` fill + `--shadow-btn`, `--text`, 14/600, `min-height:46px`, full-width | alternate actions |
| **Small / ghost** (`button.small`, `button.ghostbtn`) | width:auto, transparent, `1px --line` border, no shadow, 12–13/600 | inline utilities (`Skip landmark`, `Dismiss`, `Watch source video`) |
| **Header pill** (`.pill`) | lime, width:auto, `min-height:40px`, padding `0 18px`, 14/600, `:disabled{opacity:.35}` | header action only |
| **Start chip** (`button.startchip`) | lime, `min-height:32px`, padding `6px 16px`, 12/600, right-aligned in list rows | row-level actions in `.lrow` lists |
| **Chip** (`.correctionRow button`) | width:auto, `min-height:34px`, 12/600, `1px --line` border, transparent; `.active` = lime fill + 700 | dense multi-choice rows — uninstantiated since the Challenge archive (2026-07-29, `archive/challenge-ui/`) |
| **Segmented** (`.seg`) | one `--seg-bg` capsule track, equal-flex 34 px pill buttons, active = `--surface` fill + small shadow | range/mode pickers (1W/1M/3M/6M); the review pane switcher (§8.21) |
| **Two-way segment** (`.corrSeg`) | two separate equal-width pills, 6 px gap, no shared container; unselected `1px --line` transparent, selected `--accent-bg`/`--accent-text` 700 — full rules in §8.18 | binary choices — uninstantiated since the Challenge archive (§8.18) |
| **Stepper** (`.stepper button`) | 44×44 transparent circle, 26/400 glyph (−/+), `:active{background:var(--line)}`; groups divided by `1px --line`; center `.stepUnit` 13 dim label (`1 s`, `1 fr`) | frame/second nudging |
| **Play** (`.playBtn`) | stepper-style circle with 22×22 stroke SVG | transport |
| **Icon-only** (`#hdrBack`, `#themeBtn`, zoom) | 44×44 transparent; stage-floating ones get `text-shadow:0 1px 3px rgba(0,0,0,.9)` instead of a fill | chrome |

Do: give every icon-only button an `aria-label`. Don't: mix variants in one row, invent
in-between sizes, or put two primary-styled buttons on screen at once.

### 8.2 Header

Back chevron (`‹`, 30/400, 44 px) — step label (ellipsizing) — theme toggle (animated
sun/moon SVG, §10) — action pill. Hidden elements use `.hidden`, layout never reflows
around them (label flexes).

The back chevron steps up one phase (hidden on section roots, which are siblings) — a
header affordance on the single shell header, not new nav chrome (§18): the §8.3 nav
pill remains the only section router. There is no home shortcut; leaving a flow means
stepping back through it.

### 8.3 Nav dock (`#navPill` — dark, icon-only)

```css
#navPill{position:fixed; left:50%; bottom:calc(14px + env(safe-area-inset-bottom));
  transform:translateX(-50%); z-index:40; display:flex; gap:2px; padding:7px;
  border-radius:999px; background:var(--nav-bg);
  box-shadow:0 18px 36px rgba(16,17,20,.28)}
#navPill button{width:46px; height:46px; border-radius:50%;
  color:rgba(255,255,255,.55)}
#navPill button.active{background:var(--accent-bg); color:var(--accent-text)}
```

The dock is ink-dark in **both** themes (`--nav-bg`), icon-only 46 px circles (20×20
icons, labels present in markup for a11y but visually hidden — every button carries an
`aria-label`), active = lime circle. It is the app's **section tab bar** — exactly 4
items: **Dashboard · Analysis · Training · Progress** (not judge/label modes; Label mode
lives in the Dashboard dev row, §8.15). Visible only on the four section roots
(`p-load`, `p-matches`, `p-coach`, `p-progress`); hidden inside flows and sub-pages.
Back chevron is hidden on section roots (they are siblings — the dock switches between
them) and shown everywhere else. No glass anywhere — the dock is opaque. **This is the
app's only dock:** the call page's `.callTabs` (Review/Challenge) shared this shell
until it was archived 2026-07-29 (`archive/challenge-ui/`).

### 8.4 Inputs

- Number: `--surface` fill, radius 12, 17 px tabular, `min-height:48px`, no border.
- Range: native, `accent-color:var(--accent-bg)`, 44 px hit area, paired `.sliderlabel`
  (14 dim label + 600 tabular value `output`).
- File: hidden input inside `label.filebtn`.
- Text: `color-mix(in srgb, var(--surface) 86%, var(--bg))` fill, radius 12 (§4.4), 17 px,
  `min-height:48px`, `1px --line` border — the border is what distinguishes it from the
  Number variant, needed because a text field sits inside a `--surface` card (post-hoc
  player naming, §4.6) rather than always on `--bg` the way Number fields do. The floor
  step's profile-name row reuses this recipe on `--bg` (one text-field look everywhere);
  native dialogs (`window.prompt`/`confirm`/`alert`) are never an alternative — the iOS
  shell's WKWebView has no UI delegate, so they silently no-op there.

### 8.5 Progress (`.progressbox`)

Surface card (radius 18, `--shadow-card`) containing `.stat` rows (14 px, dim label /
value right) and a 10 px pill track with accent fill; width transitions `.18s ease`.
Indeterminate = 35 % accent segment sweeping via `slidebar` 1.1 s ease-in-out infinite.
Always prefer real
numbers (frames, fps, ETA) over indeterminate when known.

### 8.6 Error banner (`#errBanner`)

Fixed top (z 30), `--surface` fill, radius 18, `--shadow-card`, 15/600 message + small
"Dismiss" button in `--seg-bg`. No red fills for errors (red = OUT verdict); errors are
calm surface + bold text.

### 8.7 Verdict box (`.verdict`) — reserved-height pattern

Radius 18, centered, `min-height:86px`, 14 px caption + 32 px `strong` word. Four states:
`.in` (green fill, `#03230c` ink) · `.out` (red fill, white ink) · `.neutral` (surface +
`--shadow-card`, for event classifications) · `.blank` (transparent + dashed dim border —
placeholder so the box always occupies identical space). Never show/hide the box itself.

Above the verdict word sits `.verdictPlayer` — 13/800 uppercase, tracking `.04em`,
inheriting the state's ink: *who hit it* ("Hit by Player 1", or "Player attribution
unavailable"). It renders in **every** state, including the "Judging…" and "No call"
placeholders, so the box never changes height as attribution resolves. A rally's first
front-wall contact is judged against the service line, so its `strong` word is prefixed
`SERVE ` (`SERVE IN` / `SERVE OUT`) — a prefix, never a fifth state or a new color.

### 8.8 Film strips & timelines (the app's signature)

Three-tier pattern, video-editor grammar:

1. **Overview rail** (`.clipOverview`, 44 px): whole-clip thumbnails, radius 8, `1px
   --line` border, `--strip-bg` well; carries a 2 px `--dim` viewport rectangle and a 2 px
   white cursor.
2. **Editor strip** (`.clipEditor` 96 px / `.hitTimeline` 88 px): zoomable filmstrip that
   **slides under a center-fixed playhead** — a 3 px `#f5f5f5` bar with dark hairline; the
   playhead itself never moves. `touch-action:none; cursor:grab`.
3. **Readout** (`.tlReadout`): centered 14 px dim, tabular values in 600 `--text`.

**Trim:** darkened shades `rgba(0,0,0,.62)` outside the selection; 3 px accent selection
frame (radius 6); 14 px accent side handles with a dark center grip, `cursor:ew-resize`;
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

### 8.9 Cards (`.card` / `.targetZones` pattern)

`--surface` fill, **no border**, radius 24, `--shadow-card`. Two shapes:

- **Plain card** (`.card`): 16 px padding, `.cardtitle` (15/600) heading inline with the
  content. Used by the lab surfaces (weekly report, trend cards, best-mark row).
- **Headed card** (`.targetZones`): header row (`14px 16px` padding, bottom `--line`
  hairline, 15/600 title + 12 dim meta right), body content, and an optional 2-column
  `.targetMeta` footer (13 dim `strong` labels above 15 px values).

Inner tiles inside either shape (`.scol`, `.coachMetric`, `.mvtile`-style) are
`1px --line` bordered, radius 14, transparent — depth never stacks two shadows.

### 8.10 Court visualizations (the sanctioned "literal" art)

- **Front-wall target chart** (`.targetCourt`): perspective miniature of a court —
  plaster wall, wood floor, red `#d83a2e` court lines, white label tags — with an absolute
  positioned zone grid: cells outlined `rgba(216,58,46,.78)`, `rgba(245,197,24,.10)` fill,
  13/900 zone number + 26/900 tabular percentage in dark blue-gray inks.
- **Floor bounce map** (`#floorMapSvg`): flat SVG court, `--dim` lines (stroke .28),
  `#f5c518` bounce dots with dark stroke.
- **Floor wizard diagram** (`#floorDiagram`, 118 px): `--surface` card; marks progress
  through landmarks — dim → `active` accent (pulsing radius) → `done` `#3ddc84` →
  `warned` `#f5c518`.
- **Mini court** (`.mini-court`, post-rally replay): the two-camera trajectory over a flat
  court outline — a plan view (from above: court width × depth) and a side elevation
  (depth × height, showing the out line and tin heights) — sharing this family's stroke
  weight and line treatment rather than inventing a new rendering style. **The two panels
  are stacked vertically in one `.mini-court` card** (§8.9 fill; plan view above, side
  elevation below), always both visible together — never toggled behind a control and
  never placed side by side. Toggling would hide evidence behind a tap, against §1's
  "trust through evidence, not decoration"; side by side would squeeze each panel into
  roughly half the phone's width, too narrow to read either court clearly under §7's
  phone-first, no-breakpoint constraint. The trajectory itself is a `--dim` polyline; the
  impact point is a single marker colored by the verdict — `--in` / `--outcall` /
  `--mk-unknown` gray for a no-call — the same three colors the verdict box (§8.7) uses,
  so the replay always agrees with the call that was made.
  Purpose: let a player see the shot that was just called, not just hear the word. No
  touch target: display-only, like the floor bounce map above.

These are the only components allowed to use non-token "physical" colors, and their
palette is fixed. `.mini-court`'s trajectory/marker are token-only by design — the marker
*is* a verdict, so §0.3 governs it, not the literal-art allowance. Extend zones/dots;
don't restyle the court.

### 8.11 Legend dots

10 px circles (`.dot`) using the exact marker/verdict tokens, 9 px right margin, inline
with 15 px labels — the legend and the timeline must always agree because they share
tokens.

### 8.12 Stage overlays

- **Analyzing:** full-stage scrim `rgba(0,0,0,.55)` + "ANALYZING…" 22/700 uppercase
  pulsing `.35→1` opacity, 1.4 s. Pattern for any in-place canvas work.
- **Zoom controls:** top-right borderless white glyph buttons with heavy text-shadow —
  floating chrome over video never gets a fill. **They move to bottom-center
  (`body.zoom-low`) while tapping the out line, tin, service line or wall corners.**
  Floating chrome sits *above* the canvas and swallows taps meant for it, and the
  out line meets the right side wall directly under the default position — so any
  overlay placed over the frame must be checked against the tap targets of every
  phase that shows it. Floor landmarks all sit low, so those phases keep top-right.

### 8.13 Feature cards (hub pages)

`.featureCard` — the tappable card variant: a `<button>` on the §8.9 card recipe.
`--surface` fill, `--shadow-card`, radius 18, full width, `min-height:56px`, padding
`12px 14px`, flex row, gap 12, text-align left. Contents, left → right:

- line icon (§9 grammar) inside a 36 px `--bg` icon circle;
- title 14/600 + one-line 12/400 `--dim` description;
- right-aligned readiness tag (`.fcTag`): 11/600 uppercase chip (`--bg` fill, radius
  999). It carries a **readiness word the reader can act on — `Live` or `Soon`** — and
  **never an internal phase number**. "Phase 6" named a row in our plan, not anything a
  player can do with the card; the two words say the only thing the tag is for, which is
  whether tapping it does something today.

`:active` = instant `brightness(.97)` (0 ms, §10). The whole card is one target, ≥ 44 pt.

Feature cards live on **hub pages** (currently the Training hub, `p-coach`). Hubs carry no
guidance copy — the cards are the page. They navigate to sub-pages via `setPhase()` — this is
sanctioned drill-down navigation, **not** nav chrome: the nav pill stays a section tab
bar (§8.3) and §3.3/§18 still hold.

### 8.13a Coaching advice (`p-coach-advice`)

The leaf under the Training hub, and the one page in the app whose content is advice
rather than measurement. Top → bottom:

- **Provenance card** (§8.9 card): `.cardtitle` headline counting what there is to work
  on, then two 13 px `--dim` `.adviceMeta` lines — the shot and session totals, then the
  pooling statement. Advice pools only the player the user identified on each Analysis
  card; Player A in one match and Player B in another can therefore describe the same
  person without mixing the opponent's shots into Training.
- **Drill cards** (`.adviceDrills > .drillCard`) — the §8.9 card recipe (surface fill,
  `--shadow-card`, radius 18, padding 14) rather than the hairline-bordered variant used
  inside the per-run report, because here the cards *are* the page. Ollama reviews the
  user's identified matches oldest to newest and returns a summary, trend observations,
  two concrete drills with dosage and success measures, and one next-match focus.
  No deterministic coaching copy is substituted when Ollama is unavailable; the
  provenance card states the failure and confirms that measured statistics still exist.

### 8.14 Placeholder pages

`.placeholderHero` — the §13 `.blank` dashed treatment scaled to a page: dashed
`1px` `--dim`-mixed border, radius 24, `min-height:180px`, centered column containing
the feature's line icon at 40 px in `--dim`, then `Coming soon!` — 15/600, sentence
case, tracking `-.01em`, `--dim` (the one sanctioned exclamation mark, §14). Drawn
inline (SVG + text); no image files, no network (§0.5).

Below the hero, a §8.9 card titled "Planned" lists capabilities as rows: a leading
chip tag + a 15 px sentence-case label. Chip tags are 11/600 uppercase, radius 999px,
padding `4px 10px`: `CORE` = `--accent-soft` fill, `--text`; `LATER` = transparent,
`1px dashed` `--dim`-mixed border, `--dim`. Chips are informational only (not
interactive), so they are exempt from the 44 px rule.

Placeholder phases have **no primary action** — the header shows the step label plus a
back chevron on sub-pages (like `p-label`). `p-matches` is a section root: no chevron;
the nav pill switches away from it (§8.3).

### 8.15 Dashboard (`p-load` — the lab view-dashboard)

The Dashboard is the section root the app boots into. Top → bottom:

- **Ink hero** (`.hero`): `--hero` fill (ink in both themes), radius 24, white ink.
  `.herotop` row = 38 px translucent-white icon circle (accent-colored icon) · title
  15/600 + 12 dim-white subtitle · right-aligned `.ringbadge` progress ring (40 px,
  lime arc on `rgba(255,255,255,.14)` track, centered 11/700 tabular value —
  sessions-this-week / 5). `.statgrid` = two `.stattile`s (`--tile` fill, radius 14):
  11/500 dim-white label over 22/700 tabular value with a small unit suffix
  (Sessions · Unforced errors — the share of the last match's decided rallies lost to
  an unforced error; the third slot is deliberately unassigned until the next useful
  headline metric is decided, per the 2026-07-29 metric review). `.heronote` = 12 px
  dim-white sentence — the guidance copy shown before any run exists; hidden as
  soon as one does.
- **Action row** (`.btnrow`): the accent `label.filebtn` "Analyze a clip" (it *is* the
  file input, recast around the hidden `<input type=file>`). Exactly one accent action
  per screen, ever. The archived Record / Live cards (`.heroCard`, hidden) keep the old
  card recipe so un-hiding them is still one attribute.
- **Coach Notes rail** (`.airow`, hidden until live data exists): horizontally scrolling
  212 px `.aicard`s (radius 18, `--shadow-card`) — 28 px `--accent-soft` icon circle +
  13/600 title + 12 dim sentence, filled from the latest run's coach feedback.
- **Dev row:** after a `1px --line` hairline: `DEV` micro-label (11/600 uppercase,
  `--dim`) + `button.small` utilities — the "Label mode" toggle (`.active` = accent
  fill + 700, like correction chips). Browser-only: the hairline, label, and row
  all carry `.devOnly`, which `body.shell-embed` hides (§3.2). Anything added here
  takes that class too.

The hero's `#heroNote` is the **empty state and nothing else**. Coach Notes owns
coach prose on this page; both once rendered the same sentences from the same
feedback string, stacked.

All live regions render from `/api/runs` + `/api/runs/<id>/coach` (§8.20) and keep
their markup defaults as the empty state — no fake sample data, ever.

### 8.16 Record screen components (`p-record`)

- **Live preview** (`#camVid`): a `<video>` absolutely filling the stage,
  `object-fit:contain` on the `--strip-bg` well, z-index 4 (under the §3.2 overlay
  ladder). Visible only in the record phase; the canvas stays beneath it.
- **REC readout** (`.recRow`, reserved 24 px): 10 px `.recDot` + `#recClock` `m:ss.t`
  14/600 tabular. Idle: dot `--line`, clock `--dim`. Recording: dot **accent lime**
  with an opacity-only 1.2 s pulse (reduced-motion wrapped), clock `--text`. The dot is
  lime because **red belongs to OUT verdicts** (§5.2) — never a red record dot.
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

### 8.17 Coaching report panel (`.coachPlayerPanel`)

The per-player report body on `p-player1-report` / `p-player2-report`. A §8.9 card
("Player N report" + a right-aligned 13 dim source tag — `Local feedback` / `Ollama
feedback` / `LLM feedback` / `Unavailable`) wrapping the §8.22 provenance line and four
stacked regions. The provenance line comes **first**, directly under the title: how much
of this report rests on guessed attribution is stated before the numbers that rest on it,
never appended after them as a caveat.

1. `.coachIntro` — 14/400 `--dim` sentence naming what the report is built from.
2. `.coachMetrics` — the **metric tile grid**: 2 columns (1 below 560 px), gap 10,
   padding `12px 14px`. Each `.coachMetric` is the §8.9 card recipe shrunk to a tile —
   radius 8, `1px --line`, `color-mix(--surface 86%, --bg)` fill, padding `10px 12px` —
   holding a 13/400 `--dim` label, a 22/800 value, and a 12/400 `--dim` caption. Absent
   values render as `—`, never a blank tile or a hidden row (reserved height, §0).
   The tile set (2026-07-29 metric review): Shots analyzed · Unforced errors (caption
   carries the out/tin split when known) · Unforced error rate · Average rally
   duration · Main targets · the five zone-usage rates. Shot height and pace tiles were
   removed — those values persist in the backend and the LLM prompt only.
3. `.coachOutcomeComparison` — the **won/lost split**: two `.coachOutcomeSection` peers
   side by side (stacking below 560 px), each a nested card (radius 8, `1px --line`,
   `color-mix(--surface 92%, --bg)`) with a `.coachOutcomeHead` (17/700 title left,
   13 dim rally/shot counts right) over a single-column `.coachMetrics`. **Winning** is
   the solid-bordered filled section; **losing** is `border-style:dashed` on a
   transparent fill — the §8.14 chip grammar reused. This split is data, not a call, so
   it never uses `--in`/`--outcall` (§5.2). Either side with no decided rallies shows
   `.coachOutcomeEmpty` (13 dim sentence) in place of the tiles, so both sections are
   always present.
4. `.coachFeedback` — 16/400 prose, `white-space:pre-wrap`, above a hairline.

Below the report card sits the **per-player movement panel** (`#movementPanelPn`,
2026-07-29 review): a `.targetZones` card ("Player N movement" + right-aligned
coverage tag "n% of rally time observed") holding a `.coachMetrics` grid (Distance
covered · On the T · Front court · Back court · Average speed · Peak speed, the last
two in ft/s from players_v2) over the **court-position heatmap** — the §8.10 flat SVG
court with the backend's 7×8 grid drawn as `--accent-bg` cells whose opacity scales to
the busiest cell (≤ .85 so the wireframe stays legible), and a `.metaline` naming the
detector. Track A/B maps to player 1/2 through the same attribution anchor as the
naming card (§16); the panel hides entirely when players_v2 is absent or the player's
track has no observed coverage — absence over zeros, per Principle 3.

The local template renders immediately; the LLM narration is fetched afterwards and
**replaces the text in place** — the panel never shows a spinner and never changes
height on arrival (§18). If Ollama is unreachable, truncated, or returns invalid
structured output, the local text stays, the source tag reports the exact state, and a
compact `Retry Ollama` button appears beside it. The Ollama prompt receives only the
coaching-relevant metrics (not the repeated full nine-zone tables), and its schema
bounds every narrative field so a local model cannot spend the output budget on a
runaway summary before producing the player sections.

### 8.18 Two-way segment (`.corrSeg`)

Two independent pill buttons, not one shared capsule; exactly one selected.
**Currently uninstantiated on the web:** its only instance was the Challenge
pane's Bounce / Not-bounce toggle, archived 2026-07-29 with that pane
(`archive/challenge-ui/`) — the CSS left `index.html` with it. The grammar
below stays normative for the next instance, web or native.

- Two equal-width pills (`border-radius:999px`, the standard button radius),
  6 px gap between them. No outer container — no shared fill, border, or
  radius wrapping the pair.
- Unselected: 1 px `--line` border, transparent background, inherited
  `--text` label color, weight 600.
- Selected: `--accent-bg` fill with `--accent-text`, border-color
  transparent, weight 700. Never green/red — selection is not a verdict
  (§0.3).
- Labels uppercase with `letter-spacing:.05em` (§0.7).
- Both halves are always present and always the same size, so selection
  never reflows anything (§0.9).
- No third state. A segment that needs one is a different component.
- A segment may start with neither pill selected when no option has a safe
  default, reverting to exactly one selected the instant the user picks. Not
  the third state above: that rule is about a segment needing a third *visual
  mode*, not about a value nobody has chosen yet.

**§0.6 debt (web).** The shipping control is `min-height:38px` — below the
44 px binding minimum. This is a pre-existing violation, not a sanctioned
exception: it ships below the line today and is owed a fix, tracked the same
as any other §0.6 gap would be. Do not read the 38 px figure as permission;
new web work reusing `.corrSeg` inherits the debt until it's fixed, not a
license to repeat it.

**Native.** Any SwiftUI instance of this control is built to the same grammar
above — two equal-width pills, one selected, no third state — but at the full
44 px minimum, so it meets §0.6 outright. New native work is not inheriting the
web control's shortfall.

### 8.19 Auto-detect confirm screen (`#p-confirm`)

The single review surface for an automatically detected court. Overlays the
fitted calibration edges (their existing reserved hues — out `#35e0ff`,
service `#ff9f43`, tin `#b4ff3a`) and the floor wireframe on the frame, with
draggable anchor pucks at the four wall corners and the two short-line ends.

- Anchor pucks use the floor-wizard residual palette (§8.10): `#3ddc84` when
  the detection is trusted, `#f5c518` otherwise. **Never red** — §0.3 reserves
  red for OUT verdicts. Colour is never the only carrier: `#confirmWarn` names
  every reason in words, so the screen reads correctly in greyscale.
- Every draggable puck carries a stable black numeral: **1** out line at left
  wall, **2** out line at right wall, **3** front-wall/floor seam at the right
  corner, **4** the same seam at the left corner, **5** short line at left wall,
  and **6** short line at right wall. Numbers belong to the court landmark
  rather than the current array position, so a missing derived anchor never
  renumbers the remaining pucks. The visible puck is 20 CSS px in diameter for
  numeral legibility; its effective drag target remains the independent 44 CSS
  px target below.
- **The frame gets the height, and nothing below it competes.** This screen
  makes one judgement — does the overlay sit on the real paint — and it cannot
  be made small. The controls once left the frame at **185 × 104 CSS px** on a
  390 × 844 phone, with a 300 px "Court anchors" card (head, help paragraph, and
  six chips naming each numeral's landmark) taking 57% of the control area to
  restate what the pucks already draw. That card is **deleted**, not shrunk: the
  pucks carry the numerals, `#instr` carries the affordance, and
  `confirmDivergence` names corners by their human `WALL_CORNERS` label, so
  nothing on the screen needed the lookup. Below the frame sit only the verdict
  line, the two secondary actions, and the reasons. The frame reaches its
  full-bleed ceiling (**390 × 219** at zoom 1 for 16:9), and the black well left
  over is the pinch-zoom headroom this screen previously had none of — zooming
  into a 104 px stage yielded a 104 px slit. Do not reintroduce a card here; any
  new copy must justify itself against the frame it takes height from.
- `#instr` and `#confirmSummary` are split by job, never restating each other:
  `#instr` is the action ("Drag any anchor that sits off the line."),
  `#confirmSummary` is the verdict alone. Both are one line.
- The two secondary actions (`UNDO DRAG`, `TAP IT MANUALLY`) size to their text
  via `.confirmActions>*{flex:0 0 auto}` instead of `.row`'s default `flex:1`.
  They keep the 44 px touch height; two full-width outlines gave secondary
  controls more weight than the frame above them.
- **Green is earned, never defaulted into.** The `.status ok` line
  ("Court detected.") requires all of:
  a detection is loaded, its `status` is `ok`, its `confidence` is `high`,
  its `checks_verified` is above zero, no check came back `off`, and all six
  anchors were derived. `checks_verified` is tested separately from
  `confidence` deliberately — it is the count of independent evidence behind
  the verdict, so green never rests on the server's verdict alone. Anything else is
  the `.status warn` line, and the reasons are listed below it. The earlier
  version made green the *default* — a screen with no detection at all and
  zero anchors placed still rendered `class="status ok"`.
- The floor wireframe on *this* screen shares that same two-colour ok/warn
  signal rather than the floor wizard's own per-landmark residual palette
  (§8.10's `#floorDiagram`/`drawFloorOverlay`, dim → active → done → warned,
  which turns a marker red past a 4 px residual). This screen has no red to
  spend — it is not a wizard tracking individual landmark quality, it is one
  fitted picture the player either accepts or corrects — so the wireframe and
  the anchor pucks always agree on a single colour. `drawWallOverlay`'s
  `#ffd60a` quad and numbered pucks are **phase-guarded off** here for the
  same reason: unguarded they render under the confirm pucks, putting amber
  `#f5c518` markers on a `#ffd60a` line.
- Anchors are draggable: a drag moves that anchor and refits the wall/floor
  homographies live, so the wireframe and pucks follow the finger.
  **Their colour does not.** The ok/warn verdict comes from the server's
  self-verification checks, which are distance-to-paint lookups against a mask
  that only exists on the server, so a drag cannot recompute them. The screen
  says so in words rather than letting the stale colour imply otherwise:
  "This verdict is from the original detection; dragging an anchor moves it
  but does not re-check it." Do not restate this as live re-checking unless
  the checks actually become client-side.
  A drag never touches the fitted line overlays (`#35e0ff`/`#ff9f43`/`#b4ff3a`
  above) — those come from a detected fit through hundreds of edge pixels,
  authoritative over any two-point fit a dragged corner could produce (spec
  §8.1). If a dragged wall corner walks more than a few pixels off the fitted
  out line, `#confirmDrift` names it rather than silently trusting either the
  drag or the fit.
- **Drag targets are 44 CSS px** (§0.6), computed from the canvas's *displayed*
  width, not its pixel buffer: `(44 / 2) * (S.W / canvas.getBoundingClientRect().width)`.
  `getBoundingClientRect()` already carries the pinch-zoom transform, so the
  target stays 44 px on the glass at any zoom. A canvas-pixel constant cannot
  do this — the previous `Math.max(18, S.W/60)` measured 13.0 CSS px at
  390×844 on a 1080p clip, and its `S.W` terms cancel, so resolution never
  helped. Because the target is still larger than the 20 CSS px puck, the
  grab carries an **offset**: the anchor tracks the finger's movement rather
  than jumping to it, so the bigger target costs no placement precision.
- **Drags are undoable.** `UNDO DRAG` sits beside `TAP IT MANUALLY`, matching
  every other canvas phase (`wallUndoBtn`, `floorUndoLandmarkBtn`,
  `clearTapsBtn`). It is disabled until a drag has actually moved an anchor —
  a grab-and-release with no movement pops its own snapshot — so the control
  never promises an undo that would do nothing. Without it the only escape
  from a bad drag was `TAP IT MANUALLY`, which discards the whole detection.
- Three status strings, split by *when* their text can change, which §0.9
  requires here rather than merely preferring. The canvas above is vertically
  centered in a flex-grow stage (§7), so any height change below it reflows the
  stage and re-centers the canvas — under an in-progress drag that corrupts the
  pointer math (a steady drag jumped ~50 px the instant a sentence wrapped).
  - `#confirmSummary` — the one-line verdict. Fixed while on screen.
  - `#confirmWarn` — every reason: off checks (by human label), anchors not
    derived, the detector's own warnings verbatim, and the drag caveat above.
    All come from `S.detect`, which a drag never rewrites, so its height
    settles on arrival and it needs no reserve beyond `.status`'s one line.
    Both of these stay in flow below the frame and default to `&nbsp;` so they
    never collapse to zero height.
  - `#confirmDrift` — the only drag-varying text (which dragged corners left
    the fitted out line). It lives **inside `#stage`** as an absolutely
    positioned bottom banner, `hidden` whenever it has nothing to say or the
    phase is not `confirm`. Having no layout height, it *cannot* reflow the
    canvas — a stronger guarantee than the `min-height:60px` reserve it
    replaced, and it costs the frame none of the 60 px that reserve did. It
    takes stage-overlay treatment (§8.12: white on a bottom scrim,
    `text-shadow`, `pointer-events:none`) rather than `.status`, whose
    `var(--text)` would be dark-on-dark over the scrim in the light theme.
- Detection runs behind the sanctioned analyzing scrim (§8.12).
- Primary is the proxied `USE THIS CALIBRATION` (§3.4); the manual wizard is
  always one tap away via the secondary `TAP IT MANUALLY`.
- Detection failure copy leads with the reason detection actually failed, in
  human words — `court_detect` orders its warnings so `warnings[0]` is that
  reason, and the client shows it followed by "Tap the lines instead."
  Internal entity names (`front_seam`) never reach the screen, and an
  incidental note like "Camera appears to be moving" must never displace the
  real cause: on the product's own fin mount that sends the player off to
  re-mount a phone that was mounted fine.

### 8.20 Live-analytics surfaces (Analysis & Progress roots)

The Analysis and Progress roots render **real end-of-pipeline data** — `/api/runs` for
the index, `/api/runs/<id>/report` for the per-tier report, `/api/runs/<id>/coach` for
analytics + feedback — ported from the design lab with its sample data removed. Shared
rules: coach feedback and capability reasons are server text, so both are `escapeHtml`-ed
at every innerHTML sink; run loading is cached per run id and refreshed on every root
entry; every surface has an honest `.emptycard` state ("No analyzed sessions yet…" /
"Trends need at least two analyzed sessions.").

**The user's unit is a session, not a run.** The Analysis root's `.viewhead` meta span
counts sessions and singularizes at one ("1 session" / "2 sessions"). A *run* is a job
the server executed; the reader has a match they played. The old "3 runs · live
pipeline" spent a whole line naming our plumbing — and read "1 runs" the first time
anyone used the app.

**A run qualifies for the list on ANY tier having produced something** — never on wall
hits. The list originally required `has_analytics` plus `total_wall_hits > 0`, which
silently dropped exactly the clips the analysis ladder exists to serve: 30 fps
camera-roll footage that yields rally structure and player movement but no ball tracking
had nothing to show and appeared as "no analyzed sessions".

**Absence is always rendered as its reason, never as an empty result.** This is
Principle 3 of the analysis design carried into the UI. A tier that could not run shows
its `capabilities[tier].reason`; a run predating capability gating shows its
`legacy_reason`. The ball-tier surfaces (Unforced errors / Floor bounces, feedback rows)
are **omitted entirely** when that tier did not run — drawing em-dashes reads as "we
looked and found nothing", which is the single claim the capability card exists to
prevent.

**Which metrics are user-facing was decided in the 2026-07-29 metric review.** Unforced
errors (out + tin), rally length, zone usage, floor bounces, and player movement are the
coaching surface; line-call rates (IN %), ball pace, and shot height are still computed,
persisted, and fed to the LLM coach, but are **not** shown as tiles, gauges, or trends.
Do not reintroduce them to the UI without a deliberate DESIGN.md change.

- **Analysis run cards** (`.clipcard`): §8.9 card at padding 0, **head row only** —
  Analysis is a list of matches, not a stack of unrolled reports. Head row
  (`.cliptop`): 62×46 gradient thumb (hue rotates per run — the sanctioned gradient)
  with a court line-sketch + play glyph · run date 14/600 + duration `.metaline` ·
  right-aligned `ANALYZED` `.statechip` · a `.deleteRunBtn` trash icon (36 px round,
  `--dim`, poor-tint hover). **The head row is the card's one action:**
  tapping it opens that match's analysis page, `p-match` (§16). It is a `<div>`, so it
  carries `role="button"`, `tabindex="0"` and an Enter/Space handler; without them a
  match would be reachable by pointer only. The trash button is the row's only other
  control (it stops propagation) and deletes by the §8.16 arm pattern, never a modal:
  first tap swaps the icon for a 12/700 "Confirm" accent pill (`.arm` — the
  `aria-label` flips with it), which disarms back to the icon after ~2.6 s; a failed
  delete restores the idle icon and reports through the §8.6 error banner. Native
  `confirm()`/`alert()` cannot be used — the iOS shell's WKWebView renders no JS
  dialogs, so they silently no-op. The analysis itself (`#matchBody`) is
  rendered on that page, in ladder order — rally structure first because it is the
  tier that always runs, ball detail last because it is the one most often gated off:
  **"Rallies"** — one full-width `.scol` tile: **Longest rally**, labeled with the
  rally number and the running score it happened at when the hit-derived rallies can
  name it ("Longest rally · Rally 7 · at 4–3"). The signal it came from ("From impact
  sounds and frame motion" / "From frame motion only — no audio track") and the
  disagreement note when the timeline and the hit-derived rallies differ are
  **provenance about a number that is shown**, so they ride in a §8.23 disclosure on
  the *heading* — `sectionHead('Rallies', why)` — not in a `.metaline` under the tile.
  The user-facing wording of that disagreement names ball contacts, not hits ·
  **"Movement"** — one four-tile `.scorecols` row per player (distance / On the T /
  Front % / Back %), with the detector backend and the fraction of rally time actually
  observed in the same §8.23 disclosure on the heading. A motion-blob fallback and
  real weights are not equally trustworthy and the card must not present them as if
  they were — but that is a caveat about numbers the card *does* show, so it sits one
  tap away rather than spending a line under every row ·
  **ball tier, only when it ran** — two `.scol` tiles (Unforced errors / Floor
  bounces) · `.fbrow` feedback sentences ·
  **"Not measured"** — rendered **only when a tier was gated off** (or the run is
  legacy). One `.fbrow` per disabled tier, label left, `OFF` `.statechip` right,
  and the `reason` as dim `.s` text under the label. When every tier ran the whole
  section is omitted: the card exists so absence is rendered as its reason, and
  four `ON` chips assert nothing a reader needs. The reason stays **visible body
  text** — never behind the §8.23 disclosure ·
  a dim provenance `.metaline` ("run id"), marked `.devOnly` — a 13-digit epoch id
  names a job to whoever is sitting at the Mac and nothing at all to a player, so it
  leaves the native shell with the rest of the workshop chrome ·
  one `.ghostbtn` **"Open full review"**, the only entrance to the frame-by-frame
  call page (§16, `p-track`), which owns "Watch source video". One level per screen:
  the list names the matches, this page reads the analysis, the review page judges
  frames. The primary identity choice lives in the full review's Players card: each
  crop/name card ends with a full-width **"This is me"** button, and the selected card
  receives the accent treatment plus a checkmark. The Analysis row repeats the choice
  as `.selfSelector` so an older completed match can still be identified without
  reopening review. Both surfaces persist `user_player_number` for that run through
  `POST /api/runs/<id>/me` and remain synchronized. This is a per-run choice because
  Player A/B are match-local tracks, not permanent identities.
- **Training stats** (`p-stats`): a live personal summary pooled only across the runs
  with a `.selfSelector` choice — front-wall shot count, unforced-error share, wide
  usage, low attacking rate, average wall height, and most-used target zones. With no
  identified run it directs the user back to Analysis; it is not a roadmap placeholder.
- **Progress** (`p-progress`): `.seg` range picker (1W/1M/3M/6M) · `#deltastrip` —
  three `.delta` cards (radius 18) with 11 dim label, 20/700 value, and a `.chip`
  delta vs the previous run (`good`/`poor` tint, `flat` when unchanged; direction-aware
  — for Unforced errors *down* is `good`) over Unforced errors / Time on the T / Wide
  usage · `.trend` cards (Unforced error share, Time on the T, Distance covered, Wide
  usage, Low attacking) — 15/600 label, 26/700 value + unit + `vs typical`
  (tint-colored, direction-aware), and an inline-SVG spline chart: quartile "typical
  band" (`--seg-bg`), dashed midline, accent line + soft area fill, accent end-dot with
  the value labeled. A range window holding fewer than two identified sessions says so in the card
  — it never silently substitutes the all-time series. · a `.bestrow` card (best
  personal T-position mark). Baselines are the selected player's own per-run quartiles
  — never invented targets. Every point first resolves that run's `user_player_number`,
  so a Player A selection in one run and Player B in another remain one personal series.

### 8.21 Review pane switcher (`#reviewSeg`)

The three-way `.seg` (§8.1) that moves between the match review page's panes:
`Call | <player 1 name> | <player 2 name>`, labelled by `playerDisplayName(n)` so a run
with saved names reads `Call | Ian | Sam`. Exactly one pill active; a tap calls
`setPhase()` on that pane's phase.

- **It sits between `<header>` and `#instr`, outside `<main>` — deliberately.** Every
  review pane makes `main` its own scroller (`body.phase-track` / `body.phase-target`,
  §3.1), so a switcher inside `main` would scroll away from the page whose whole purpose
  is moving between panes.
- It is chrome for **one page**, not a router: it renders only on the review phases and
  the §8.3 dock remains the app's only section tab bar (§18). It takes the 14 px page
  gutter (`margin:2px 14px 8px`) so it lines up with the content below it, and its
  height is subtracted by the §3.1 stage clamp.
- The §8.1 `.seg` recipe verbatim, no fork — which inherits `.seg`'s 34 px pills (42 px
  including the track's 4 px padding). On a primary navigation control that is **§0.6
  debt** of the same standing as `.corrSeg`'s 38 px (§8.18): a gap owed a fix, not a
  precedent for sizing new controls at 34 px.
- Pane labels are re-read on every pane entry and after a name is saved, so the switcher
  and the report headings can never disagree about who Player 1 is.
- The switcher is the only movement *between* panes. Back is not a pane control — it
  leaves the page (§16).

### 8.22 Rally attribution provenance (`.rallySegment.attr-*`, `.rallyLegend`, `.attrProvenance`)

Every shot on the review page was assigned to a player by parity within its rally, not
by watching who hit it. This component states, per rally, how much of that is evidence
and how much is a guess — it is the honest counterweight to a report that otherwise
reads as fact.

For new runs, Player A is defined as the first server. Front-wall contacts alternate
A/B within a rally, and the inferred winner becomes the next rally's server. Person
detection maintains tracks and supplies the two naming photos; it does not override
the serving sequence. Consequently new runs use `attr-assumed`. The other states below
remain supported so previously stored observed-attribution runs still render honestly.

**Ribbon states** — modifier classes on the rally segmentation ribbon's `.rallySegment`
(§16, `p-track`):

| State | Treatment | Means |
|---|---|---|
| `attr-observed` | the ribbon's plain neutral segment, no marker | serve was seen and the next rally's serve corroborates it |
| `attr-repaired` | same neutral segment + a `↻` marker (10/700, top-right, `--dim`) | serve was a guess; the next observed serve corrected it |
| `attr-assumed` | `--bg` fill, `--dim` ink, `1px dashed --line` | serve was a guess with nothing to check it against |
| `attr-conflict` | `--tint-poor-bg` fill, `--tint-poor-fg` ink, dashed `--tint-poor-fg` border + a `!` marker | two independent observations disagree |

`.active` — the rally under the playhead — still wins over all four. It marks position in
the clip, a different axis from provenance, and the page would be unreadable if one axis
could hide the other.

**This is a data-quality split, not a verdict and not a calibration edge (§5.2).** Dashed
vs solid carries *not confirmed* (§8.14, §13); the quality tints carry the one
contradicted state; `--in`/`--outcall` and `--out`/`--service`/`--tin` are never borrowed
here.

Colour is never the only carrier (§0.3): each segment's `title` and `aria-label` name its
state in words ("serve assumed", "attribution conflict"), and the `.rallyLegend` beneath
the ribbon — 11 px dim rows, 12 px swatch + the same words — spells out the key. Each
swatch is a **miniature of the segment it decodes**, marker glyph included (§8.11: the
legend and the thing it decodes share tokens, or the legend lies). That is why `observed`
and `repaired` swatches share the neutral segment fill and are told apart by the `↻` inside
the swatch: accent belongs to `.active`, so a legend that painted these accent would decode
a colour the ribbon never shows. The legend renders **only when two or more states are
present**: a run where every rally was observed shows none, because a key full of warnings
it never triggered makes a clean run look uncertain.

**Provenance line** (`.attrProvenance`) — one 13/400 `--dim` sentence at the top of each
player's report panel (§8.17), counting **rallies, not shots**, because the rally is the
unit attribution is decided in: `Shot attribution across 4 rallies: 1 assumed · 1
corrected from the next serve · 1 conflicting.` An all-observed run reads `Shot
attribution observed across all 4 rallies.`; a run with no rallies hides the line
entirely rather than printing a zero. It takes `.warn` when any rally conflicts —
`--text` at 700, still no red (§13) — so the loud case is loud in weight, not in hue.

### 8.23 Provenance disclosure (`.whybtn` / `.whynote`)

Provenance and confidence about **a number that is shown** ride behind a tap, not
in body copy. A `.whybtn` — 15 px circled-`i`, `--dim`, inside a 44 px target
whose negative margins keep the heading row its original height — sits at the end
of the heading the caveat belongs to. Tapping toggles a `.whynote` (13 / `--dim` — the
Sub-meta rung of the §6.1 ladder, same as `.metaline`)
directly beneath. Built by `whyDisclosure(text)` / `sectionHead(label, why)`.

- Collapsed by default. `aria-expanded` + `aria-controls`; glyph `aria-hidden`.
  `aria-label="Why this number"`.
- Plain show/hide via `.hidden`. No animation, so §10's reduced-motion rule has
  nothing to suppress.
- `whyDisclosure` escapes its own text — callers pass plain strings.
- The click handler is **delegated** from `document`, because every call site
  renders into an innerHTML sink that is rebuilt on each report load.
- **One button per heading, never per tile.** Several facts about one section
  concatenate into a single note.
- **Player report panel** (`#coachTitleP*` / `.coachHedge`): the sample-size hedge
  from `coaching_advice.low_sample_note` hangs off the "Player N report" heading,
  because the shot count qualifies every number below it — the tiles, the drills,
  and the LLM narration, which quotes those numbers flat however the prompt is
  worded. Present whenever fewer than `LOW_SAMPLE_HITS` (20) shots were analyzed,
  including when no weakness stood out; that reading is the one most in need of it.
- **Never route a capability `reason` through this.** §8.20 requires that the
  reason a tier could not run stay visible; hiding it re-creates the "we looked
  and found nothing" reading that card exists to prevent.

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
| Wipe | `.5s cubic-bezier(.25,0,.3,1)` | theme swap only — a `clip-path` circle on `::view-transition-new(root)` opening from the theme button until it clears the farthest corner. Percentage geometry is mandatory, see §3.5 |
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
| Empty / first-run | Dashboard (§8.15) with its markup-default hero copy; live regions stay hidden; Analysis/Progress show `.emptycard` sentences (§8.20) |
| Working (known progress) | `.progressbox` with real stats (frames, fps, ETA) + determinate bar |
| Working (unknown) | Indeterminate bar or stage scrim + pulsing uppercase label |
| Inline status | `.status` line (14 dim), reserved height; `.warn` = 700 `--text` (still no red), `.ok` = `--text` |
| Unverified / contradicted data | Dashed border + dim fill for *not confirmed*; `--tint-poor-*` only where two observations disagree; weight, never hue, for the loud case — never `--in`/`--outcall` or a calibration hue (§8.22 is the worked example) |
| Error | `#errBanner` top banner: bold message + Dismiss. Recoverable, calm, specific |
| Result | Verdict box state change within reserved space |
| Placeholder | Dashed-border `.blank` treatment; full-page placeholders use `.placeholderHero` (§8.14) |

Copy for statuses is specific and actionable ("Tap the two ends of the out line", not
"Error"). Numbers over adjectives ("132/300 frames" over "working…").

---

## 14. Voice & copy

**Referee's voice: calm, terse, factual.**

- Verdicts and telemetry: uppercase single words (IN, OUT, ANALYZING…).
- Buttons: verb-first, ≤ 3 words ("Analyze", "Use this frame", "Judge frame").
- Instructions: one sentence, present tense, name what the user sees ("Load a clip from
  this phone to begin."). Colored keywords (`b.out`, `b.service`, `b.tin`) when referring
  to fitted lines.
- Domain terms exactly: *out line, tin, service line, front/side wall, floor, rally,
  bounce* (never "boundary", "net", etc.).
- No exclamation marks, no praise ("Great!"), no anthropomorphism. The app states facts.
  Single sanctioned exception: the "Coming soon!" hero text on roadmap placeholder pages
  (§8.14) — nowhere else.
- **Provenance and confidence sit behind a disclosure, not in body copy** (§8.23).
  State the fact; offer the caveat. The exception is *absence* — the reason a tier
  could not run, or that no rallies were found, stays visible (§8.20).

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
| `p-load` | Dashboard section root | ink hero (focus + ring + stat tiles + note) · accent "Analyze a clip" file action · Coach Notes rail · dev row — full spec §8.15 (native replaces the file action with its record screen — see §3.2's native-shell note) | — (no chevron; section root) |
| `p-record` | Record rallies + on-site calibration | stage = live camera preview · REC readout · Calibrate court + calibration status · Recordings card (§8.16) | "Record" ↔ "Stop" |
| `p-frame` | Pick a clean calibration frame | overview rail · editor strip w/ playhead · readout · transport+steppers | "Use this frame" |
| `p-tap` | Tap out line, tin, then service line on frame | stage-driven; clear-selection small button | "Looks right" (disabled until the current line has a fit) |
| `p-review` | Approve fitted lines (cyan/amber/lime on stage) | minimal; evidence is the stage | "Use these lines" |
| `p-tap-floor` | Floor calibration wizard | `.floorRow`: diagram (progress marks) + prompt/side actions · skip-all / save-profile (tapping "Save as profile" swaps that row **in place** for a §8.4 text input + Save of the same 48 px height — never `window.prompt`, which the iOS shell's WKWebView leaves unimplemented; Esc or an empty-field blur restores the buttons) | "Use floor map" |
| `p-clip` | Trim rally clip | overview · trim editor (accent handles) · transport+readout row · start/end nudge steppers · full-width "Select entire clip" secondary · frame summary | "Analyze" |
| `p-analyze` | Honest processing | `.progressbox` stats + bar (+ stage ANALYZING pulse) | — (auto-advances) |
| `p-track` | **Match review — Call pane.** Review track, judge calls, name the players, identify yourself | control area keeps its pre-rally-visualization height so the video stage does not shrink, floored against the stage per §3.1; the added content scrolls inside that footprint · scrub hint lives in the header `#instr` line (detection failures replace it, `.warn`) · per-rally front-wall impact mini-map · rally segmentation card (proportional neutral ribbon, active segment in accent, `attr-*` provenance states + legend §8.22; per-rally winners/scores stay backend-only per the 2026-07-29 review) · overview w/ marker minis · hit timeline (neon bars, center playhead) · readout · transport · frame input + Judge row · verdict box · Players card (two equal-width detected-player cards, each with a 4:5 crop, its own name field, and a full-width "This is me" identity button; a quiet "No photo available" placeholder preserves the pair when an old run has only one crop) · ghost "Watch source video". One pane, no switcher — the Challenge pane and its dock were archived 2026-07-29 (`archive/challenge-ui/`) | "Judge frame" |
| `p-player1-report` / `p-player2-report` | **Match review — Player 1 / Player 2 panes.** Per-player coaching report | Player N front-wall map (§8.10 court chart + `.targetMeta`; serves excluded) · Player N report panel (§8.17, opening with the §8.22 provenance line) · Player N movement panel (§8.17: distance / position split / speeds + court heatmap) · **P1 only:** the run's floor-bounce map (§8.10 `#floorMapSvg` + `.targetMeta`) — bounces are per-run, not per-player, so the panel renders once under the first report rather than twice | — (no primary) |
| `p-label` | Human bounce labeling | overview · label timeline · transport+zoom · 2-col type grid (dot+label) · delete (destructive = plain secondary, disabled until selection) | — |
| `p-matches` | Analysis section root — session library | view head (`n runs · live pipeline` meta only) · one `.clipcard` per analyzed run: head row opens that match's analysis page and `.selfSelector` identifies which player is the user (§8.20) · `.emptycard` when none | — (no chevron; section root) |
| `p-match` | **Match analysis.** One analyzed match, read end to end | view head (clip duration only — the match date is the header label) · `#matchBody`: the §8.20 analysis stack (Rallies · Movement · ball tier when it ran · "Not measured", only when a tier was gated off · provenance line) · ghost "Open full review" · `.emptycard` when the run is gone | — (back chevron only, like the review panes; the native shell hides its tab bar and settings gear here, §3.2) |
| `p-coach` | Training section root (hub) | three feature cards (§8.13), including the live Your coach entry — no view head, it held only the title | — (no chevron; section root) |
| `p-coach-advice` | Your coach — Ollama feedback and drills from the user's identified match history | headline card — the coach's own headline, with the shot/session counts, the reading order and the pooling caveat behind one §8.23 disclosure on it; a failure state keeps its reason as visible `.whynote` body copy · chronological observations · drill cards (§8.13a) | — (back chevron → Training hub) |
| `p-progress` | Progress section root — personal cross-session trends | range `.seg` (no view head, it held only the title) · delta strip · trend cards · best-mark card (§8.20) · `.emptycard` under two identified runs | — (no chevron; section root) |
| `p-live` | Placeholder: live match | placeholder hero · Planned card (§8.14) | — (back chevron only) |
| `p-stats` | Personal training stats pooled across identified runs | view head (pooled-session meta only) · personal metric card · target summary · source statement | — (back chevron only) |

Section roots and their leaves carry **no in-page `<h2>`** — `#stepLabel` (an
`<h1>`) is the screen's only title, as `p-load` has always done. A `.viewhead`
survives only where it lays out a dim meta span carrying something the header
does not say (session count, clip duration).

**The match review page.** `p-track` + `p-player1-report` + `p-player2-report` are one
page in three panes (§3.3), reached only by finishing an analysis or by opening a match
from Analysis. Header on every pane: back chevron · the match date as `#stepLabel` ·
theme toggle — **no step number**. The run is committed, so there is
no calibration behind the page to walk back into and no position in a flow to report;
`stepSequence()` therefore ends at `clip`. Only the Call pane has a primary ("Judge
frame", proxied per §3.4) — a report pane has nothing to act on. The §8.21 switcher is
the only movement between panes; Back exits to Analysis from **any** pane, never
retreating P2 → P1 → Call.

`p-load` is the **Dashboard section root** (§8.15). Sub-page back routes: `p-record` →
Dashboard; `p-live` → Dashboard; `p-stats` → Training; every
review pane → Analysis. The calibration wizard (`p-tap` → … → `p-tap-floor`) serves two
flows: entered from `p-frame` it exits to `p-clip`; entered from `p-record` ("Calibrate
court", on a frame frozen from the live camera) it exits back to `p-record` and the
result rides along with each recording. Analysis completion routes to the review page and
tears the run's calibration and clip state down with it, so the wizard is unreachable for
a committed run — starting a new analysis from the Dashboard is the only way back into
it. The nav dock is the section tab bar and appears on the four section roots only
(§8.3). Deep links: `#tab=load|matches|coach|progress` boot straight to a root; `#run=<id>`
seeds the restore stash and lands on the review page — the same stash `p-match`'s
"Open full review" seeds (§8.20).

Production surfaces (section roots and placeholder pages) carry **no `#instr` guidance
copy** — the UI leads; onboarding will teach, later. The `#instr` strip remains for the
working tool flows (calibration, clip, call), whose instructions are operational, not
explanatory. The Play hint line (`#loadHint`) is empty in judge mode (reserved height,
no CLS) and speaks only for the dev-row label mode. A healthy backend is silent —
`#loadStatus` only reports problems.

Mode switch Judge ↔ Label lives in the Dashboard dev-row toggle (§8.15). The call
page once carried a second instance of the same dark dock, switching Review ↔
Challenge; that pane and its dock were archived 2026-07-29
(`archive/challenge-ui/`), so **`#navPill` is the only dock in the app** — a
second one is a thing to justify, not a pattern to copy.

**The `SOON` tag on "Live match".** §8.15's rule is a question about one surface — does
tapping *this* card lead to a working experience. `p-live` is a roadmap placeholder on
web, so the card keeps its `SOON`. The native app does not show the card at all: its
Play tab is the record screen directly.

A two-camera implementation of live match was built and then archived on 2026-07-27 —
the blueprints for its `p-pair` screen, its live stage, and the four components they
used (call flash, call banner, link status, pair code) are preserved in
[archive/stereo/docs/DESIGN-live-sections.md](archive/stereo/docs/DESIGN-live-sections.md).
Restoring that path means restoring those sections here in the same change. The one rule
from it worth carrying forward regardless of how live match is eventually built:
**capability is added, never gated** — whatever a live mode needs, a phone without it
must still record exactly as it does today.

---

## 17. Extending the system (roadmap: live match mode, auto editor, stats/AI coaching)

1. **New screen** = new phase `<section id="p-…">` + step-label entry + proxied primary.
   Reuse §8 components; a new component requires a new subsection here first.
2. **New color** = new `:root` token with a light override and a §5.2 family assignment.
   PR must state which family it joins. Hex literals outside tokens are allowed only in
   the canvas palette (§4.3) and court miniature (§8.10).
3. **Live-mode surfaces**: verdict box and marker grammar scale up — a live call is a
   full-stage verdict-colored flash + uppercase word, not a new visual language. The
   archived two-camera implementation's components are specified in
   [archive/stereo/docs/DESIGN-live-sections.md](archive/stereo/docs/DESIGN-live-sections.md);
   start from those rather than inventing a second grammar. Stats/coaching screens are
   §8.9 cards + §12 rules stacked in `<main>`.
4. **CSS lives in the `<style>` block of `index.html`**, grouped by component with the
   existing terse `/* purpose */` comments; JS constants that mirror tokens (e.g. `CFG`
   hexes) must be updated in lockstep with `:root`.
5. When in doubt, find the closest existing screen in §16 and copy its structure.

---

## 18. Never do

- No webfonts, icon sets, or CDN/remote assets of any kind.
- No second accent; no green/red outside verdicts (won/lost data splits included — §8.17;
  the `--tint-*` chip pairs are the sanctioned soft palette for data quality);
  no calibration hues (cyan/amber/lime-yellow) outside calibration.
- No ad-hoc shadows — only `--shadow-card`/`--shadow-btn` and the §5.1 sanctioned set;
  no glass anywhere; no decorative gradients outside the court miniature and the
  Analysis run thumbs.
- No fake or sample data on production surfaces — live regions hide or show an honest
  empty state instead.
- No spinners where real progress or evidence is possible; no fake progress.
- No layout shift from appearing content; no moving playhead (the strip moves).
- No page-transition animations; nothing animated longer than 500 ms except ambient
  pulses (§10 Micro at .28s). The Analysis card expand this used to sanction is gone
  entirely: the card is a head row and the analysis is its own page (§8.20).
- No hover-only affordances; no touch targets under 44 px; no removal of safe-area math.
- No scrolling app shell; no second header/tab-bar/nav chrome.
- No uppercase buttons or paragraphs (uppercase is for chips/tags/micro-labels/telemetry
  only); no exclamation marks.
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
