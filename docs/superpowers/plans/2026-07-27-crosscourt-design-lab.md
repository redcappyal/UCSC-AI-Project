# CrossCourt Design Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `design-lab/crosscourt-lab.html` — a standalone, self-contained 4-tab mobile prototype translating the TennisIQ inspiration into CrossCourt's squash feedback-loop product.

**Architecture:** One HTML file with inline CSS/JS and inline fake data. Four `<section class="view">` screens (Dashboard, Analysis, Training, Progress) toggled by a bottom nav pill. A 390px phone frame centered on a desktop-gray page. SVG charts drawn by a small spline helper. No build step, no server-side anything.

**Tech Stack:** Plain HTML/CSS/JS. System font stack. Verification via the Browser pane using a `python3 -m http.server` static server.

**Spec:** `docs/superpowers/specs/2026-07-27-crosscourt-design-lab-design.md` — read it before starting any task. Its "Visual language" section governs every aesthetic decision.

## Global Constraints

- **Zero external requests**: no CDN, no fonts, no image files. Icons are inline SVG (24×24 viewBox, stroke `currentColor`, stroke-width 2). Thumbnails are inline SVG/CSS gradients.
- **This file must not import from or modify the shipping app** (`index.html`, `DESIGN.md` untouched).
- Light theme only. Phone frame content width 390px.
- Palette (exact values): canvas `#F2F3F5`, card `#FFFFFF`, ink `#17181A`, dim `#8A8F98`, accent `#D9F24F` (always dark ink on it), severity tints good `#E4F5E9`/`#1E7A3C`, average `#FBF3DD`/`#9A6B00`, poor `#FBE4E1`/`#B3362A`.
- Cards: radius 20px, no borders, shadow `0 1px 3px rgba(20,22,25,.06)`, 12px stack gap.
- Stat numerals: 700 weight, `font-variant-numeric: tabular-nums`.
- Squash vocabulary only (straight drive, boast, ghosting, tin, out line, service line, rally) — never tennis terms.
- Every control that looks primary must respond visibly (expand, tab switch, toast, state flip). No dead taps on primary affordances.
- Commit after every task.

**Verification setup (used by every task):** `.claude/launch.json` gets a static-server entry in Task 1. Open the preview with `preview_start {name: "design-lab"}`, then navigate to `http://localhost:8123/design-lab/crosscourt-lab.html`. "Verify" in a step means: reload the tab, check rendering matches the listed acceptance points, and check `read_console_messages` shows no errors.

---

### Task 1: Scaffold — phone frame, tokens, nav pill, view switching

**Files:**
- Create: `design-lab/crosscourt-lab.html`
- Create: `.claude/launch.json` (if absent) with the static server entry

**Interfaces:**
- Produces: CSS custom properties on `:root` (`--canvas`, `--card`, `--ink`, `--dim`, `--accent`, `--tint-good-bg/fg`, `--tint-avg-bg/fg`, `--tint-poor-bg/fg`, `--radius:20px`, `--shadow`); four empty `<section class="view" id="view-dashboard|analysis|training|progress">`; JS `switchTab(name)` global; `toast(msg)` global; `.card`, `.pill`, `.icontile`, `.chip` utility classes all later tasks rely on.

- [ ] **Step 1: Create the launch config**

If `.claude/launch.json` does not exist, create:

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "design-lab",
      "runtimeExecutable": "python3",
      "runtimeArgs": ["-m", "http.server", "8123"],
      "port": 8123
    }
  ]
}
```

If it exists, append the configuration to its `configurations` array.

- [ ] **Step 2: Write the scaffold**

Create `design-lab/crosscourt-lab.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CrossCourt — Design Lab</title>
<style>
:root{
  --canvas:#F2F3F5; --card:#FFFFFF; --ink:#17181A; --dim:#8A8F98;
  --accent:#D9F24F;
  --tint-good-bg:#E4F5E9; --tint-good-fg:#1E7A3C;
  --tint-avg-bg:#FBF3DD;  --tint-avg-fg:#9A6B00;
  --tint-poor-bg:#FBE4E1; --tint-poor-fg:#B3362A;
  --radius:20px; --shadow:0 1px 3px rgba(20,22,25,.06);
}
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{background:#E4E5E9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:var(--ink);display:flex;justify-content:center;min-height:100vh;padding:24px 0}
.phone{width:390px;background:var(--canvas);border-radius:34px;overflow:hidden;
  box-shadow:0 8px 40px rgba(20,22,25,.18);display:flex;flex-direction:column;
  height:844px;position:relative}
.screen{flex:1;overflow-y:auto;padding:18px 16px 96px}
.view{display:none;flex-direction:column;gap:12px}
.view.active{display:flex}
.card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px}
.pill{border:0;border-radius:999px;min-height:48px;padding:0 18px;font:600 15px inherit;
  font-family:inherit;display:flex;align-items:center;justify-content:center;gap:8px;
  cursor:pointer;color:var(--ink);background:var(--card);box-shadow:var(--shadow)}
.pill.accent{background:var(--accent);font-weight:700}
.pill:active{filter:brightness(.96)}
.icontile{width:36px;height:36px;border-radius:12px;background:#EEEFF2;display:flex;
  align-items:center;justify-content:center;flex:none;color:var(--ink)}
.icontile svg{width:18px;height:18px}
.chip{display:inline-flex;align-items:center;border-radius:999px;padding:4px 10px;
  font-size:12px;font-weight:600}
.chip.good{background:var(--tint-good-bg);color:var(--tint-good-fg)}
.chip.avg{background:var(--tint-avg-bg);color:var(--tint-avg-fg)}
.chip.poor{background:var(--tint-poor-bg);color:var(--tint-poor-fg)}
.num{font-variant-numeric:tabular-nums;font-weight:700}
h2{font-size:22px;font-weight:700}
.dim{color:var(--dim)}
/* nav */
nav{position:absolute;left:50%;bottom:16px;transform:translateX(-50%);display:flex;gap:4px;
  padding:6px;border-radius:999px;background:rgba(255,255,255,.85);
  backdrop-filter:blur(16px);box-shadow:0 6px 24px rgba(20,22,25,.14)}
nav button{border:0;background:none;border-radius:999px;min-height:44px;padding:0 14px;
  display:flex;align-items:center;gap:6px;font:600 12px inherit;font-family:inherit;
  color:var(--dim);cursor:pointer}
nav button.active{background:var(--accent);color:var(--ink)}
nav button svg{width:17px;height:17px}
/* toast */
#toast{position:absolute;left:50%;bottom:84px;transform:translateX(-50%);background:var(--ink);
  color:#fff;font-size:13px;font-weight:600;padding:10px 16px;border-radius:999px;
  opacity:0;transition:opacity .18s ease;pointer-events:none;white-space:nowrap}
#toast.show{opacity:1}
</style>
</head>
<body>
<div class="phone">
  <div class="screen">
    <section class="view active" id="view-dashboard"></section>
    <section class="view" id="view-analysis"></section>
    <section class="view" id="view-training"></section>
    <section class="view" id="view-progress"></section>
  </div>
  <nav id="nav"></nav>
  <div id="toast"></div>
</div>
<script>
const TABS = [
  {id:'dashboard', label:'Dashboard', icon:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/></svg>'},
  {id:'analysis',  label:'Analysis',  icon:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="4" width="18" height="14" rx="3"/><path d="M10 9l4 2.5L10 14z"/></svg>'},
  {id:'training',  label:'Training',  icon:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg>'},
  {id:'progress',  label:'Progress',  icon:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19l5-6 4 3 7-9"/></svg>'}
];
function switchTab(name){
  document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active', v.id==='view-'+name));
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active', b.dataset.tab===name));
  document.querySelector('.screen').scrollTop = 0;
}
let toastTimer;
function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(()=>t.classList.remove('show'), 1600);
}
document.getElementById('nav').innerHTML = TABS.map(t=>
  `<button data-tab="${t.id}" onclick="switchTab('${t.id}')">${t.icon}<span>${t.label}</span></button>`).join('');
switchTab('dashboard');
</script>
</body>
</html>
```

- [ ] **Step 3: Verify in the browser**

Start `preview_start {name:"design-lab"}`, navigate to `http://localhost:8123/design-lab/crosscourt-lab.html`. Accept when: gray page, centered 390×844 rounded phone frame in `#F2F3F5`, floating white nav pill with 4 icon+label tabs, active tab is an accent capsule, tapping tabs switches (empty) views, no console errors.

- [ ] **Step 4: Commit**

```bash
git add design-lab/crosscourt-lab.html .claude/launch.json
git commit -m "Design lab: scaffold phone frame, tokens, nav pill"
```

---

### Task 2: Dashboard view

**Files:**
- Modify: `design-lab/crosscourt-lab.html` (fill `#view-dashboard`, add its CSS, add JS at the marked spot before `switchTab('dashboard')`)

**Interfaces:**
- Consumes: `.card`, `.pill`, `.icontile`, `.chip`, `switchTab()`, `toast()` from Task 1.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Add dashboard markup + styles**

Inside `#view-dashboard` (static HTML, no data-driven render needed):

1. **Header row**: `CrossCourt` wordmark — 20px/800, with a 8px accent dot ::before or inline SVG ball glyph; right-aligned 40px circular white bell button (`toast('No notifications')`).
2. **Hero card** (`.hero`): background `var(--ink)`, radius 24, color `#fff`, padding 18px. Contents:
   - 12px/600 dim-on-dark label `FOCUS` (opacity .55) + trend badge right-aligned: `.chip`-style capsule, accent bg, dark ink, text `IN% 78 ▲3`.
   - Headline 17px/700 line-height 1.35: `Straight-drive depth — drives are landing short of the service box in 3 of your last 4 sessions.`
   - Sub-line 13px, rgba(255,255,255,.6): `From your player report · 4 sessions analyzed`
3. **Button row** (flex, gap 10): accent pill `Start Recording` with camera SVG (flex 1.2, `toast('Recording is a lab stub')`), white pill `Upload Video` (flex 1, `toast('Upload is a lab stub')`).
4. **Next up card**: 13px/700 header row `NEXT UP` (letter-spacing .04em, dim) + 2 drill rows. Each row (`.drillrow`, min-height 56px, flex, gap 12, align center): `.icontile` (target / racket SVG) · column of 15px/600 name + 13px dim meta (`Straight-drive depth · 20 min`, `Boast accuracy · 15 min`) · right accent `.chip`-sized Start button (min-height 34px, padding 6px 14px, radius 999, font 12/700, onclick `switchTab('training')`). Rows separated by 1px `#F0F1F3` hairline.

Example hero CSS to add:

```css
.hero{background:var(--ink);border-radius:24px;color:#fff;padding:18px;display:flex;
  flex-direction:column;gap:10px}
.hero .tag{font-size:12px;font-weight:600;letter-spacing:.04em;opacity:.55}
.hero .badge{background:var(--accent);color:var(--ink);border-radius:999px;
  padding:4px 10px;font-size:12px;font-weight:700}
.drillrow{display:flex;align-items:center;gap:12px;min-height:56px}
.drillrow + .drillrow{border-top:1px solid #F0F1F3}
.startchip{border:0;background:var(--accent);border-radius:999px;min-height:34px;
  padding:6px 14px;font:700 12px inherit;font-family:inherit;cursor:pointer;margin-left:auto}
```

- [ ] **Step 2: Verify in the browser**

Reload. Accept when: dashboard fits one screenful (no scroll needed above the nav), hero reads as the loud element, exactly one accent button (Start Recording) plus small Start chips, Start chips switch to the (empty) Training tab, both toasts fire, no console errors.

- [ ] **Step 3: Commit**

```bash
git add design-lab/crosscourt-lab.html
git commit -m "Design lab: dashboard view"
```

---

### Task 3: Analysis view — clip accordion + fake analyze

**Files:**
- Modify: `design-lab/crosscourt-lab.html`

**Interfaces:**
- Consumes: Task 1 utilities.
- Produces: `const CLIPS` data array and `renderClips()`; Task 6 reads nothing from it, but keep the names exactly: `CLIPS`, `renderClips()`, `toggleClip(i)`, `analyzeClip(i)`.

- [ ] **Step 1: Add data + renderer**

Data (inline JS):

```js
const CLIPS = [
  {id:1, date:'Today, 9:41', dur:'12:24', analyzed:true,  score:84,
   scores:{Accuracy:82, Placement:71, Consistency:68},
   feedback:[
     {t:'Good length on straight drives — 74% past the service box.', s:'good'},
     {t:'Boasts landing mid-court. Aim shorter, two racket-lengths off the front wall.', s:'avg'},
     {t:'Serve returns catching the tin under pressure — lift your target above the service line.', s:'poor'}],
   hue:200},
  {id:2, date:'Jul 25, 18:02', dur:'08:51', analyzed:true, score:79,
   scores:{Accuracy:77, Placement:69, Consistency:64},
   feedback:[
     {t:'Rally length up — averaging 9 shots, your best this month.', s:'good'},
     {t:'Cross-court drives drifting to the middle third.', s:'avg'}],
   hue:150},
  {id:3, date:'Jul 22, 17:30', dur:'15:07', analyzed:false, hue:30},
  {id:4, date:'Jul 19, 19:12', dur:'06:40', analyzed:false, hue:280}
];
```

`renderClips()` writes into `#cliplist` inside `#view-analysis`. Card per clip:

- **Collapsed row** (always visible, whole row clickable → `toggleClip(i)`): thumbnail 64×44, radius 12 — inline `linear-gradient(135deg, hsl(hue 30% 82%), hsl(hue 25% 62%))` with a small white play triangle SVG centered; column: 15px/600 date + 13px dim `12:24 · rally clip`; right side: `ANALYZED` chip (accent bg, 11px/700, with `84` score in `.num`) or `NOT ANALYZED` chip (bg `#EEEFF2`, dim text).
- **Expanded body** (`.clipbody`, `max-height:0; overflow:hidden; transition:max-height .25s ease`; open class sets a generous max-height): 
  - Unanalyzed: two meta rows (13px dim: `Recorded on court · phone on fin`, `Calibration attached`) + full-width accent pill `Analyze` → `analyzeClip(i)`.
  - Analyzed (**condensed report**): score row — `Overall` big `.num` 28px + `/100` 13px dim suffix, then three chips `Accuracy 82 · Placement 71 · Consistency 68` (bg `#EEEFF2`, `.num` values); feedback list — each row: severity `.chip` (`Good`/`Average`/`Poor`) + 13.5px/1.45 text; ghost `Re-analyze` button (transparent, 1px solid `#E2E3E7`, radius 999, 13px/600, `toast('Re-analysis is a lab stub')`).

Header above the list: `h2` `Analysis` + dim 13px right meta `4 clips · 2 analyzed` (computed from `CLIPS`).

- [ ] **Step 2: Wire accordion + fake analyze**

```js
let openClip = null;
function toggleClip(i){
  openClip = (openClip === i) ? null : i;
  renderClips();
}
function analyzeClip(i){
  const btn = document.querySelector(`#clip-${i} .analyzebtn`);
  btn.disabled = true;
  let pct = 0;
  const iv = setInterval(()=>{
    pct += 4 + Math.floor(Math.random()*7);
    if(pct >= 100){
      clearInterval(iv);
      const c = CLIPS.find(c=>c.id===i);
      c.analyzed = true; c.score = 76;
      c.scores = {Accuracy:74, Placement:70, Consistency:65};
      c.feedback = [
        {t:'Drives holding good width along the side wall.', s:'good'},
        {t:'Short of length in the final third of the rally — fatigue pattern.', s:'avg'}];
      renderClips();
      toast('Analysis complete — in the app this runs the real pipeline');
    } else {
      btn.textContent = `Analyzing… ${pct}%`;
    }
  }, 120);
}
```

Event delegation note: stop propagation on the Analyze/Re-analyze buttons so they don't toggle the accordion.

- [ ] **Step 3: Verify in the browser**

Accept when: 4 clip cards render; tapping expands one at a time with animation; unanalyzed clip shows accent Analyze; clicking it counts to 100% then flips to the condensed report; analyzed clips show Overall 84/100, three score chips, severity-chipped feedback rows; header meta updates to `3 analyzed` after the fake run; no console errors.

- [ ] **Step 4: Commit**

```bash
git add design-lab/crosscourt-lab.html
git commit -m "Design lab: analysis view with clip accordion and fake analyze"
```

---

### Task 4: Training view

**Files:**
- Modify: `design-lab/crosscourt-lab.html`

**Interfaces:**
- Consumes: Task 1 utilities.
- Produces: nothing downstream.

- [ ] **Step 1: Add markup**

Static HTML in `#view-training`:

1. `h2` `Training`.
2. **Program card**: `.hero` style reused (dark, radius 24). Left column: 15px/700 `Intermediate Program` + 13px rgba-white `Level 4 · Week 3 of 8`. Right: 44px progress ring — SVG circle, track `rgba(255,255,255,.18)`, arc `var(--accent)` at 75% (stroke-dasharray), centered 11px/700 white `75%`.
3. **Today Missions card**: `TODAY MISSIONS` 13px/700 dim header; 2 `.drillrow`s (reuse Task 2 classes): `Power drives · 60 min` and `Flick serve targets · 30 min`, each with `.startchip` → `toast('Mission start is a lab stub')`.
4. **Exercises card**: `EXERCISES` header; 3 rows, each: 48×48 thumbnail (gradient, radius 12) · column of 15px/600 name + meta line (difficulty `.chip` + 12px dim `2 min · 3 sets`) · right 40px accent circular play button (dark play triangle SVG, `toast('Player is a lab stub')`).
   - `Ghosting — six corners`, chip `Easy` (good tint), `2 min · 3 sets`
   - `Boast accuracy ladder`, chip `Medium` (avg tint), `5 min · 4 sets`
   - `Tin-margin pressure drill`, chip `Hard` (poor tint), `10 min · 4 sets`

- [ ] **Step 2: Verify in the browser**

Accept when: program ring reads 75% in accent on the dark card; missions match the drill vocabulary; difficulty chips use the three tint pairs; play buttons are perfect accent circles ≥40px; whole page scrolls smoothly under the nav pill; no console errors.

- [ ] **Step 3: Commit**

```bash
git add design-lab/crosscourt-lab.html
git commit -m "Design lab: training view"
```

---

### Task 5: Progress view — Whoop-style trends

**Files:**
- Modify: `design-lab/crosscourt-lab.html`

**Interfaces:**
- Consumes: Task 1 utilities.
- Produces: `splinePath(pts)` helper, `TRENDS` data, `renderProgress()`, `setRange(r)`.

- [ ] **Step 1: Add data + spline helper**

```js
// per metric, per range: array of session values (oldest → newest)
const TRENDS = {
  ranges:['1W','1M','3M','6M'],
  metrics:[
    {key:'score',   label:'Session score', unit:'', baseline:[70,80],
     data:{ '1W':[76,81,79,84], '1M':[68,74,71,76,73,79,81,79,84],
            '3M':[62,66,71,68,74,70,76,73,79,84], '6M':[55,58,63,61,66,71,68,74,79,84] }},
    {key:'inpct',   label:'Front-wall IN %', unit:'%', baseline:[68,78],
     data:{ '1W':[74,76,73,78], '1M':[66,70,68,73,71,74,76,73,78],
            '3M':[60,64,66,63,70,68,73,74,76,78], '6M':[54,58,60,64,66,70,68,73,76,78] }},
    {key:'minutes', label:'Active minutes', unit:'m', baseline:[25,45],
     data:{ '1W':[32,41,28,44], '1M':[22,30,26,35,32,41,28,38,44],
            '3M':[18,24,28,22,30,35,32,41,38,44], '6M':[15,18,24,28,30,35,32,41,38,44] }}
  ]
};

function splinePath(pts){ // pts: [{x,y}] — Catmull-Rom → cubic bezier
  if(pts.length < 2) return '';
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for(let i=0;i<pts.length-1;i++){
    const p0 = pts[Math.max(0,i-1)], p1 = pts[i], p2 = pts[i+1],
          p3 = pts[Math.min(pts.length-1,i+2)];
    const c1x = p1.x + (p2.x-p0.x)/6, c1y = p1.y + (p2.y-p0.y)/6;
    const c2x = p2.x - (p3.x-p1.x)/6, c2y = p2.y - (p3.y-p1.y)/6;
    d += ` C ${c1x} ${c1y}, ${c2x} ${c2y}, ${p2.x} ${p2.y}`;
  }
  return d;
}
```

- [ ] **Step 2: Add renderer + markup**

`#view-progress` contains: `h2` `Progress` · range segmented control · `#deltastrip` · `#trendcards`.

1. **Range selector**: 4 equal buttons in a white radius-999 container (`#rangeseg`), active = accent capsule, 13px/700. `setRange(r)` stores `currentRange`, re-runs `renderProgress()`.
2. **Delta strip**: 3 equal white mini-cards (grid, gap 8): 12px dim label + row of 20px `.num` value and delta chip (`▲3` good tint / `▲5` good tint / `—` neutral `#EEEFF2`). Labels: `IN%` `78`, `Score` `84`, `Rallies` `42`. (Static — "since last session" doesn't change with range.)
3. **Trend cards** — one per metric, built by `renderProgress()`:
   - Card top row: 13px dim label; below it 28px `.num` current value + unit, and beside it a 12px/700 delta vs baseline midpoint, tinted good if above, poor if below (`+9 vs typical`).
   - SVG 358×120, `viewBox="0 0 358 120"`, padding 8: 
     - **baseline band**: `<rect>` spanning the y-range of `metric.baseline`, fill `#EEEFF2`, radius 6, plus a dotted mid line `stroke:#D5D7DC; stroke-dasharray:2 4`.
     - **area fill**: spline path closed to the bottom, fill `rgba(217,242,79,.25)`.
     - **line**: `splinePath` stroke `#B7CC3F` (accent darkened for contrast on white), width 2.5, round caps.
     - **last point**: 5px circle fill `var(--accent)`, stroke `var(--ink)` 2px, with a small ink value label above it.
   - y-scale: min/max of data ∪ baseline, 10% padding; x evenly spaced.
4. **Personal best card**: trophy-ish inline SVG in an `.icontile` + 14px/600 `Best IN% to date — 84%` + 13px dim `Jul 22 · beat it to set a new mark`.

- [ ] **Step 3: Verify in the browser**

Accept when: switching `1W/1M/3M/6M` visibly re-renders all three charts with more/fewer points; lines are smooth (no polyline corners); baseline band + dotted line visible behind the line; latest point emphasized with accent dot + label; delta strip shows the three tiles; no chart overflows its card; no console errors.

- [ ] **Step 4: Commit**

```bash
git add design-lab/crosscourt-lab.html
git commit -m "Design lab: progress view with Whoop-style trend charts"
```

---

### Task 6: Full-pass verification and polish

**Files:**
- Modify: `design-lab/crosscourt-lab.html` (fixes only)

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1: Walk every interaction in the browser**

With the preview open: all 4 tabs; both dashboard toasts; Start chip → Training; expand each clip; run the fake analyze to completion; every Training stub; all 4 ranges on Progress. `read_console_messages` must be clean after the whole walk.

- [ ] **Step 2: Spec conformance sweep**

Re-read the spec's "Visual language" and "Screens" sections against the rendered result. Checklist: exact palette values used; radius 20–24 on cards; one very soft shadow; tabular numerals on all stats; squash vocabulary everywhere (grep the file for `tennis|smash|forehand` → must be zero hits); no external URLs (grep for `http` inside the file → only the `lang`/doctype-free zero hits expected, no `src=`/`href=` pointing off-disk); accent always carries dark ink.

- [ ] **Step 3: Screenshot evidence**

Screenshot each of the four tabs (Browser pane `computer {action:"screenshot"}`) and send them to the user with SendUserFile or inline as the completion proof.

- [ ] **Step 4: Fix anything found, re-verify, commit**

```bash
git add design-lab/crosscourt-lab.html
git commit -m "Design lab: full-pass polish"
```

---

## Self-review notes

- Spec coverage: scaffold/frame/no-CDN (T1), Dashboard (T2), Analysis + condensed report + fake pipeline handoff (T3), Training (T4), Progress Whoop charts + range selector + deltas + personal best (T5), success criteria & guardrails (T6). Dark theme, real data, DESIGN.md changes: correctly absent (out of scope).
- Names consistent: `switchTab`, `toast`, `CLIPS`, `renderClips`, `toggleClip`, `analyzeClip`, `TRENDS`, `splinePath`, `renderProgress`, `setRange`.
- No pytest tasks on purpose: `design-lab/` has no paired test file, so the PostToolUse hook won't fire; verification is the browser walk defined per task.
