# Archived DESIGN.md sections — the live-call and pairing components

**Archived 2026-07-27 with the two-camera stereo/peer feature. See
[../README.md](../README.md).**

These were DESIGN.md §8.17–§8.20. All four existed for the live two-camera
rally: the verdict flash and banner that `RecordView` rendered mid-rally, and
the link-status row and pair code that `p-pair` was built from. With the live
path archived, none of them has an implementation on either surface — web or
native — so they were removed from DESIGN.md rather than left there as
specification for components nobody can point at.

They are reproduced verbatim below. Restoring the live path means restoring
these to DESIGN.md §8 (renumbering as needed) in the same change.

Note: the web `#p-live` section in `index.html` is **not** related to these. It
is a "Coming soon" placeholder for live match as a product idea, predates the
stereo work, and was deliberately left in place.

---

### 8.17 Call flash (`.call-flash`)

The live-mode payoff §17.3 pre-authorises: scaled-up verdict-box grammar, not a new
visual language. A full-stage color wash occupying the stage overlay layer (§3.2's
z-index ladder, same tier as the §8.12 analyzing scrim) with a single centered uppercase
word — `IN` / `OUT` / `DOWN` / `NO CALL` — 28/700+, tracking `.04em` (the §6.1 verdict-word
spec, scaled up for full-stage legibility instead of the boxed 78 px well).

**Tokens:** `IN` fills `--in` (`Theme.inCall`); `OUT` and `DOWN` both fill `--outcall`
(`Theme.outCall`) — a tin fault ends the rally the same way an out ball does, so it takes
the same verdict color rather than a third hue; `NO CALL` fills `--mk-unknown` gray
(`Theme.unknown`) and never borrows green/red — a no-call is explicitly not a verdict
(§0.3).

**States:** idle (hidden; stage shows the live camera feed unobscured) → flash (word +
wash, the §10 **Flash** token — ≤ 500 ms, opacity/fill only — this is a flash, not a
progress state) → idle again, handing
off to the persistent `.call-banner` (§8.18) so the honest state stays legible after the
wash clears. When `prefers-reduced-motion` is set, show the end state directly with no
wash transition — a state change, not an animation.

**Layout:** stage overlay only, never inserted into `<main>`'s flow — appearing and
clearing it causes zero layout shift (§0.9), the same guarantee the analyzing scrim
gives.

**Touch targets:** none — display-only, like the analyzing scrim (§8.12).

### 8.18 Call banner (`.call-banner`)

A persistent one-line result under the stage, reserving its height at all times (the
§8.7 `.verdict{min-height:78px}` reserved-space pattern) so a call appearing or clearing
never shifts the layout beneath it. Format is a template, not a fixed list:
**`<verdict word> · <confidence phrase>`**. The verdict word is `IN` / `OUT` / `DOWN` —
the same three words `.call-flash` (§8.17) can show — each pairable with either
confidence phrase, `high confidence` or `one view`: `IN · high confidence`,
`IN · one view`, `OUT · high confidence`, `OUT · one view`, `DOWN · high confidence`,
`DOWN · one view` are all valid (6 combinations, 3 words × 2 tiers). `NO CALL` is the
exception: it never carries a confidence phrase, only one of two fixed reasons —
`NO CALL · obstructed` or `NO CALL · floor bounce`. This is the honest-states contract:
`called (high)` / `called (one-view)` / `no-call (obstructed)` / `no-call (floor
bounce)` are the only states that exist — never a guess rendered as a confident call —
and confidence is always carried by that text label, never by color alone (§0.3).

**Tokens:** the word takes the same color as `.call-flash` (`--in` / `--outcall` /
`--mk-unknown`); the confidence phrase is always `--dim` sentence case (§0.7) regardless
of the word's color, so it stays legible and calm under a red or green fill.

**States:** `.blank` (dashed `--line` border, dim — the §8.7 blank treatment) before any
rally has been called → filled, one of the 8 combinations above (6 verdict × confidence
+ 2 no-call reasons). A floor bounce is not a line verdict, so it renders `NO CALL ·
floor bounce`, never a colored call.

**Touch targets:** none — informational, like a `.status` line (§13).

### 8.19 Link status (`.link-status`)

A row reporting the pairing/link state as words, never as color alone. A naive green
"connected" dot would silently claim the verdict family (§0.3 reserves green/red for
IN/OUT), so this component is built so that mistake can't happen: an 8 px `--dim` fill
dot (decorative only, carries no meaning by itself), an 8 px gap (§4.5), then an explicit
sentence-case word — `Not paired` / `Looking for the other phone…` / `Codes match?` / `Syncing
clocks…` / `Paired · sync ±1.4 ms` / `Link lost — still recording` / the session's
failure reason, verbatim. The sync-quality figure is tabular numerals (§0.8) since it is
a number that updates.

**Tokens:** dot and text both `--dim`; text may step up to `--text` for emphasis (the
`.status.ok` pattern, §13) but never to `--in` / `--outcall` — link state is never a
verdict.

**States:** not-paired → searching → confirming → syncing → paired (ready) → degraded
(link lost, still recording — never a silent downgrade) → failed. See the `p-pair` state
table in §16 for the full mapping.

**Layout:** reserves its line height (§7's reserved-heights rule) so a state change never
shifts the pair-code or the primary button below it.

**Touch targets:** none — a status row, not a control.

### 8.20 Pair code (`.pair-code`)

A 4-digit confirmation code shown while pairing so both operators can visually confirm
the phones are pairing with each other and not a stray nearby device. Set large — 36–40
px, 700 weight, tracking `.04em`, tabular numerals (§0.8) so the digits never jitter —
legible from across a court at arm's length, the §1 glanceability principle applied to a
two-person distance instead of a one-handed one. Rendered on a `--surface` card, radius 8
(the §8.9 card fill), under a sentence-case "Codes match?" caption.

**Tokens:** `--text` digits on `--surface`; no accent, no verdict color — a pairing code
is neither an action nor a call.

**States:** `.blank` (dashed `--line` border, dim — the §8.7 blank treatment) before the
`confirming` step → filled with one code, shown for exactly one pairing attempt — a new
attempt gets a new code, never a silently reused one.

**Layout:** `.pair-code` sits inside the single continuous `p-pair` section, so it
appears and clears in place, not by navigating elsewhere — exactly the case §0.9
targets. It reserves the same card footprint (`--surface`, radius 8, the code's own
36–40 px line height plus the card's `12px 14px` padding, §4.5) in its blank state as in
its filled state, so `CONFIRM`/`START RALLY` and anything below it never jump when the
code appears or clears.

**Mismatch path:** alongside the code, a secondary text action reads "Codes don't
match" — styled as `button.small` (§8.1, 44 px, not a filled button) so it never
competes with the phase's one proxied primary (§7). Tapping it ends the session and
returns `p-pair` to Searching; this is the wrong-phone / stranger's-phone case, and it
must be reachable without waiting for a timeout. See the `p-pair` state table in §16 for
where this sits in the flow.

**Touch targets:** the code itself isn't tappable (confirming is the phase's one
proxied primary button, `CONFIRM`, 48 px, §3.4); "Codes don't match" is a real control
at 44 px (`button.small`, §8.1).
