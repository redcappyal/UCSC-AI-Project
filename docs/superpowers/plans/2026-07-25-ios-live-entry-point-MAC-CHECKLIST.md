# Mac verification checklist — iOS live-match entry point

**Why this file exists.** Task 11 of
[the plan](2026-07-25-ios-live-entry-point.md) is the verification pass. It was run on
the Windows dev box, where `xcodebuild`, `xcodegen`, `swift`, and `swiftc` do not exist
and this repo has no `.venv` — so **no Swift source on this branch has ever been
compiled, and no test in `ios/Tests/` has ever been executed.** The static checks that
*could* be done there were done (see `.superpowers/sdd/task-11-report.md`); everything
below is what remains, and none of it should be assumed to pass.

Work top to bottom. Steps 1–2 gate everything after them.

---

## 1. Build and run the Swift suite

```bash
cd ios && xcodegen generate
xcodebuild test -scheme SquashLineCalling \
  -destination 'platform=iOS Simulator,name=iPhone 15'
```

Expected: PASS, including the suites this branch added or extended —
`LiveSessionModelTests` (new, ~1500 lines), `RecordingOwnershipTests` (new),
`LiveCallTests` (new), `PeerSessionTests`, `PairingModelTests`,
`CaptureOrientationTests`, `StereoEngineTests`, `RunSubmissionTests`,
`StereoWiringTests`.

`ios/SquashLineCalling.xcodeproj` is generated and gitignored — never edit it; edit
`ios/project.yml`. `project.yml` globs `Sources` and `Tests`, so the new files need no
project change.

**Expect first-run compile errors and treat them as real findings, not chores.** Ten
tasks of Swift were written without a compiler. Fix them on the branch and note what
broke.

## 2. Confirm the goldens survived a real run

```bash
git status --porcelain tests/stereo_goldens.json ios/Tests/Fixtures/stereo_goldens.json
```

Expected: empty. Both files were verified byte-identical to `main` statically, but
`StereoGoldenTests` has never actually run — Task 3 extracted `StereoTrack.analyze`
out of `detectImpacts` as a pure move, and `StereoGoldenTests` passing unchanged is the
only evidence that no arithmetic moved with it. If a golden test fails, revert Task 3's
extraction and redo it as a pure move. **Never run `tests/generate_stereo_goldens.py`
to make it pass** — that rewrites the evidence instead of the bug.

## 3. Python suite (reporting completeness only)

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 348 passed. No Python file is touched by this branch (`git diff --stat
main...HEAD` lists none), so this is a formality — but the plan asks for the number and
the Windows box has no `.venv` to produce it.

## 4. `/verify` the three screens, both themes, 390 × 844

Use the `/verify` skill on `p-load`, `p-pair`, and `p-live` (§0.12). Confirm:

- **`p-load`** — two hero cards, not three; "Record a clip" is the single accent card,
  "Live match" is a surface card with **no `SOON` tag** (§8.15, §16's per-client note).
- **`p-pair` no layout shift** — the role segment is hidden with `.opacity`, so its
  footprint must survive pairing starting; the two-line link-status reservation must not
  shove the pair-code card down when the status text wraps; the pair-code's blank and
  filled states must occupy the identical footprint.
- **`p-live` no layout shift** — the mini-court appearing after a call must shift
  nothing (it reserves `MiniCourtView.reservedHeight` in both branches), and the STOP
  button's footprint must be reserved on the secondary before a degraded link grants it
  one.
- **Both themes.** Note honestly: `ios/Sources/Theme.swift` is dark-only — there are no
  light-theme values in the native client at all, so "both themes" cannot presently be
  satisfied natively. Either confirm that and record it as a documented deviation in
  DESIGN.md §3.2's native-shell note, or add light values. This is pre-existing (Theme
  has not changed since the iOS scaffold) but it is what blocks §0.12 on native.

**Also re-check by hand, on device or simulator, at 390 × 844:** the "Codes don't
match" secondary action on `p-pair`. See the finding in
`.superpowers/sdd/task-11-report.md` — its `.frame(minHeight: 44)` is applied outside
the `Button`, which the codebase's own comment in `LiveStageView.stop` says leaves the
hit region at the label's intrinsic ~20 pt height. Tap near the top and bottom edges of
the row and confirm whether it fires; if it does not, that is a §0.6 violation on the
stranger's-phone escape hatch.

## 5. Drive the DEBUG stereo demo end to end

On the record stage (`p-record`), tap the cube button (top-trailing, under where the
peer-bench button sits one screen up).

Expected: `IN · high confidence` in the call banner, with the call flash preceding it.
This is the check that Task 3's event-shape change (`StereoEvent.impact` now carries
`track:`) did not break the engine → `CallPresentation` → flash/banner path. It runs on
one device with no peer and no ball model, so it is the cheapest end-to-end signal
available before hardware.

Then confirm the mini-court: `p-live`'s replay only draws when `RecordModel.liveTrack`
is non-empty, which is now populated from the same event. The DEBUG demo sets it too
(`RecordModel.startStereoDemo`), so a non-empty track there is the first proof Task 3
actually delivered the geometry the replay needs.

## 6. Verify plain single-camera recording still works

Task 9 changed `RecordView` from owning its `RecordModel` (`@StateObject`) to receiving
one (`@ObservedObject`) from `PlayRootView`. That is the one file the shipping
single-camera path depends on, and the whole ownership design exists to keep pairing
from being able to break it.

Record → stop → `ResultsView` → upload, with **no pairing anywhere in the session**.
Then confirm the opposite direction: while a live rally is running, `p-record`'s record
button is replaced by "The live match is using this camera — end the rally there
first." and no tap there can touch the recording.

## 7. Two-phone, on court (the part only hardware can answer)

Follow `ios/PEER.md`'s "Two-phone bring-up" runbook, which was rewritten for this
change. The specific things this branch introduced and nothing has ever exercised over
a real radio:

- [ ] The **role picker** gates PAIR on both phones, and one "calls" + one "assists"
      actually pairs (two "calls" must fail to find each other — check that too).
- [ ] The **calibration gate**: a phone with no usable calibration shows the server's
      reason verbatim and cannot PAIR.
- [ ] The **calibration exchange**: the secondary's model reaches the primary, builds
      its `StereoEngine`, and START RALLY lights on the calling phone only. Confirm the
      "Paired · waiting for the other phone's calibration" state is actually transient
      and not the resting state.
- [ ] A **rally end to end**: START RALLY on the primary starts the secondary with no
      tap on it; STOP on the primary stops both.
- [ ] **Degraded link mid-rally**: walk a phone out of range. Recording must continue,
      and the secondary must gain its own STOP.
- [ ] **Paired upload**: both clips land with the same `session_id`, roles "a" and "b",
      and the server produces a `stereo-<session_id>` fuse run
      (`job_runner.try_start_stereo_fuse`).
- [ ] **More than one rally per pairing**: run three. Each must mint its own
      `session_id` (`<pairingID>-r<n>`) and fuse independently — one id per pairing was
      a real bug, fixed in `f6d3a4e`, and only hardware proves the fix.
- [ ] The PEER.md items that were already owed: the Phase 1 transport bench table, the
      Phase 2 ≤ 2 ms sync assertion, a real `BallDetector.mlpackage`.

---

## Findings this checklist inherits

From `.superpowers/sdd/task-11-report.md` — carry these into the Mac pass rather than
rediscovering them:

1. **`PairingView.swift:128–131`** — "Codes don't match" applies `.frame(minHeight: 44)`
   outside its `Button`. Likely §0.6 violation; step 4 above is how to confirm it. Not
   fixed in Task 11 (verification-only pass).
2. **`Theme.swift` is dark-only** — blocks §0.12's "both themes" on native. Pre-existing
   and undocumented in DESIGN.md.
3. **Native font is `.font(.system(...))`, not Chakra Petch** (§0.5), and native icons
   are SF Symbols, not §9's inline SVG. Both pre-existing and app-wide; neither is
   recorded as a deviation in DESIGN.md §3.2's native-shell note.
4. **DESIGN.md drift** — see the report's item 4 for the four small ones (role-segment
   position in §16's `p-pair` body cell, `p-load`'s "PLAY" heading / dev row, the
   Degraded row's "— (unaffected)" primary, and the `.link-status` 9 pt gap being off
   §4.5's spacing scale).
