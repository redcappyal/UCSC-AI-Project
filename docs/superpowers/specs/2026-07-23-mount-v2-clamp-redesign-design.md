# Fin mount v2/v3 — clamp redesign (2026-07-23)

> **v3 addendum (same day, supersedes the v2 decision summary below):** the team asked
> to *borrow only the mechanism* of the Printables clamp and fit the **old mast design**.
> v3 therefore keeps the entire v1 architecture — the mast STL is emitted geometrically
> identical (reuse the printed part), the saddle keeps its v1 slot position, and the
> wedges keep their v1 steep faces (90°−pitch; the phone plane lies parallel to the face
> as the cradle did) with the 2 mm-embedment fix and per-wedge rail placement. The phone
> holder is an original design: `phone_clamp_body` (fixed top hook that registers the
> phone's top edge, front dovetail rail, M12×2.5 nut boss, v1-cradle back boss) +
> `phone_jaw_carriage` (sliding bottom jaw, coin-slot capture of the screw's glued-on
> tip disc) + `phone_jaw_screw`. Jaw span 68–95 mm, thickness 8–11 mm. No third-party
> files; the vendored Printables STLs were removed from the repo (mechanism-only
> attribution note in hardware/README.md). Two v1 latent bugs fixed: the 0.05 mm rail
> unions AND a cross-axis misalignment of the wedge↔cradle peg bores (offset along the
> slope on one part, along the wall on the other). Optics: with the clamp stack ~9 mm
> thicker than the cradle and the mast pinned at 48 mm, the legacy h/s ≥ 4.92 ratio
> (derived for a 90 mm lens height) is not met (3.7/4.5); the binding *functional*
> requirement — spec §10 floor visibility ≤0.6 m of the back wall — is computed and
> asserted instead: 0.57 m (A) / 0.47 m (B). Court-plane margin 3 mm, asserted.
> The T-joint of v2 is no longer needed and was dropped.

Redesign of the printed fin mount after the first hardware test (2026-07-23), replacing
the custom phone cradle with the Printables "Tripod Mobile Phone Clamp V2" (Stamos,
CC BY-NC-SA — remix allowed; this is a non-commercial student project).

Written under autonomous operation: the field-test feedback in the request is treated as
the requirements review; implementation proceeds without a second approval gate.

## Field results driving the redesign

Worked (keep exactly):

- W1. Saddle ↔ mast dovetail + tapered peg — fits perfectly.
- W2. Both wedges ↔ mast (female slot on male rail + peg) — fits perfectly.

Failed / change requested:

- F1. Cradle could not be mounted (blocked by F2).
- F2. Male dovetails on the wedge inclined faces printed messed up.
  Root cause found in the generator: the rail/stop solids were sunk only **0.05 mm**
  into the wedge body before the union — near-coincident faces produce degenerate
  boolean output. (The joints that worked used deep, clean overlaps.)
- F3. Cradle has no phone-size adjustability.
- F4. Saddle is blocked at the end of the fin channel; the team will simply slide the
  saddle as far toward the court as possible and let screw friction hold position — the
  wall-stop bosses / gap-filler geometry (datum C) are unwanted.
- F5. The TPU swivel pad (pad-cap over ball tip) falls out when nothing is clamped — it
  isn't attached to the screw. Also: the assembled mount must be **one connected part**
  for transport.
- F6. Some peg holes arrived filled with non-support filament — holes must be modeled
  fully open (through wherever function allows) and verified open.
- F7. No non-printable parts anywhere (the stock clamp foot needs a metal 1/4-20 nut —
  its nut cavity is visible in the mesh — so the foot interface cannot be used as-is).

## Decision summary

**Stack: saddle → mast → wedge (A 40° / B 32°) → modified clamp lower ("clamp_lower_mount") → stock clamp upper/screws → phone.**

1. **Phone holding = the Printables clamp** (screw-driven jaws, slimmer/thicker jaw
   variants) — solves F3. Its upper part, knurled head, split screw and thread insert
   print unmodified from the original files (copied into `hardware/third_party/` with
   attribution). Only the **lower** part is remixed.
2. **Remixed lower part**: the tripod foot's underside (arca bevels + 1/4-20 nut cavity
   + tripod hole) is swallowed by a grafted solid block carrying a **T-slot**; the nut
   cavity and thread hole are filled (solves F7). The foot plate itself is untouched
   above the graft because it doubles as the lower jaw's backing.
3. **Wedge ↔ clamp joint = T-rail / T-slot + end stop + tapered peg** (not a dovetail):
   on the clamp's print orientation a dovetail flank would be a ~25°-from-horizontal
   ceiling (unprintable without support); a T-profile decomposes into vertical walls
   plus 3.2–3.5 mm designed bridges on both parts. Same slide-to-stop + peg UX as the
   joints that worked (W1/W2). All unions now use ≥2 mm embedment (fixes F2, F1).
4. **Wedge faces re-derived**: the clamp bar mounts perpendicular to the wedge face, so
   the camera axis runs along the face's **slope**, not its normal. Face inclination now
   equals the optical pitch directly (A: 40°, B: 32°) and the face descends **toward**
   the wall. (The v1 cradle lay flat on the face, so v1 used 90°−pitch.)
5. **Saddle simplified** (F4): wall-stop bosses, chamfers and wall TPU pads deleted;
   plain 130 mm channel; position on the fin is set by the user + screw friction.
   Datum C (repeatable setback) is consciously given up — per-session calibration
   absorbs it.
6. **Clamp screw tip** (F5): ball tip + snap cap deleted. The screw threads through the
   boss and a **Ø18 tip disc with an internal printed thread** screws onto its end
   (CA-glued = captive), carrying a glued TPU face pad. Everything in the assembly is
   now pinned/threaded/glued to something — one connected part.
7. **Peg holes** (F6): every peg hole is modeled through wherever a through hole is
   functionally allowed (the saddle-cap hole stays blind by design — it must never
   reach the glass channel) and the generator asserts each bore is open by point
   containment along the axis.
8. **Placement solved, not guessed**: mast height, mast position along the saddle, and
   each wedge's rail position along its face are solved in the generator against the
   spec §8 constraints — WSF court-plane rule (topmost leaning hardware stays ≥3 mm
   outboard of the wall's outer face, i.e. ≥15 mm from the court-side plane) and edge
   occlusion (lens height/setback ratio ≥ 4.92 over phone widths 68–90 mm). Result:
   the mast grows to ~105 mm and sits ~56 mm from the wall end of the saddle. Asserted
   at generation time; violating parameter edits fail the build.

## Parts (v2)

| Part | Change |
|---|---|
| saddle | wall bosses/stops/pads removed; slot repositioned to solved mast position; nut thread unchanged |
| mast | column 48 → solved ~105; wedge-peg hole moved to the new low side; holes opened |
| wedge A/B | face = pitch angle, descending toward wall; T-rail + stop + peg holes; robust unions |
| clamp_lower_mount | remixed Printables lower: graft block + T-slot + peg hole, cavities filled |
| clamp upper / threads / knurl head / screw halves | stock Printables files, unmodified |
| clamp_screw | tip disc thread instead of ball |
| screw_tip_disc | new: internal-thread Ø18 disc, TPU pad pocket |
| pin_x3 | unchanged pin, 3 used (saddle-mast, mast-wedge, wedge-clamp) |
| tpu_seat_strip, tpu_jaw_pad_x2 | unchanged |
| tpu_screw_pad | new: Ø15 disc for the tip |
| cradle, pad_cap, tpu_wall_pad, tpu_swivel_face, tpu_band, pin_x2 | **deleted** |

Retention-band note: the cradle's mandatory TPU bands are dropped — the screw-driven
jaw clamp retains the phone far more positively than the open cradle did. If ball
strikes ever dislodge a phone in practice, bands can be reintroduced around bar+phone.

## Verification (in-generator, must pass to emit STLs)

- All meshes watertight, positive volume, posed on the build plate.
- Every peg bore open (containment sampling along the axis).
- T-joint: posed clamp mesh does not intersect the wedge mesh (assembly clearance).
- Optics: court-plane margin and occlusion ratio asserted for both wedges over the
  phone-width range; results printed.
- Renders of the posed world assembly for visual review.
