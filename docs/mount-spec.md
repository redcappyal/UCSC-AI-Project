# Fin mount — 3D modeling specification

Camera/mic mount for the squash recording iPhone. Clamps a structural glass fin behind the ASB
glass back wall; phone periscopes over the wall top edge on the 0.5× ultrawide. Goals, in order:
(1) repeatable pose (calibrate once), (2) vibration isolation for the mic, (3) zero protrusion
into the court volume, (4) zero risk to the tempered glass.

All dimensions mm. AFF = above finished (court) floor.

## 1. Confirmed site facts (ASB wall, from photo + manufacturer specs)

| Item | Value |
|---|---|
| Wall / fin / door glass thickness | 12 (tempered — no point loads, no hard edge contact) |
| Wall top edge height | 2130 AFF |
| Fin depth (outward from wall) | ~300 |
| Fin height | full wall height — top edge at 2130, coplanar with wall top |
| Fin anchoring | steel base plate on floor + black patch fittings to wall panel |
| Door | full height (2130 × 750–900), hinged on the center fins |
| Wall attachment | fins joined to wall via patch plates — expect a small fin-to-wall gap |

## 2. Governing rules & court constants (WSF Specifications for Squash Courts, 2013/17 ed.)

### 2.1 Rules that bind the mount design

- **WSF 3.01**: play may be filmed "from above the court or through any of the clear walls" —
  the over-the-edge camera position is explicitly sanctioned.
- **WSF 3.02**: "No camera or other equipment may project into any part of the court or below
  the minimum clear height" (clear height falls to 4000 AFF at the back-wall end, WSF 4.04).
  ⇒ **Hard requirement**: every part of the assembly, at worst-case tilt, stays behind the
  court-side wall face plane. This is constraint 2 in §8 — verify it in CAD, not by eye.
- **WSF 5.03**: on a transparent back wall exactly 2130 high the back wall line is omitted —
  **the wall top edge itself is the out boundary and is in play**. Balls legitimately strike at
  and above the edge, so the mount *will* take occasional direct hits: the retention band is
  mandatory, and the cradle gets a low prow on the court side ahead of the phone's lower edge.
- **WSF 7.01/7.02**: the door sits in the middle third and **opens into the court** — its swing
  can never reach a mount on the fin outside, and the flush-surface requirement confirms
  nothing may be added court-side.

### 2.2 Court geometry constants (also the pipeline's reference numbers)

Measured to finished playing surfaces at 1000 AFF; all heights to the **underside** of markings;
markings are 50 wide and are out; line positions accurate to ±2 (WSF 5.01).

| Feature | Nominal | Tolerance |
|---|---|---|
| Court length | 9750 | ±10 |
| Court width (singles) | 6400 | ±10 |
| Diagonals | 11665 | ±20 |
| Front wall line (out) | 4570 | — |
| Service line (front wall) | 1780 | — |
| Tin | 480 | — |
| Back wall out boundary | 2130 (= glass top edge, line omitted) | — |
| Short line, nearest edge from back wall | 4260 | — |
| Service boxes | 1600 square, internal | — |
| Half-court line | equidistant from side walls, short line → back wall | — |

Implication for the pipeline: courts legally differ by up to ±10 mm and line heights are datumed
to the marking's underside — treat WSF nominals as priors and the per-court calibrated
homography as truth. The underside of each line is the precise calibration feature.

## 3. Fin selection

- **Primary: the fin immediately RIGHT of the door (viewed from outside the court).**
  ~450 mm off court centerline — best horizontal coverage of the far back corner. Door slams
  ring this fin, but doors slam between rallies, not during play.
- **Fallback: the outer right fin** (panel joint, ~1.8 m off center) if the door hinge / patch
  hardware fouls the saddle. Same part fits both; only the calibration profile differs.
- Always the same fin on the same court. Mark it (tape dot). One homography + distortion
  profile per court, keyed by court number.

## 4. Datum scheme (how the mount self-registers, no adjustments)

| Datum | Surface | DOF constrained |
|---|---|---|
| A | Fin top edge (horizontal, 2130 AFF) | height (Z), pitch |
| B | Fin faces (both sides of 12 mm glass) | lateral (Y), yaw, roll |
| C | ~~Wall glass outer face~~ **dropped in v2** (2026-07-23 field decision) | setback (X) by hand |

Gravity seats A; the jaws close on B. **v2**: datum C (wall stop bosses) was removed at
the team's request — the saddle is slid as far toward the court as it goes and screw
friction holds setback. Setback repeatability is consciously traded away; per-court
calibration (and re-calibration after remounting) absorbs it.

## 5. Measure on site before finalizing CAD (parameters)

| Sym | Measurement | Expected | Used in |
|---|---|---|---|
| T_f | Fin thickness incl. any film (calipers) | 12.0 ±0.3 | jaw channel |
| G | Fin-to-wall gap at the junction | 2–8 | wall-stop boss length |
| F1 | Top patch fitting: distance below top edge | ? | jaw notch |
| F2 | Top patch fitting: extent outward from wall face | ? | jaw notch / saddle position |
| F3 | Patch fitting plate thickness (standoff off fin face) | ~5–10 | jaw notch depth |
| E_t | Wall top edge detail (bare polished glass vs. alu cap; chamfer size) | bare | seat profile |
| P_w×P_h×P_t | Phone + case envelope | model-specific | cradle pocket |
| E | Ultrawide lens center → nearest long edge of phone+case | ~22 | lens setback check |
| M | Mic port position relative to lens center | model-specific | mic keep-out |

## 6. Assembly overview

```
[4] mic scoop (optional clip-on)
[3] Printables "Tripod Mobile Phone Clamp V2" (screw-driven jaws, any phone;
    phone landscape, lens corner up) — its LOWER part remixed with a T-slot block
    ↕ T-rail on the wedge face + end stop + tapered pin
[2] tilt wedge — A: 40° (16:9) / B: 32° (4:3)   ← swappable
    ↕ dovetail on the mast top + tapered pin
[1b] mast — dovetail into the saddle cap + tapered pin
[1a] saddle (straddles the fin top edge at the wall junction)
```

**Fully printed — no purchased hardware** (iteration-1 decision: durability
traded for zero-BOM). Every joint is a printed dovetail (0.4 clearance,
end-stop, Ø5 tapered retaining pin); the clamp is a printed coarse-thread
thumbscrew. Saddle and mast are separate parts joined by the same dovetail —
a merged part has no support-free print orientation. Relaxed fits throughout — the
registration datums (A/B/C) still come from glass contact, so pose
repeatability survives sloppy part-to-part fits.

## 7. Part specifications

### 7.1 Saddle (PETG)

- **Jaw channel**: width = T_f + 3.5 (12 → **15.5**) between PETG faces; **4 mm TPU 68A pads
  in 2.5-deep recesses each side (1.5 proud)** → light slip fit over the glass (~0.25/side),
  the clamp screw closes the gap; PETG stays ≥1.75 off the glass at all times. 68A note: at
  these pad areas flat-pad stiffness is set by area, not durometer — pose stiffness is
  unaffected; soft pads squirm in shear, hence the deeper pockets (walls carry shear) and
  thicker sections. All TPU elements are pocketed. Inner faces parallel ≤0.2.
  Channel length along fin: **130**.
  Jaw depth below top edge: **80** fixed jaw (wall side may shorten to clear F1/F2/F3 —
  model the patch-fitting notch parametrically).
- **Cap** (bridges the edge): 15 thick above the edge; underside carries a
  **3 × 12 × 110 TPU 68A seat strip** in a 2.5-deep recess, 0.5 proud (datum A — glass edge
  never touches PETG).
- **Wall stop: deleted in v2** (field decision 2026-07-23 — see §4 datum C).
- **Clamp (fully printed)**: coarse printed thread — **M12 × 2.5 trapezoid-profile
  thumbscrew** (PETG, integrated Ø32 notched knob, thread runs to the tip) threading into a
  printed nut thread in the outer-jaw boss (0.4 radial clearance). v2 tip: a **Ø18 tip
  disc with an internal printed thread** screws onto the protruding end inside the channel
  and is CA-glued (captive — the v1 snap-on pad cap fell out when unclamped); a 2.5 mm
  68A face pad glues into its pocket. The disc rotates with the screw against the glass —
  acceptable at finger-tight loads on TPU. Advancing the screw presses the glass onto the
  fixed jaw's bonded pad (datum B). Snug finger-tight only (≤ ~50 N).
- Edge fillets ≥2 everywhere near glass; no printed corner may touch a glass edge.

### 7.2 Mast (PETG, joined to the saddle by dovetail + pin)

- Column 30 × 45. v2: height and position along the saddle are **solved by the
  generator** against §8 (currently ~160 mm tall, base ~50 mm from the wall end) — the
  clamp bar holds the lens farther from the wall than the v1 cradle did, so the lens must
  sit higher to keep the occlusion ratio. Base carries a male rail underneath that slides
  into a slot in the cap top (pinned); lies flat on its side for printing.
- Top face carries the **wedge dovetail**: male rail (root 20, crest 26, height 6, ~63°
  flanks — prints support-free, embedded 2 mm for robust booleans), hard end-stop, axis
  parallel to the wall. Wedge is retained by a Ø5 printed tapered pin dropped through the
  wedge into the column beside the rail's low end (friction fit).

### 7.3 Tilt wedges (PETG) — print both

| Wedge | Optical axis pitch | Capture mode | Frame top hits | Engrave |
|---|---|---|---|---|
| A | **40° below horizontal** ±0.5° | ultrawide 16:9 (stock) | front-wall service line (1780) | `UW-16:9-40` |
| B | **32° below horizontal** ±0.5° | ultrawide 4:3 (full sensor, custom AVFoundation) | front-wall out line (4570) | `UW-4:3-32` |

Residual angle error is absorbed by calibration — repeatability matters, accuracy doesn't.
v2 geometry: the clamp bar mounts **perpendicular** to the inclined face, so the camera
axis runs along the face's slope — **face inclination = optical pitch directly** and the
face descends toward the wall (v1's cradle lay flat on the face, so v1 used 90°−pitch).
Each wedge: female dovetail underneath (mast rail + 0.4 clearance), and a **T-rail +
end-stop on the inclined face** (stem 14×3.3, head 26×3.0, 2 mm embedment) for the clamp
block — T instead of dovetail because on the clamp's print orientation a dovetail flank
would be an unsupported ~25° ceiling; the T decomposes into vertical walls + 3–4 mm
designed bridges on both parts. Same slide-to-stop + tapered-pin retention. The rail's
position along the face is per-wedge, solved by the generator against §8.

### 7.4 Phone holder (v2: Printables "Tripod Mobile Phone Clamp V2", remixed lower)

v1's custom cradle is replaced by the Stamos clamp (CC BY-NC-SA, files vendored in
`hardware/third_party/`): screw-driven sprung jaws fit any phone 68–95 mm wide —
the field-reported adjustability gap. Upper jaw, thread insert, knurled wheel and split
screw print unmodified. The **lower part is remixed** (`clamp_lower_mount.stl`):

- The tripod foot's underside — arca bevels, 1/4-20 hole and the **metal-nut cavity**
  (stock clamp needs a non-printable nut) — is swallowed by a grafted solid block; the
  foot plate above it is untouched (it is the lower jaw's backing).
- The block carries the **T-slot** (mouth on the underside face, through both ends),
  a peg bore beside the bar, and seats mouth-face-flat on the wedge face.
- Phone mounts **landscape, ultrawide lens corner up, back facing the court**; bar is
  vertical-leaning-over-the-wall (⊥ the wedge face). Phone center lands ~8 mm off the
  fin midplane (within the ±20 budget).
- Retention: the screw-driven jaws replace the v1 bands (a clamped jaw is far more
  strike-proof than the open cradle + bands; reintroduce bands around bar+phone only if
  strikes ever dislodge a phone in practice).
- USB-C: both phone short edges are clear of the clamp — cable routes freely; no
  dedicated channel in v2.

### 7.5 Mic scoop (optional, TPU)

Clip-on horn around the mic port aimed down-court. Separate part, zero contact with the lens
window, removable for A/B testing. Any scoop change ⇒ audio threshold retune.

## 8. Optical placement (drives mast + wedge geometry)

| Parameter | Value | Derivation |
|---|---|---|
| Lens center height | v1: 2220 AFF · **v2: ~2395 AFF (~265 above edge), solved** | see setback coupling below |
| Lens center setback | v1: 18 · **v2: ~54 outboard of the court-side face (clamp-bar bulk), solved** | edge-occlusion + phone-lean limits |
| Optical axis yaw | 0° (⊥ back wall) ±0.5° | datum B |
| Optical axis pitch | wedge A/B (§7.3) | |
| Cradle lateral position | phone center within ±20 of the fin midplane | mass over the clamp |

Governing constraints (re-verify in CAD if any input changes):
1. **Edge occlusion**: bottom of frame is ~76.5° below horizontal (both modes).
   Require `h / s ≥ tan 78.5° = 4.92` (2° margin), h = lens height above edge,
   s = lens setback from the court-side face. 90 / 18 = 5.0 ✓
2. **Court-plane clearance (WSF 3.02 — mandatory)**: at 40° tilt the phone's upper long edge
   leans court-side of the lens by `E·sin 40°` (≈ 12.9 for E = 22). Require
   `s − E·sin(tilt) ≥ 3`. 18 − 12.9 = 5.1 ✓ — **no part of the assembly may cross the
   court-side face plane.** Check the prow, band, and cable too.
3. **FOV keep-out frustum** from the lens center: ±55° horizontal, +15° to −78° vertical
   (covers both wedges). Nothing — cradle, band, prow, scoop, cable — inside it.

## 9. Materials, hardware, print settings

| Part | Material | Orientation / notes |
|---|---|---|
| Saddle | PETG (ASA if the gallery gets hot sun) | upside-down (cap on plate, jaws opening up) — datum pockets print crisp; teardrop boss keeps it support-free |
| Mast, wedges | PETG | lying on the side — every feature is a Y-extrusion, support-free |
| clamp_lower_mount | PETG | stock clamp orientation (flat) — T-slot ceilings are 3–4 mm designed bridges, support-free |
| Clamp upper/threads/knurl/screws | PETG | per the Printables author (threads vertical) |
| Saddle screw, tip disc, pins | PETG | screw vertical knob down; disc face down |
| Seat strip, jaw pads, screw pad, scoop | **TPU 68A** | print flat; glued (CA) into recesses |

0.2 layers, 4 perimeters, ≥40% gyroid on saddle+mast. **No PLA anywhere** — creep under
sustained clamp load is pose drift, which defeats the calibrate-once design.
**STLs are exported already posed for printing** — no rotation in the slicer.

**BOM: none.** Fully printed — printed M12×2.5 clamp thread, printed dovetail/T joints +
tapered pins, printed clamp (the stock clamp foot's 1/4-20 metal nut is designed out by
the graft block). Durability is explicitly traded away; reprint worn parts.

## 10. Install & verification

Install: hook cap over fin edge at the junction → slide as far toward the court as it
goes → finger-tighten thumbscrew → clamp phone with the knurled wheel. Target < 30 s.
(Setback is no longer datumed — re-run the calibration wizard after each remount.)

Acceptance tests before trusting data:
1. **Repeatability**: 10× full remount; court-line homography corner residuals ≤ 2 px
   (calibrate against marking **undersides** — the WSF datum, §2.2).
2. **Ball strike / door slam**: record while striking the wall near the top edge and slamming
   the door; verify no frame shift > 1 px and that structure-borne transients don't trigger
   the classifier.
3. **FOV**: both wedges — floor visible to ≤ 0.6 m of the back wall; frame top at service line
   (A) / out line (B); wall edge not in frame.
4. Then: one-time ultrawide distortion calibration → homography → **audio threshold retune**
   (new mic position invalidates current levels) → re-run the ground-truth eval.

## 11. Open items

- [ ] Site measurements table (§5)
- [ ] Confirm door-fin patch/hinge hardware clears the saddle, else fall back to outer fin
- [ ] Phone model + case → cradle pocket, E, M
- [ ] 4:3 full-sensor capture path in the pipeline (custom AVCaptureSession) before printing wedge B matters
- [x] Cross-check `court_model.py` constants against §2.2 (underside datum, 50 mm line width) —
  done 2026-07-20: all ft constants land on the WSF datum edges within ~4 mm (inside the
  ±10 mm build tolerance); datum convention documented in `court_model.py`. Follow-up same
  day: wizard snap landmarks re-datumed to paint CENTERLINES (`LINE_HALF_WIDTH_FT` shift),
  because the snap refiner's RANSAC+PCA fit lands mid-stripe — the edge datum left every
  snapped landmark ~25 mm biased (~35 mm diagonal at box corners), vs ~0.26 in/px at the
  short line. Zone/judge constants stay WSF edge datums; labels now say "middle of the
  line's width" so the no-snap tap fallback matches.
