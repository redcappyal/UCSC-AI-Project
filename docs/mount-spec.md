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
| C | Wall glass outer face | setback (X) |

Gravity seats A; the jaws close on B; a sprung stop butts C. Drop-on, one thumbscrew,
same pose every session. Target repeatability: ≤1 mm translation, ≤0.3° any axis.

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
[5] mic scoop (optional clip-on)
[4] phone cradle (TPU-lined pocket, lens corner up, court-side prow)
[3] tilt wedge — A: 40° (16:9) / B: 32° (4:3)   ← swappable, dovetail + lock screw
[2] mast (bolted + doweled to saddle)
[1] saddle (straddles fin top edge at the wall junction)
```

Separate parts so each prints in its ideal orientation and the wedge swaps without
touching registration. Saddle↔mast joint: 4× M5 into heat-set inserts + 2× Ø4 press-fit
dowel pins, asymmetric pattern (cannot assemble rotated).

## 7. Part specifications

### 7.1 Saddle (PETG)

- **Jaw channel**: width = T_f + 3.5 (12 → **15.5**); accepts 2 mm uncompressed TPU pads each
  side with 0.5 preload. Inner faces parallel ≤0.2. Channel length along fin: **130**.
  Jaw depth below top edge: **80** fixed jaw (wall side may shorten to clear F1/F2/F3 —
  model the patch-fitting notch parametrically).
- **Cap** (bridges the edge): 15 thick above the edge; underside carries a
  **2 × 12 × 110 TPU seat strip** in a 1.5-deep recess (datum A — glass edge never touches PETG).
- **Wall stop** (datum C): two bosses at the wall end, faced with 2 mm TPU, contacting the wall
  outer face; boss length = G + 2 (parametric); vertical spread ≥ 60 for a stable couple.
- **Clamp**: one M5 stainless thumbscrew, axis 45 below edge at channel mid-length, through
  clearance hole in the moving-side jaw into an M5 heat-set insert; tip carries a Ø18 swivel
  pressure pad faced with 2 mm TPU. Thumbwheel Ø≥28. Design clamp force ≤ ~50 N —
  finger-tight only; the screw holds position, the geometry holds pose.
- Edge fillets ≥2 everywhere near glass; no printed corner may touch a glass edge.

### 7.2 Mast (PETG)

- Bolts to saddle cap (M5 ×4 + dowels ×2). Column section ≥ 30 × 45, ribbed; rises at the
  wall end of the saddle so the cradle lands over the wall glass.
- Top face carries the **wedge dovetail**: 45° male dovetail, hard end-stop, one M5 side lock
  screw into insert. Dovetail axis parallel to the wall.

### 7.3 Tilt wedges (PETG) — print both

| Wedge | Optical axis pitch | Capture mode | Frame top hits | Engrave |
|---|---|---|---|---|
| A | **40° below horizontal** ±0.5° | ultrawide 16:9 (stock) | front-wall service line (1780) | `UW-16:9-40` |
| B | **32° below horizontal** ±0.5° | ultrawide 4:3 (full sensor, custom AVFoundation) | front-wall out line (4570) | `UW-4:3-32` |

Residual angle error is absorbed by calibration — repeatability matters, accuracy doesn't.

### 7.4 Phone cradle (PETG body, TPU liner)

- Pocket = phone+case envelope + 0.4 clearance per side; **1.5 TPU liner on every contact
  face** — the phone never touches PETG (grip + structure-borne vibration decoupling).
- Landscape, **ultrawide lens corner at the top, facing the court**.
- Corner window for the whole lens cluster + mic: no material within a **Ø40 clear zone**
  around the mic port (parameter M), none within the FOV frustum (§8).
- Retention: insertion from the top, retention lip + **silicone-band groove (mandatory —
  the wall top edge is in play, WSF 5.03, and the mount will take occasional ball strikes)**.
  No rigid latch over the top edge — it would sit in the FOV.
- **Prow**: low 45° chamfered ridge on the cradle's court side ahead of the phone's lower
  edge, below the FOV frustum, to shed direct ball strikes.
- **USB-C slot** 14 × 8 at the phone's lower edge + cable channel with strain-relief boss
  down the mast (match recordings need wall power).

### 7.5 Mic scoop (optional, TPU)

Clip-on horn around the mic port aimed down-court. Separate part, zero contact with the lens
window, removable for A/B testing. Any scoop change ⇒ audio threshold retune.

## 8. Optical placement (drives mast + wedge geometry)

| Parameter | Value | Derivation |
|---|---|---|
| Lens center height | **2220 AFF** (90 above top edge) | see setback coupling below |
| Lens center setback | **18 outboard of the court-side wall face** (= 6 outboard of the wall's outer face) | edge-occlusion + phone-lean limits |
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
| Saddle | PETG (ASA if the gallery gets hot sun) | channel opening up — jaw faces are vertical walls, parallel off the printer |
| Mast, wedges | PETG | wedge angled face up-facing at the printed angle for a clean seat |
| Cradle body | PETG | pocket opening up |
| Pads, liner, scoop | TPU 95A | glued (CA) into recesses |

0.2 layers, 4 perimeters, ≥40% gyroid on saddle + mast. **No PLA anywhere** — creep under
sustained clamp load is pose drift, which defeats the calibrate-once design.

BOM: M5 brass heat-set inserts ×7 · M5×25 SS thumbscrew ×1 (clamp) · M5×16 SS ×5
(mast bolts + wedge lock) · Ø4×10 SS dowel pins ×2 · Ø18 swivel pad ×1 · silicone band ×2.

## 10. Install & verification

Install: hook cap over fin edge at the junction → slide until wall stops touch → finger-tighten
thumbscrew → seat phone → band. Target < 30 s.

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
