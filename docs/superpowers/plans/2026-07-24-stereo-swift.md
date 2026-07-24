# Stereo Swift Mirror + Live Engine (Plan B2: spec Phase 3, part 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Swift half of spec Phase 3: `CameraModel`/`StereoMath`/`StereoTrack` ported
from the Python authority with golden-vector parity tests, solved-model exchange over the
existing peer layer, and the live `StereoEngine` that fuses local + remote detections
into 3D impacts with line calls on the primary phone.

**Architecture:** Python (`stereo_engine.py`, merged to main @ cae776c) remains the
authority; Swift mirrors it EXACTLY — including documented quirks — verified against
extended golden vectors. Task 1 extends the goldens (Python) to cover every confidence
tier and embeds the resample timeline per trajectory case, eliminating cross-language
`np.arange` ambiguity. The live engine is owner-pumped (like `PeerSession.tick`) on its
own serial queue.

**Tech Stack:** Swift 5.9 / iOS 17 (simd, Foundation), XCTest with the
`ios/Tests/Fixtures/stereo_goldens.json` resource; Python 3 + numpy for Task 1 only.

## Global Constraints

- **Mirror-exactly rule:** Swift ports replicate `stereo_engine.py` behavior including
  known quirks (the locality guard's inertness when the median window gap is 0; greedy
  landmark matching) — parity beats polish; every such spot carries a comment
  `// mirrors stereo_engine.py:<function> — do not "fix" without changing the authority`.
- **Numerics parity contract:** algebraic cases (triangulation/snap/call/project) match
  goldens within **1e-7 ft / 1e-7 px**; trajectory impacts match within **t: 1e-6 s
  (timelines are embedded, so grid-exact), point: 1e-3 ft**, and call/surface/confidence
  **exactly**. Quadratic fits use centered-time normal equations (3×3 solve) — valid
  because windows are ≤ 7 samples and centered (documented in code).
- Court frame: FEET, origin front-left floor seam, x right 0→21, y front 0→back 32, z up.
- Undistortion (division model, the exact JS↔Python↔Swift contract, from
  `court_model.undistort_point`): `p_u = c + (p_d − c) / (1 + k1·r²)`, `r = |p_d − c| / norm_px`;
  identity when distortion is null; factor `|1 + k1·r²| ≤ 1e-9` is an error.
  **Raw pixels in, undistort internally** — same boundary as Python (`triangulate`,
  `snapToPlane` undistort; `CameraModel.project` returns undistorted px, `ray` expects them).
- Surfaces are an ORDERED array mirroring the Python tuple:
  `["floor", "front_wall", "back_wall", "left_wall", "right_wall"]`; impact sort key is
  `(t_s, surface)`.
- Constants (verbatim from stereo_engine.py): PARALLEL_EPS 1e-9; OUT_LINE 15.0;
  TIN_TOP 19/12; BACK_WALL_OUT 7.0; COURT_WIDTH 21.0; COURT_LENGTH 32.0; bounds slack
  0.5; FIT_WINDOW_SAMPLES 7; MIN_FIT_SAMPLES 4; window gap ratio max 3.0;
  SNAP_DISAGREEMENT_MAX 0.3; IMPACT_PROXIMITY 1.5; merge 0.060 s; pre-impact window
  0.25 s / guard 1/240 s.
- iOS 17.0 / Swift 5.9; new Swift files in `ios/Sources/Stereo/`; tests in `ios/Tests/`;
  never `Date()` in stereo/peer timing code.
- Test commands: Python `pytest tests/ -q` (Task 1); Swift
  `cd ios && xcodegen generate && xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0.1'`
  (append `-only-testing:` for focused runs). Baselines: Python 216, Swift 54.
- Fixture loading in Swift tests:
  `Bundle(for: Self.self).url(forResource: "stereo_goldens", withExtension: "json")` —
  xcodegen already includes `ios/Tests/Fixtures/*.json` as test-target resources via the
  `Tests` source folder; if the resource is missing at runtime, add
  `resources: [Tests/Fixtures]`-style handling to project.yml rather than copying files.
- Server/Python changes beyond Task 1: none. Peer-layer changes limited to the one
  `onCalibration` seam (Task 5).
- Commit after every task with trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Branch: `claude/stereo-swift` (created off main @ cae776c).

## File Structure

- `tests/generate_stereo_goldens.py` + both `stereo_goldens.json` copies (Task 1, modify): goldens v2.
- `stereo_engine.py` (Task 1, small additive change): `build_track3d(..., timeline_s=None)`.
- `ios/Sources/Stereo/CameraModel.swift` (Task 2): decode + undistort + project + ray.
- `ios/Sources/Stereo/StereoMath.swift` (Task 3): triangulate, surfaces, snap, calls.
- `ios/Sources/Stereo/StereoTrack.swift` (Task 4): samples, fits, track, impacts.
- `ios/Sources/Stereo/StereoEngine.swift` (Task 6): live engine.
- `ios/Sources/API/APIClient.swift` + `ios/Sources/Peer/PeerSession.swift` (Task 5, small).
- `ios/Sources/Record/RecordModel.swift` + `ios/Sources/Peer/PeerBenchView.swift` (Task 7).
- Tests: `ios/Tests/StereoGoldenTests.swift` (Tasks 2–4 grow it), `ios/Tests/StereoEngineTests.swift` (Task 6), `ios/Tests/StereoWiringTests.swift` (Tasks 5, 7).

---

### Task 1: Goldens v2 + numerics parity contract (Python)

**Files:**
- Modify: `tests/generate_stereo_goldens.py`, `stereo_engine.py` (additive param), `tests/test_stereo_goldens.py`
- Regenerate: `tests/stereo_goldens.json`, `ios/Tests/Fixtures/stereo_goldens.json`

**Interfaces:**
- Consumes: everything in `stereo_engine.py`; `simulate_front_wall_shot`/`sample_camera` from `tests/test_stereo_track.py`.
- Produces (Tasks 2–4 depend on this exact schema):
  - `stereo_engine.build_track3d(model_a, samples_a, model_b, samples_b, hz=120.0, timeline_s=None)` — additive optional param: when given, iterate that exact list instead of `np.arange` (behavior unchanged when None).
  - Golden schema v2: top level gains `"schema": "stereo-goldens-v2"`; `trajectory` (singular) is REPLACED by `"trajectories"`: a list of cases
    `{"name", "samples_a", "samples_b", "timeline_s": [...], "impacts": [...]}` with
    impact entries as before plus `"snap_disagreement_ft": float|null`. Cases:
    1. `"clean"` — the existing shot (1 impact: front_wall/in/high).
    2. `"occluded_one_view"` — camera B's samples dropped in `[t_true−0.3, t_true+0.1]` (1 impact: front_wall, one_view).
    3. `"no_call"` — BOTH cameras' samples dropped in `[t_true−0.3, t_true−0.005]` (pre-impact windows empty on both sides). Record whatever `detect_impacts` produces (expected: front_wall impact with confidence `no_call`, point from the raw track minimum — if the generator observes something different, that IS the authority behavior; record it).
    4. `"pair_agreement"` — separate top-level key, not a trajectory: `{"obs_lattice": [{"court_ft", "px_a", "px_b"}], "good": {report...}, "biased": {"bias_ft": [0,0,0.5], report...}}` where obs are projections through the true models and `biased` evaluates model_b shifted by bias_ft (mirrors `test_pair_agreement_biased_model_fails_gate`).
  - `timeline_s` per trajectory = the exact `np.arange(t_lo, t_hi, 1/120)` list the generator's own `build_track3d` used (embedded so Swift never re-derives it).

- [ ] **Step 1: Add the `timeline_s` param (TDD)**

Append to `tests/test_stereo_track.py`:

```python
def test_build_track3d_accepts_explicit_timeline():
    left, right = make_fin_pair()
    states, _ = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    default_track = stereo_engine.build_track3d(left, samples_a, right, samples_b)
    timeline = [p.t_s for p in default_track]
    replayed = stereo_engine.build_track3d(left, samples_a, right, samples_b,
                                           timeline_s=timeline)
    assert len(replayed) == len(default_track)
    for a, b in zip(replayed, default_track):
        assert a.t_s == b.t_s
        assert np.allclose(a.point_ft, b.point_ft, atol=0.0)
```

Run: `pytest tests/test_stereo_track.py -q` → FAIL (unexpected keyword `timeline_s`).

Implement in `stereo_engine.py` — change `build_track3d`'s timeline construction to:

```python
def build_track3d(model_a, samples_a, model_b, samples_b, hz=120.0, timeline_s=None):
    if not samples_a or not samples_b:
        return []
    if timeline_s is None:
        t_lo = max(samples_a[0].t_s, samples_b[0].t_s)
        t_hi = min(samples_a[-1].t_s, samples_b[-1].t_s)
        timeline_s = np.arange(t_lo, t_hi, 1.0 / hz)
    track = []
    for t_s in timeline_s:
        ...  # existing loop body unchanged from here
```

(The `...` marks the EXISTING loop body carried verbatim — not new code.) Note the
subtle contract kept intact: the timeline default is identical to today, so all
existing tests and the v1 golden behavior are unchanged. Run the new test → PASS.

- [ ] **Step 2: Extend the generator to v2**

Rewrite the trajectory section of `tests/generate_stereo_goldens.py` (triangulation/
snap/call sections unchanged):

```python
    def trajectory_case(name, samples_a, samples_b):
        t_lo = max(samples_a[0].t_s, samples_b[0].t_s)
        t_hi = min(samples_a[-1].t_s, samples_b[-1].t_s)
        timeline = [float(t) for t in np.arange(t_lo, t_hi, 1.0 / 120.0)]
        impacts = stereo_engine.detect_impacts(
            left, samples_a, right, samples_b,
            track=stereo_engine.build_track3d(left, samples_a, right, samples_b,
                                              timeline_s=timeline))
        return {
            "name": name,
            "samples_a": [{"t_s": s.t_s, "px": list(map(float, s.px))} for s in samples_a],
            "samples_b": [{"t_s": s.t_s, "px": list(map(float, s.px))} for s in samples_b],
            "timeline_s": timeline,
            "impacts": [{"t_s": i.t_s, "surface": i.surface,
                          "point_ft": [float(v) for v in i.point_ft], "call": i.call,
                          "margin_ft": i.margin_ft, "confidence": i.confidence,
                          "snap_disagreement_ft": i.snap_disagreement_ft}
                         for i in impacts],
        }

    states, (t_true, _p) = simulate_front_wall_shot()
    samples_a = sample_camera(states, left, fps=60.0)
    samples_b = sample_camera(states, right, fps=60.0, phase_s=0.007)
    occluded_b = [s for s in samples_b if not (t_true - 0.3 <= s.t_s <= t_true + 0.1)]
    gap_a = [s for s in samples_a if not (t_true - 0.3 <= s.t_s <= t_true - 0.005)]
    gap_b = [s for s in samples_b if not (t_true - 0.3 <= s.t_s <= t_true - 0.005)]
    trajectories = [
        trajectory_case("clean", samples_a, samples_b),
        trajectory_case("occluded_one_view", samples_a, occluded_b),
        trajectory_case("no_call", gap_a, gap_b),
    ]

    lattice = [np.array([x, y, z]) for x in (5.25, 10.5, 15.75)
               for y in (4.0, 12.0, 20.0, 28.0) for z in (1.0, 8.0)]
    obs_a = [(p, np.asarray(left.project(p))) for p in lattice]
    obs_b = [(p, np.asarray(right.project(p))) for p in lattice]
    import dataclasses
    bias = np.array([0.0, 0.0, 0.5])
    biased_right = dataclasses.replace(
        right, camera_center_ft=right.camera_center_ft + bias)
    pair_case = {
        "obs_lattice": [{"court_ft": [float(v) for v in p],
                          "px_a": [float(v) for v in pa], "px_b": [float(v) for v in pb]}
                         for (p, pa), (_p2, pb) in zip(obs_a, obs_b)],
        "good": stereo_engine.pair_agreement(left, obs_a, right, obs_b),
        "biased": {"bias_ft": [0.0, 0.0, 0.5],
                    **stereo_engine.pair_agreement(left, obs_a, biased_right, obs_b)},
    }
```

Assemble with `"schema": "stereo-goldens-v2"`, `"trajectories": trajectories`,
`"pair_agreement": pair_case` (drop the old singular `"trajectory"` key). Keep
`sort_keys=True, indent=2` and the dual-file write. NOTE: `pair_agreement` returns
numpy floats in some fields — coerce every report value with
`{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in report.items()}`
before serializing.

- [ ] **Step 3: Update the golden test + regenerate**

Update `tests/test_stereo_goldens.py`: schema assert becomes `"stereo-goldens-v2"`;
`test_trajectory_impact_goldens` iterates `data["trajectories"]`, rebuilding each case's
samples and replaying with `timeline_s=case["timeline_s"]`, asserting impacts reproduce
exactly (surface/call/confidence equality, t/point at 1e-9) — and add:

```python
def test_trajectory_cases_cover_confidence_tiers():
    data, _, _ = load()
    tiers = {i["confidence"] for c in data["trajectories"] for i in c["impacts"]}
    assert {"high", "one_view", "no_call"} <= tiers


def test_pair_agreement_goldens():
    data, left, right = load()
    case = data["pair_agreement"]
    obs_a = [(np.array(e["court_ft"]), np.array(e["px_a"])) for e in case["obs_lattice"]]
    obs_b = [(np.array(e["court_ft"]), np.array(e["px_b"])) for e in case["obs_lattice"]]
    good = stereo_engine.pair_agreement(left, obs_a, right, obs_b)
    assert good["ok_pair"] == case["good"]["ok_pair"]
    assert abs(good["median_err_ft"] - case["good"]["median_err_ft"]) < 1e-9
    import dataclasses
    biased_model = dataclasses.replace(
        right, camera_center_ft=right.camera_center_ft + np.array(case["biased"]["bias_ft"]))
    biased = stereo_engine.pair_agreement(left, obs_a, biased_model, obs_b)
    assert biased["ok_pair"] is False and biased["ok_pair"] == case["biased"]["ok_pair"]
```

Run generator, verify determinism (second run → clean `git status`), full suite
`pytest tests/ -q` (expect 218 = 216 + timeline test + tier test + pair test − 0
removed; adjust expected count to actual and report it). **Verify the no_call case
actually produced a `no_call` impact** — if the authority produced something else,
record reality in the goldens and flag it in your report (do not force it).

- [ ] **Step 4: Commit**

```bash
git add stereo_engine.py tests/generate_stereo_goldens.py tests/test_stereo_goldens.py tests/test_stereo_track.py tests/stereo_goldens.json ios/Tests/Fixtures/stereo_goldens.json
git commit -m "feat(stereo): goldens v2 with confidence-tier trajectories and embedded timelines"
```

---

### Task 2: Swift CameraModel

**Files:**
- Create: `ios/Sources/Stereo/CameraModel.swift`
- Test: `ios/Tests/StereoGoldenTests.swift` (create)

**Interfaces:**
- Consumes: `ios/Tests/Fixtures/stereo_goldens.json` (v2).
- Produces (Tasks 3–6 depend on):
  ```swift
  struct Distortion { let k1: Double; let centerPx: SIMD2<Double>; let normPx: Double }
  struct CameraModel {
      let focalPx: Double
      let centerPx: SIMD2<Double>
      let rotation: simd_double3x3          // world -> camera (rows as in Python)
      let cameraCenterFt: SIMD3<Double>
      let distortion: Distortion?
      static func fromJSON(_ data: Data) throws -> CameraModel   // parses to_dict output
      func undistort(_ px: SIMD2<Double>) -> SIMD2<Double>       // division model; identity if nil
      func project(_ courtXYZ: SIMD3<Double>) -> SIMD2<Double>?  // nil when at/behind camera (depth <= 1e-9); UNDISTORTED px out
      func ray(_ undistortedPx: SIMD2<Double>) -> (origin: SIMD3<Double>, dir: SIMD3<Double>)
  }
  ```
  JSON keys are `court_model.to_dict()`'s: `focal_px`, `center_px` [2], `rotation` [[3]×3],
  `camera_center_ft` [3], `distortion` (null or {`model`, `k1`, `center_px`, `norm_px`}),
  plus ignorable `fit_rms_px`/`point_count`. Parse with `JSONSerialization` (matches
  the codebase's tolerance for loose JSON) or `Codable` with snake_case keys — pick one
  and keep it. `rotation` rows load in order (row-major from Python); note
  `simd_double3x3(rows:)` exists — use it and add a golden test that catches a
  transpose mistake (project a known point).
  Math (mirror court_model.CameraModel): `camera_point = R @ (X − C)`; project =
  `center + focal * (x/z, y/z)` with guard `z > 1e-9`; ray direction =
  `normalize(Rᵀ @ [(u−cx)/f, (v−cy)/f, 1])`, origin = C.
  Undistortion formula (exact contract): `r² = |p−c|²/norm²`; `factor = 1 + k1·r²`;
  guard `|factor| > 1e-9`; `p_u = c + (p−c)/factor`.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/StereoGoldenTests.swift
import XCTest
import simd
@testable import SquashLineCalling

final class StereoGoldenTests: XCTestCase {
    static var goldens: [String: Any] = [:]
    static var left: CameraModel!
    static var right: CameraModel!

    override class func setUp() {
        super.setUp()
        let url = Bundle(for: StereoGoldenTests.self)
            .url(forResource: "stereo_goldens", withExtension: "json")!
        let data = try! Data(contentsOf: url)
        goldens = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
        let cameras = goldens["cameras"] as! [String: Any]
        func decode(_ key: String) -> CameraModel {
            let json = try! JSONSerialization.data(withJSONObject: cameras[key]!)
            return try! CameraModel.fromJSON(json)
        }
        left = decode("left"); right = decode("right")
    }

    private func vec3(_ any: Any) -> SIMD3<Double> {
        let a = any as! [Double]; return SIMD3(a[0], a[1], a[2])
    }
    private func vec2(_ any: Any) -> SIMD2<Double> {
        let a = any as! [Double]; return SIMD2(a[0], a[1])
    }

    func testSchemaIsV2() {
        XCTAssertEqual(Self.goldens["schema"] as? String, "stereo-goldens-v2")
    }

    func testProjectMatchesGoldenPixels() {
        for case_ in Self.goldens["triangulation_cases"] as! [[String: Any]] {
            let point = vec3(case_["point_ft"]!)
            let pxA = Self.left.project(point)!
            let pxB = Self.right.project(point)!
            XCTAssertEqual(pxA.x, vec2(case_["px_a"]!).x, accuracy: 1e-7)
            XCTAssertEqual(pxA.y, vec2(case_["px_a"]!).y, accuracy: 1e-7)
            XCTAssertEqual(pxB.x, vec2(case_["px_b"]!).x, accuracy: 1e-7)
            XCTAssertEqual(pxB.y, vec2(case_["px_b"]!).y, accuracy: 1e-7)
        }
    }

    func testRayPassesThroughGoldenPoint() {
        for case_ in Self.goldens["triangulation_cases"] as! [[String: Any]] {
            let point = vec3(case_["point_ft"]!)
            let (origin, dir) = Self.left.ray(vec2(case_["px_a"]!))
            // Distance from the golden point to the ray must be ~0.
            let w = point - origin
            let dist = simd_length(w - simd_dot(w, dir) * dir)
            XCTAssertLessThan(dist, 1e-7)
        }
    }

    func testProjectBehindCameraReturnsNil() {
        // Behind the back wall relative to the camera's view direction.
        XCTAssertNil(Self.left.project(SIMD3(10.5, 40.0, 5.0)))
    }

    func testUndistortIdentityWhenNil() {
        let px = SIMD2(123.4, 567.8)
        XCTAssertEqual(Self.left.undistort(px), px)
    }

    func testUndistortDivisionModel() {
        let distorted = CameraModel(
            focalPx: 1000, centerPx: SIMD2(960, 540),
            rotation: matrix_identity_double3x3, cameraCenterFt: SIMD3(0, 0, 0),
            distortion: Distortion(k1: -0.1, centerPx: SIMD2(960, 540), normPx: 1000))
        let px = SIMD2<Double>(1160.0, 540.0)   // 200 px right of center; r2 = 0.04
        let undistorted = distorted.undistort(px)
        XCTAssertEqual(undistorted.x, 960.0 + 200.0 / (1.0 - 0.1 * 0.04), accuracy: 1e-9)
        XCTAssertEqual(undistorted.y, 540.0, accuracy: 1e-9)
    }
}
```

(This requires a memberwise init — declare the struct with `let`s and no custom init so
the synthesized memberwise init is internal, visible to `@testable`.)

- [ ] **Step 2: Run to verify failure** — build fails: `cannot find 'CameraModel'`.
Command: `... -only-testing:SquashLineCallingTests/StereoGoldenTests` (OS=17.0.1 pin).
If the fixture URL force-unwrap crashes instead, fix resource inclusion per Global
Constraints before proceeding.

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Stereo/CameraModel.swift
import Foundation
import simd

/// Division-model lens distortion — the JS<->Python<->Swift contract
/// (mirrors court_model.undistort_point).
struct Distortion {
    let k1: Double
    let centerPx: SIMD2<Double>
    let normPx: Double
}

/// Calibrated pinhole camera in court coordinates (FEET). Mirrors
/// court_model.CameraModel: `project` returns UNDISTORTED pixels and `ray`
/// expects them; callers undistort raw observations first.
struct CameraModel {
    let focalPx: Double
    let centerPx: SIMD2<Double>
    let rotation: simd_double3x3        // world -> camera
    let cameraCenterFt: SIMD3<Double>
    let distortion: Distortion?

    enum DecodeError: Error { case malformed(String) }

    static func fromJSON(_ data: Data) throws -> CameraModel {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let focal = obj["focal_px"] as? Double,
              let center = obj["center_px"] as? [Double], center.count == 2,
              let rows = obj["rotation"] as? [[Double]], rows.count == 3,
              rows.allSatisfy({ $0.count == 3 }),
              let cc = obj["camera_center_ft"] as? [Double], cc.count == 3 else {
            throw DecodeError.malformed("missing or malformed camera model fields")
        }
        var distortion: Distortion?
        if let d = obj["distortion"] as? [String: Any] {
            guard let k1 = d["k1"] as? Double,
                  let dc = d["center_px"] as? [Double], dc.count == 2 else {
                throw DecodeError.malformed("malformed distortion")
            }
            distortion = Distortion(k1: k1, centerPx: SIMD2(dc[0], dc[1]),
                                    normPx: d["norm_px"] as? Double ?? 1000.0)
        }
        let rotation = simd_double3x3(rows: [
            SIMD3(rows[0][0], rows[0][1], rows[0][2]),
            SIMD3(rows[1][0], rows[1][1], rows[1][2]),
            SIMD3(rows[2][0], rows[2][1], rows[2][2]),
        ])
        return CameraModel(focalPx: focal, centerPx: SIMD2(center[0], center[1]),
                           rotation: rotation, cameraCenterFt: SIMD3(cc[0], cc[1], cc[2]),
                           distortion: distortion)
    }

    func undistort(_ px: SIMD2<Double>) -> SIMD2<Double> {
        guard let d = distortion else { return px }
        let delta = px - d.centerPx
        let r2 = simd_length_squared(delta) / (d.normPx * d.normPx)
        let factor = 1.0 + d.k1 * r2
        precondition(abs(factor) > 1e-9, "Distortion factor collapsed to zero.")
        return d.centerPx + delta / factor
    }

    func project(_ courtXYZ: SIMD3<Double>) -> SIMD2<Double>? {
        let cameraPoint = rotation * (courtXYZ - cameraCenterFt)
        guard cameraPoint.z > 1e-9 else { return nil }
        return centerPx + focalPx * SIMD2(cameraPoint.x / cameraPoint.z,
                                          cameraPoint.y / cameraPoint.z)
    }

    func ray(_ undistortedPx: SIMD2<Double>) -> (origin: SIMD3<Double>, dir: SIMD3<Double>) {
        let cameraDir = SIMD3((undistortedPx.x - centerPx.x) / focalPx,
                              (undistortedPx.y - centerPx.y) / focalPx,
                              1.0)
        let worldDir = simd_normalize(rotation.transpose * cameraDir)
        return (cameraCenterFt, worldDir)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass** — 6/6 focused, then full suite
(expect 60/60 = 54 baseline + 6). If `testProjectMatchesGoldenPixels` is off by a
transpose, the `rows:` initializer is the first suspect.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Stereo/CameraModel.swift ios/Tests/StereoGoldenTests.swift
git commit -m "feat(stereo-swift): camera model with division undistortion and golden parity"
```

---

### Task 3: Swift StereoMath — triangulation, surfaces, snap, calls

**Files:**
- Create: `ios/Sources/Stereo/StereoMath.swift`
- Test: `ios/Tests/StereoGoldenTests.swift` (append)

**Interfaces:**
- Consumes: `CameraModel` (Task 2), goldens.
- Produces (Task 4/6 depend on):
  ```swift
  enum StereoMath {
      static let parallelEps = 1e-9
      static let outLineHeightFt = 15.0
      static let tinTopHeightFt = 19.0 / 12.0
      static let backWallOutHeightFt = 7.0
      static let courtWidthFt = 21.0
      static let courtLengthFt = 32.0
      static let boundsSlackFt = 0.5
      /// Ordered exactly like Python's SURFACES tuple — parity contract.
      static let surfaces = ["floor", "front_wall", "back_wall", "left_wall", "right_wall"]
      static func surfacePlane(_ name: String) -> (point: SIMD3<Double>, normal: SIMD3<Double>)
      static func sideWallOutHeightFt(_ yFt: Double) -> Double
      static func callForImpact(surface: String, point: SIMD3<Double>) -> (call: String, marginFt: Double)
      static func triangulate(_ a: CameraModel, _ b: CameraModel,
                              pxA: SIMD2<Double>, pxB: SIMD2<Double>)
          -> (point: SIMD3<Double>, gapFt: Double)?     // nil == Python's (None, inf); RAW px in
      static func snapToPlane(_ model: CameraModel, px: SIMD2<Double>, surface: String)
          -> SIMD3<Double>?                              // RAW px in
      static func fuseSnaps(_ a: SIMD3<Double>?, _ b: SIMD3<Double>?) -> SIMD3<Double>?
      static func planeDistance(surface: String, point: SIMD3<Double>) -> Double
  }
  ```
  All algorithms mirror stereo_engine.py line-for-line (closest-approach with a=c=1;
  s≤0/t≤0 rejection; per-surface bounds with 0.5 slack — floor checks x∧y, front/back
  walls x∧z≥−slack, side walls y∧z≥−slack).

- [ ] **Step 1: Append the failing golden-parity tests**

```swift
    func testTriangulationGoldenParity() {
        for case_ in Self.goldens["triangulation_cases"] as! [[String: Any]] {
            let result = StereoMath.triangulate(Self.left, Self.right,
                                                pxA: vec2(case_["px_a"]!),
                                                pxB: vec2(case_["px_b"]!))!
            let expected = vec3(case_["point_ft"]!)
            XCTAssertLessThan(simd_length(result.point - expected), 1e-7)
            XCTAssertEqual(result.gapFt, case_["gap_ft"] as! Double, accuracy: 1e-7)
        }
    }

    func testSnapGoldenParity() {
        for case_ in Self.goldens["snap_cases"] as! [[String: Any]] {
            let model = (case_["camera"] as! String) == "left" ? Self.left! : Self.right!
            let snap = StereoMath.snapToPlane(model, px: vec2(case_["px"]!),
                                              surface: case_["surface"] as! String)!
            XCTAssertLessThan(simd_length(snap - vec3(case_["point_ft"]!)), 1e-7)
        }
    }

    func testCallGoldenParity() {
        for case_ in Self.goldens["call_cases"] as! [[String: Any]] {
            let (call, margin) = StereoMath.callForImpact(
                surface: case_["surface"] as! String, point: vec3(case_["point_ft"]!))
            XCTAssertEqual(call, case_["call"] as! String)
            XCTAssertEqual(margin, case_["margin_ft"] as! Double, accuracy: 1e-7)
        }
    }

    func testSurfaceOrderMirrorsPython() {
        XCTAssertEqual(StereoMath.surfaces,
                       ["floor", "front_wall", "back_wall", "left_wall", "right_wall"])
    }
```

- [ ] **Step 2: Run to verify failure** — `cannot find 'StereoMath'`.

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Stereo/StereoMath.swift
import Foundation
import simd

/// Pure stereo geometry — mirrors stereo_engine.py exactly (constants,
/// algorithms, and quirks). Do not "improve" without changing the Python
/// authority and regenerating goldens.
enum StereoMath {
    static let parallelEps = 1e-9
    static let outLineHeightFt = 15.0
    static let tinTopHeightFt = 19.0 / 12.0
    static let backWallOutHeightFt = 7.0
    static let courtWidthFt = 21.0
    static let courtLengthFt = 32.0
    static let boundsSlackFt = 0.5
    static let surfaces = ["floor", "front_wall", "back_wall", "left_wall", "right_wall"]

    private static let planes: [String: (SIMD3<Double>, SIMD3<Double>)] = [
        "floor": (SIMD3(0, 0, 0), SIMD3(0, 0, 1)),
        "front_wall": (SIMD3(0, 0, 0), SIMD3(0, 1, 0)),
        "back_wall": (SIMD3(0, courtLengthFt, 0), SIMD3(0, -1, 0)),
        "left_wall": (SIMD3(0, 0, 0), SIMD3(1, 0, 0)),
        "right_wall": (SIMD3(courtWidthFt, 0, 0), SIMD3(-1, 0, 0)),
    ]

    static func surfacePlane(_ name: String) -> (point: SIMD3<Double>, normal: SIMD3<Double>) {
        let (p, n) = planes[name]!
        return (p, n)
    }

    static func sideWallOutHeightFt(_ yFt: Double) -> Double {
        outLineHeightFt + (backWallOutHeightFt - outLineHeightFt) * (yFt / courtLengthFt)
    }

    static func callForImpact(surface: String, point: SIMD3<Double>) -> (call: String, marginFt: Double) {
        let (_, y, z) = (point.x, point.y, point.z)
        switch surface {
        case "floor":
            return ("bounce", 0.0)
        case "front_wall":
            if z >= outLineHeightFt { return ("out", z - outLineHeightFt) }
            if z <= tinTopHeightFt { return ("down", tinTopHeightFt - z) }
            return ("in", min(outLineHeightFt - z, z - tinTopHeightFt))
        case "left_wall", "right_wall":
            let line = sideWallOutHeightFt(y)
            return z >= line ? ("out", z - line) : ("in", line - z)
        case "back_wall":
            return z >= backWallOutHeightFt
                ? ("out", z - backWallOutHeightFt) : ("in", backWallOutHeightFt - z)
        default:
            fatalError("Unknown surface: \(surface)")
        }
    }

    static func planeDistance(surface: String, point: SIMD3<Double>) -> Double {
        let (planePoint, normal) = planes[surface]!
        return simd_dot(point - planePoint, normal)
    }

    static func triangulate(_ a: CameraModel, _ b: CameraModel,
                            pxA: SIMD2<Double>, pxB: SIMD2<Double>)
        -> (point: SIMD3<Double>, gapFt: Double)? {
        let (o1, d1) = a.ray(a.undistort(pxA))
        let (o2, d2) = b.ray(b.undistort(pxB))
        let w0 = o1 - o2
        let bDot = simd_dot(d1, d2)
        let d = simd_dot(d1, w0)
        let e = simd_dot(d2, w0)
        let denom = 1.0 - bDot * bDot
        guard denom >= parallelEps else { return nil }
        let s = (bDot * e - d) / denom
        let t = (e - bDot * d) / denom
        guard s > 0.0, t > 0.0 else { return nil }
        let p1 = o1 + s * d1
        let p2 = o2 + t * d2
        return ((p1 + p2) / 2.0, simd_length(p1 - p2))
    }

    private static func inSurfaceBounds(surface: String, point: SIMD3<Double>) -> Bool {
        let lo = -boundsSlackFt
        switch surface {
        case "floor":
            return point.x >= lo && point.x <= courtWidthFt + boundsSlackFt
                && point.y >= lo && point.y <= courtLengthFt + boundsSlackFt
        case "front_wall", "back_wall":
            return point.x >= lo && point.x <= courtWidthFt + boundsSlackFt && point.z >= lo
        default:
            return point.y >= lo && point.y <= courtLengthFt + boundsSlackFt && point.z >= lo
        }
    }

    static func snapToPlane(_ model: CameraModel, px: SIMD2<Double>, surface: String)
        -> SIMD3<Double>? {
        let (planePoint, normal) = planes[surface]!
        let (origin, dir) = model.ray(model.undistort(px))
        let denom = simd_dot(dir, normal)
        guard abs(denom) >= parallelEps else { return nil }
        let t = simd_dot(planePoint - origin, normal) / denom
        guard t > 0.0 else { return nil }
        let point = origin + t * dir
        return inSurfaceBounds(surface: surface, point: point) ? point : nil
    }

    static func fuseSnaps(_ a: SIMD3<Double>?, _ b: SIMD3<Double>?) -> SIMD3<Double>? {
        switch (a, b) {
        case (nil, let b): return b
        case (let a, nil): return a
        case (let a?, let b?): return (a + b) / 2.0
        }
    }
}
```

(One deliberate mapping, comment it in code: Python returns `(None, inf)` /
distinguishes parallel vs behind-camera only by both being rejections; Swift folds both
into `nil` — Task 4's consumers only branch on nil-ness, same as Python branches on
`point is None`.)

- [ ] **Step 4: Run tests** — focused 10/10 (6 prior + 4 new), full suite 64/64.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Stereo/StereoMath.swift ios/Tests/StereoGoldenTests.swift
git commit -m "feat(stereo-swift): triangulation, surfaces, snap, and calls with golden parity"
```

---

### Task 4: Swift StereoTrack — fits, track, impact detection

**Files:**
- Create: `ios/Sources/Stereo/StereoTrack.swift`
- Test: `ios/Tests/StereoGoldenTests.swift` (append)

**Interfaces:**
- Consumes: `CameraModel`, `StereoMath` (incl. `planeDistance`), goldens v2 trajectories.
- Produces (Task 6 depends on):
  ```swift
  struct TrackSample { let tS: Double; let px: SIMD2<Double> }     // RAW pixels
  struct TrackPoint3D { let tS: Double; let pointFt: SIMD3<Double>; let gapFt: Double }
  struct StereoImpact: Equatable {
      let tS: Double; let surface: String; let pointFt: SIMD3<Double>
      let call: String; let marginFt: Double; let confidence: String
      let snapDisagreementFt: Double?
  }
  enum StereoTrack {
      static let fitWindowSamples = 7
      static let minFitSamples = 4
      static let windowGapRatioMax = 3.0
      static let snapDisagreementMaxFt = 0.3
      static let impactProximityFt = 1.5
      static let impactMergeS = 0.060
      static let preImpactWindowS = 0.25
      static let preImpactGuardS = 1.0 / 240.0
      static func evalPixelTrack(_ samples: [TrackSample], tS: Double,
                                 window: Int = fitWindowSamples) -> SIMD2<Double>?
      static func buildTrack3D(_ a: CameraModel, _ samplesA: [TrackSample],
                               _ b: CameraModel, _ samplesB: [TrackSample],
                               timelineS: [Double]) -> [TrackPoint3D]
      static func detectImpacts(_ a: CameraModel, _ samplesA: [TrackSample],
                                _ b: CameraModel, _ samplesB: [TrackSample],
                                timelineS: [Double]) -> [StereoImpact]
  }
  ```
  Mirrors stereo_engine.py exactly:
  - `evalPixelTrack`: nearest-`window` samples by |t−tS|; reject < minFitSamples; the
    locality guard (sorted internal gaps; reject when median > 0 AND max > 3×median —
    INCLUDING the inert-at-median-0 quirk, commented); centered quadratic fit per u,v
    via 3×3 normal equations (`Σ1, Σt, Σt², Σt³, Σt⁴` symmetric matrix solved with
    Cramer's rule or `simd_double3x3.inverse` — centered ts keep it conditioned),
    evaluated at 0.
  - `buildTrack3D`: iterate the PROVIDED timeline (no arange in Swift — live callers
    build their own grid, tests use the embedded golden timelines); skip nil evals /
    nil triangulations.
  - `detectImpacts`: for surface in `StereoMath.surfaces` (ordered!); local minima of
    plane distance below proximity with strict sign change (approaching < 0, leaving > 0);
    pre-impact refit window `[t−0.25, t−1/240]` evaluated at t (window passed as its own
    length like Python's `window=len(window)`); snap both cameras; confidence tiers
    exactly (both ≤ 0.3 → high; both but > 0.3 → one_view; one → one_view; none →
    no_call with the raw track point); sort by `(tS, surface)`; 60 ms same-surface merge
    keeping the deeper |plane distance|.

- [ ] **Step 1: Append the failing trajectory-parity tests**

```swift
    private func trackSamples(_ any: Any) -> [TrackSample] {
        (any as! [[String: Any]]).map {
            TrackSample(tS: $0["t_s"] as! Double, px: vec2($0["px"]!))
        }
    }

    func testTrajectoryGoldenParityAllCases() {
        for case_ in Self.goldens["trajectories"] as! [[String: Any]] {
            let name = case_["name"] as! String
            let impacts = StereoTrack.detectImpacts(
                Self.left, trackSamples(case_["samples_a"]!),
                Self.right, trackSamples(case_["samples_b"]!),
                timelineS: case_["timeline_s"] as! [Double])
            let expected = case_["impacts"] as! [[String: Any]]
            XCTAssertEqual(impacts.count, expected.count, "case \(name)")
            for (got, want) in zip(impacts, expected) {
                XCTAssertEqual(got.surface, want["surface"] as! String, "case \(name)")
                XCTAssertEqual(got.call, want["call"] as! String, "case \(name)")
                XCTAssertEqual(got.confidence, want["confidence"] as! String, "case \(name)")
                XCTAssertEqual(got.tS, want["t_s"] as! Double, accuracy: 1e-6, "case \(name)")
                XCTAssertLessThan(simd_length(got.pointFt - vec3(want["point_ft"]!)),
                                  1e-3, "case \(name)")
            }
        }
    }

    func testConfidenceTiersCovered() {
        let trajectories = Self.goldens["trajectories"] as! [[String: Any]]
        let tiers = Set(trajectories.flatMap {
            ($0["impacts"] as! [[String: Any]]).map { $0["confidence"] as! String }
        })
        XCTAssertTrue(tiers.isSuperset(of: ["high", "one_view", "no_call"]))
    }
```

- [ ] **Step 2: Run to verify failure** — `cannot find 'StereoTrack'`.

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Stereo/StereoTrack.swift
import Foundation
import simd

struct TrackSample { let tS: Double; let px: SIMD2<Double> }
struct TrackPoint3D { let tS: Double; let pointFt: SIMD3<Double>; let gapFt: Double }

struct StereoImpact: Equatable {
    let tS: Double
    let surface: String
    let pointFt: SIMD3<Double>
    let call: String
    let marginFt: Double
    let confidence: String
    let snapDisagreementFt: Double?
}

/// Track interpolation + impact detection — mirrors stereo_engine.py
/// (eval_pixel_track / build_track3d / detect_impacts) exactly, including
/// quirks. Do not "fix" behavior here without changing the Python authority.
enum StereoTrack {
    static let fitWindowSamples = 7
    static let minFitSamples = 4
    static let windowGapRatioMax = 3.0
    static let snapDisagreementMaxFt = 0.3
    static let impactProximityFt = 1.5
    static let impactMergeS = 0.060
    static let preImpactWindowS = 0.25
    static let preImpactGuardS = 1.0 / 240.0

    static func evalPixelTrack(_ samples: [TrackSample], tS: Double,
                               window: Int = fitWindowSamples) -> SIMD2<Double>? {
        guard !samples.isEmpty else { return nil }
        let nearest = samples.sorted { abs($0.tS - tS) < abs($1.tS - tS) }.prefix(window)
        guard nearest.count >= minFitSamples else { return nil }
        // Locality guard — mirrors stereo_engine.py eval_pixel_track: inert
        // when the median internal gap is 0 (duplicate timestamps), by design.
        let sortedTs = nearest.map(\.tS).sorted()
        let gaps = zip(sortedTs.dropFirst(), sortedTs).map(-)
        if !gaps.isEmpty {
            let sortedGaps = gaps.sorted()
            let median = sortedGaps.count % 2 == 1
                ? sortedGaps[sortedGaps.count / 2]
                : (sortedGaps[sortedGaps.count / 2 - 1] + sortedGaps[sortedGaps.count / 2]) / 2.0
            if median > 0.0, sortedGaps.last! > windowGapRatioMax * median { return nil }
        }
        let ts = nearest.map { $0.tS - tS }
        func fitAtZero(_ values: [Double]) -> Double {
            var s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0
            var b0 = 0.0, b1 = 0.0, b2 = 0.0
            for (t, v) in zip(ts, values) {
                let t2 = t * t
                s0 += 1; s1 += t; s2 += t2; s3 += t2 * t; s4 += t2 * t2
                b0 += v; b1 += v * t; b2 += v * t2
            }
            // Normal equations for v = c0 + c1·t + c2·t²; value at t=0 is c0.
            let m = simd_double3x3(rows: [SIMD3(s0, s1, s2),
                                          SIMD3(s1, s2, s3),
                                          SIMD3(s2, s3, s4)])
            let coeffs = m.inverse * SIMD3(b0, b1, b2)
            return coeffs.x
        }
        return SIMD2(fitAtZero(nearest.map(\.px.x)), fitAtZero(nearest.map(\.px.y)))
    }

    static func buildTrack3D(_ a: CameraModel, _ samplesA: [TrackSample],
                             _ b: CameraModel, _ samplesB: [TrackSample],
                             timelineS: [Double]) -> [TrackPoint3D] {
        guard !samplesA.isEmpty, !samplesB.isEmpty else { return [] }
        var track: [TrackPoint3D] = []
        for tS in timelineS {
            guard let pxA = evalPixelTrack(samplesA, tS: tS),
                  let pxB = evalPixelTrack(samplesB, tS: tS),
                  let result = StereoMath.triangulate(a, b, pxA: pxA, pxB: pxB) else {
                continue
            }
            track.append(TrackPoint3D(tS: tS, pointFt: result.point, gapFt: result.gapFt))
        }
        return track
    }

    private static func preImpactEval(_ samples: [TrackSample], tImpact: Double)
        -> SIMD2<Double>? {
        let window = samples.filter {
            $0.tS >= tImpact - preImpactWindowS && $0.tS <= tImpact - preImpactGuardS
        }
        guard window.count >= minFitSamples else { return nil }
        return evalPixelTrack(window, tS: tImpact, window: window.count)
    }

    static func detectImpacts(_ a: CameraModel, _ samplesA: [TrackSample],
                              _ b: CameraModel, _ samplesB: [TrackSample],
                              timelineS: [Double]) -> [StereoImpact] {
        let track = buildTrack3D(a, samplesA, b, samplesB, timelineS: timelineS)
        guard track.count >= 3 else { return [] }
        var impacts: [StereoImpact] = []
        for surface in StereoMath.surfaces {
            let dists = track.map { StereoMath.planeDistance(surface: surface, point: $0.pointFt) }
            for i in 1..<(track.count - 1) {
                guard dists[i] <= impactProximityFt,
                      dists[i] <= dists[i - 1], dists[i] <= dists[i + 1],
                      dists[i] - dists[i - 1] < 0.0, dists[i + 1] - dists[i] > 0.0 else {
                    continue
                }
                let tImpact = track[i].tS
                var snapA: SIMD3<Double>?
                var snapB: SIMD3<Double>?
                if let px = preImpactEval(samplesA, tImpact: tImpact) {
                    snapA = StereoMath.snapToPlane(a, px: px, surface: surface)
                }
                if let px = preImpactEval(samplesB, tImpact: tImpact) {
                    snapB = StereoMath.snapToPlane(b, px: px, surface: surface)
                }
                var disagreement: Double?
                let confidence: String
                let point: SIMD3<Double>
                if let sa = snapA, let sb = snapB {
                    let gap = simd_length(sa - sb)
                    disagreement = gap
                    confidence = gap <= snapDisagreementMaxFt ? "high" : "one_view"
                    point = StereoMath.fuseSnaps(sa, sb)!
                } else if let lone = StereoMath.fuseSnaps(snapA, snapB) {
                    confidence = "one_view"
                    point = lone
                } else {
                    confidence = "no_call"
                    point = track[i].pointFt
                }
                let (call, margin) = StereoMath.callForImpact(surface: surface, point: point)
                impacts.append(StereoImpact(tS: tImpact, surface: surface, pointFt: point,
                                            call: call, marginFt: margin,
                                            confidence: confidence,
                                            snapDisagreementFt: disagreement))
            }
        }
        impacts.sort { ($0.tS, $0.surface) < ($1.tS, $1.surface) }
        var merged: [StereoImpact] = []
        for impact in impacts {
            if let last = merged.last, last.surface == impact.surface,
               impact.tS - last.tS < impactMergeS {
                let prevDepth = abs(StereoMath.planeDistance(surface: impact.surface,
                                                             point: last.pointFt))
                let curDepth = abs(StereoMath.planeDistance(surface: impact.surface,
                                                            point: impact.pointFt))
                if curDepth < prevDepth { merged[merged.count - 1] = impact }
                continue
            }
            merged.append(impact)
        }
        return merged
    }
}
```

(`(Double, String) < (Double, String)` tuple comparison works in Swift ≥ 5 for
Comparable elements — matches Python's tuple sort semantics for this key.)

- [ ] **Step 4: Run tests** — focused 12/12, full suite 66/66. If `testTrajectoryGoldenParityAllCases`
misses the 1e-3 point tolerance, the normal-equations fit is the first suspect: verify
`fitAtZero` against a hand quadratic before touching tolerances (never widen them).

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Stereo/StereoTrack.swift ios/Tests/StereoGoldenTests.swift
git commit -m "feat(stereo-swift): track interpolation and impact detection with golden parity"
```

---

### Task 5: Solved-model exchange over the peer layer

**Files:**
- Modify: `ios/Sources/API/APIClient.swift` (read it first; add one method matching its
  existing request style), `ios/Sources/Peer/PeerSession.swift` (one seam),
  `ios/Sources/Config.swift` only if APIClient needs nothing new (likely none).
- Test: `ios/Tests/StereoWiringTests.swift` (create)

**Interfaces:**
- Consumes: `/api/camera-model` (server, merged), `ControlMessage.calibration(profileID:calibrationJSON:)` (existing), `PeerSession` (existing).
- Produces:
  ```swift
  // APIClient (match its existing function style — read the file and mirror it):
  func fetchSolvedCameraModel(calibrationJSON: String) async throws -> String
      // POST /api/camera-model {"calibration_json": ...}; returns the
      // "camera_model" sub-object re-serialized as a JSON string; throws
      // APIError (existing error type) when status != "ok" or key missing.
  // PeerSession additions:
  var onCalibration: ((_ profileID: String, _ payloadJSON: String) -> Void)?
      // fired (on the transport delivery context, like onRemoteDetections)
      // when a .calibration control message arrives — replaces the current
      // silent `case .calibration: break`.
  func sendCalibration(profileID: String, payloadJSON: String)
      // sends .calibration; allowed in .ready or .live (same gate as sendDetections).
  ```
  Semantic note (comment at both call sites): as of Phase 3 the `.calibration`
  message's `calibrationJSON` field carries the SOLVED camera-model JSON
  (`CameraModel.fromJSON`-parseable), not raw wizard taps.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/StereoWiringTests.swift
import XCTest
@testable import SquashLineCalling

final class StereoWiringTests: XCTestCase {
    func testCalibrationMessageRoundTripsThroughSessions() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }
        XCTAssertEqual(primary.phase, .ready)

        var received: (String, String)?
        primary.onCalibration = { received = ($0, $1) }
        secondary.sendCalibration(profileID: "ucsc-right-fin",
                                  payloadJSON: "{\"focal_px\": 1600}")
        XCTAssertEqual(received?.0, "ucsc-right-fin")
        XCTAssertEqual(received?.1, "{\"focal_px\": 1600}")
    }

    func testSendCalibrationGatedOnPhase() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        var fired = false
        primary.onCalibration = { _, _ in fired = true }
        // Not started/paired: sending from an idle session must be a no-op.
        primary.sendCalibration(profileID: "x", payloadJSON: "{}")
        XCTAssertFalse(fired)
    }
}
```

- [ ] **Step 2: Run to verify failure** — `has no member 'onCalibration'`.

- [ ] **Step 3: Implement**

In `PeerSession.swift`: add `var onCalibration: ((String, String) -> Void)?` next to
`onRemoteDetections`; replace the control-handler's `case .calibration:`... — read the
actual current handler: the merged code lists `.calibration` inside the ignored group
(`case .calibration, .record, .event, .sessionManifest: break`). Split it out:

```swift
        case .calibration(let profileID, let calibrationJSON):
            onCalibration?(profileID, calibrationJSON)
```

Add (locked entry point, same discipline as `sendDetections`):

```swift
    /// Phase 3: the payload carries the SOLVED camera-model JSON
    /// (CameraModel.fromJSON-parseable), not raw wizard taps.
    func sendCalibration(profileID: String, payloadJSON: String) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard internalPhase == .live || internalPhase == .ready else { return }
        sendControl(.calibration(profileID: profileID, calibrationJSON: payloadJSON))
    }
```

In `APIClient.swift`: read the file, mirror its existing POST/JSON pattern and error
type to implement `fetchSolvedCameraModel(calibrationJSON:)` per the Produces block
(URL path `api/camera-model`, body `{"calibration_json": <string>}` as JSON, parse
`camera_model` out of the response, re-serialize with `JSONSerialization`; a missing
`camera_model` or non-"ok" status throws the client's existing error type with a
descriptive message). No unit test for the network call itself (matches the codebase:
APIClient methods are exercised via server integration, not stubbed URLProtocol) —
state this in your report.

- [ ] **Step 4: Run tests** — focused 2/2, full suite 68/68.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Peer/PeerSession.swift ios/Sources/API/APIClient.swift ios/Tests/StereoWiringTests.swift
git commit -m "feat(stereo-swift): solved-model exchange over the calibration message"
```

---

### Task 6: Live StereoEngine

**Files:**
- Create: `ios/Sources/Stereo/StereoEngine.swift`
- Test: `ios/Tests/StereoEngineTests.swift` (create)

**Interfaces:**
- Consumes: `CameraModel`, `StereoTrack`, `StereoMath`, `DetectionTuple`, `BallObservation`.
- Produces (Task 7 depends on):
  ```swift
  enum StereoEvent: Equatable {
      case impact(StereoImpact)
  }
  /// Live fusion on the primary. Owner-pumped (like PeerSession.tick).
  /// All state confined to an internal serial queue; callbacks fire on it.
  final class StereoEngine {
      static let windowS = 4.0            // rolling sample retention
      static let processHz = 120.0        // resample grid density
      static let processIntervalS = 0.25  // how often processIfDue recomputes
      static let emitDedupeS = 0.05       // same-surface impacts within this of an emitted one are duplicates
      var onEvent: ((StereoEvent) -> Void)?
      init(localModel: CameraModel, remoteModel: CameraModel,
           remoteToLocal: @escaping (Double) -> Double?)
      func addLocalObservation(_ observation: BallObservation, frameW: Int, frameH: Int)
          // Vision-normalized bottom-left rect -> RAW pixel center:
          // (midX*W, (1-midY)*H) — same mapping as DetectionMapper.
      func addLocalPixel(_ px: SIMD2<Double>, tS: Double)          // test seam / direct feed
      func addRemote(_ tuples: [DetectionTuple])
          // ptsNs/1e9 -> remoteToLocal(t); tuples whose mapping returns nil are dropped
      func processIfDue(now: TimeInterval)
          // no-op unless processIntervalS elapsed since last run; builds the
          // 120 Hz timeline over the overlap of retained windows, runs
          // detectImpacts, emits any impact not already emitted (dedupe:
          // same surface AND |t - emitted t| < emitDedupeS)
  }
  ```
  Timeline construction (live analogue of the Python default, documented): `tLo` =
  max(first retained local t, first retained remote t), `tHi` = min(last, last);
  `count = Int(((tHi - tLo) * processHz).rounded(.down))`, grid `tLo + i/processHz` —
  matches `np.arange` semantics for our ranges. Samples older than `windowS` behind the
  newest are pruned on every add. Emitted-impact memory also prunes on the same horizon.

- [ ] **Step 1: Write the failing tests**

```swift
// ios/Tests/StereoEngineTests.swift
import XCTest
import simd
@testable import SquashLineCalling

final class StereoEngineTests: XCTestCase {
    private var goldens: [String: Any] = [:]
    private var left: CameraModel!
    private var right: CameraModel!

    override func setUp() {
        super.setUp()
        let url = Bundle(for: Self.self)
            .url(forResource: "stereo_goldens", withExtension: "json")!
        goldens = try! JSONSerialization.jsonObject(
            with: try! Data(contentsOf: url)) as! [String: Any]
        let cameras = goldens["cameras"] as! [String: Any]
        func decode(_ key: String) -> CameraModel {
            try! CameraModel.fromJSON(
                try! JSONSerialization.data(withJSONObject: cameras[key]!))
        }
        left = decode("left"); right = decode("right")
    }

    private func caseNamed(_ name: String) -> [String: Any] {
        (goldens["trajectories"] as! [[String: Any]]).first { ($0["name"] as! String) == name }!
    }

    /// Feed the clean golden trajectory through the live paths and expect the
    /// same impact the batch authority produced.
    func testCleanTrajectoryEmitsGoldenImpact() {
        let case_ = caseNamed("clean")
        let engine = StereoEngine(localModel: left, remoteModel: right,
                                  remoteToLocal: { $0 })   // same clock in tests
        var events: [StereoEvent] = []
        engine.onEvent = { events.append($0) }

        let samplesA = case_["samples_a"] as! [[String: Any]]
        let samplesB = case_["samples_b"] as! [[String: Any]]
        for s in samplesA {
            engine.addLocalPixel(SIMD2((s["px"] as! [Double])[0], (s["px"] as! [Double])[1]),
                                 tS: s["t_s"] as! Double)
        }
        let tuples = samplesB.enumerated().map { index, s in
            DetectionTuple(seq: UInt32(index),
                           ptsNs: UInt64((s["t_s"] as! Double) * 1_000_000_000),
                           x: Float((s["px"] as! [Double])[0]),
                           y: Float((s["px"] as! [Double])[1]),
                           conf: Float16(1.0), bboxH: Float16(10))
        }
        engine.addRemote(tuples)
        engine.processIfDue(now: 100.0)
        engine.flushForTesting()

        let expected = (case_["impacts"] as! [[String: Any]]).first!
        guard case .impact(let impact)? = events.first else {
            return XCTFail("no impact emitted")
        }
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(impact.surface, expected["surface"] as! String)
        XCTAssertEqual(impact.call, expected["call"] as! String)
        XCTAssertEqual(impact.confidence, expected["confidence"] as! String)
        XCTAssertEqual(impact.tS, expected["t_s"] as! Double, accuracy: 0.02)
    }

    /// Re-processing must not re-emit the same impact (dedupe), and a second
    /// processIfDue inside the interval must be a no-op.
    func testImpactEmittedExactlyOnce() {
        let case_ = caseNamed("clean")
        let engine = StereoEngine(localModel: left, remoteModel: right,
                                  remoteToLocal: { $0 })
        var count = 0
        engine.onEvent = { _ in count += 1 }
        for s in case_["samples_a"] as! [[String: Any]] {
            engine.addLocalPixel(SIMD2((s["px"] as! [Double])[0], (s["px"] as! [Double])[1]),
                                 tS: s["t_s"] as! Double)
        }
        let tuples = (case_["samples_b"] as! [[String: Any]]).enumerated().map { index, s in
            DetectionTuple(seq: UInt32(index),
                           ptsNs: UInt64((s["t_s"] as! Double) * 1_000_000_000),
                           x: Float((s["px"] as! [Double])[0]),
                           y: Float((s["px"] as! [Double])[1]),
                           conf: Float16(1.0), bboxH: Float16(10))
        }
        engine.addRemote(tuples)
        engine.processIfDue(now: 100.0)
        engine.processIfDue(now: 100.1)   // inside interval: no-op
        engine.processIfDue(now: 100.5)   // recompute: dedupe must hold
        engine.flushForTesting()
        XCTAssertEqual(count, 1)
    }

    func testRemoteTuplesWithUnmappableClockAreDropped() {
        let engine = StereoEngine(localModel: left, remoteModel: right,
                                  remoteToLocal: { _ in nil })
        engine.addRemote([DetectionTuple(seq: 0, ptsNs: 1, x: 1, y: 1,
                                         conf: Float16(1), bboxH: Float16(1))])
        engine.processIfDue(now: 1.0)
        engine.flushForTesting()
        // Nothing to assert beyond "no crash, no event": guard via onEvent.
        var fired = false
        engine.onEvent = { _ in fired = true }
        engine.processIfDue(now: 2.0)
        engine.flushForTesting()
        XCTAssertFalse(fired)
    }
}
```

`flushForTesting()` = synchronous barrier on the internal queue (`queue.sync {}`),
exposed internal (not private) for `@testable`.

- [ ] **Step 2: Run to verify failure** — `cannot find 'StereoEngine'`.

- [ ] **Step 3: Implement**

```swift
// ios/Sources/Stereo/StereoEngine.swift
import Foundation
import simd

enum StereoEvent: Equatable {
    case impact(StereoImpact)
}

/// Live stereo fusion on the primary phone. Owner-pumped: call
/// `processIfDue` on any cadence ≥ processIntervalS (RecordModel's peer pump
/// is the natural driver). All mutable state is confined to `queue`;
/// `onEvent` fires on `queue` — hop before touching UI.
final class StereoEngine {
    static let windowS = 4.0
    static let processHz = 120.0
    static let processIntervalS = 0.25
    static let emitDedupeS = 0.05

    var onEvent: ((StereoEvent) -> Void)?

    private let queue = DispatchQueue(label: "slc.stereo.engine")
    private let localModel: CameraModel
    private let remoteModel: CameraModel
    private let remoteToLocal: (Double) -> Double?
    private var localSamples: [TrackSample] = []
    private var remoteSamples: [TrackSample] = []
    private var emitted: [(tS: Double, surface: String)] = []
    private var lastProcessAt: TimeInterval = -.infinity

    init(localModel: CameraModel, remoteModel: CameraModel,
         remoteToLocal: @escaping (Double) -> Double?) {
        self.localModel = localModel
        self.remoteModel = remoteModel
        self.remoteToLocal = remoteToLocal
    }

    func addLocalObservation(_ observation: BallObservation, frameW: Int, frameH: Int) {
        let px = SIMD2(Double(observation.rect.midX) * Double(frameW),
                       (1.0 - Double(observation.rect.midY)) * Double(frameH))
        addLocalPixel(px, tS: observation.timestamp)
    }

    func addLocalPixel(_ px: SIMD2<Double>, tS: Double) {
        queue.async {
            self.localSamples.append(TrackSample(tS: tS, px: px))
            self.prune()
        }
    }

    func addRemote(_ tuples: [DetectionTuple]) {
        queue.async {
            for tuple in tuples {
                guard let tS = self.remoteToLocal(Double(tuple.ptsNs) / 1_000_000_000.0) else {
                    continue
                }
                self.remoteSamples.append(
                    TrackSample(tS: tS, px: SIMD2(Double(tuple.x), Double(tuple.y))))
            }
            self.prune()
        }
    }

    func processIfDue(now: TimeInterval) {
        queue.async {
            guard now - self.lastProcessAt >= Self.processIntervalS else { return }
            self.lastProcessAt = now
            self.process()
        }
    }

    /// Test seam: synchronous barrier so assertions observe queued work.
    func flushForTesting() {
        queue.sync {}
    }

    private func prune() {
        let newest = max(localSamples.last?.tS ?? -.infinity,
                         remoteSamples.last?.tS ?? -.infinity)
        guard newest.isFinite else { return }
        let horizon = newest - Self.windowS
        localSamples.removeAll { $0.tS < horizon }
        remoteSamples.removeAll { $0.tS < horizon }
        emitted.removeAll { $0.tS < horizon }
    }

    private func process() {
        guard let localFirst = localSamples.first?.tS,
              let localLast = localSamples.last?.tS,
              let remoteFirst = remoteSamples.first?.tS,
              let remoteLast = remoteSamples.last?.tS else { return }
        let tLo = max(localFirst, remoteFirst)
        let tHi = min(localLast, remoteLast)
        guard tHi > tLo else { return }
        // np.arange-equivalent grid: tLo + i/hz for i in 0..<count, strictly < tHi.
        let count = Int(((tHi - tLo) * Self.processHz).rounded(.down))
        guard count >= 3 else { return }
        let timeline = (0..<count).map { tLo + Double($0) / Self.processHz }
        let impacts = StereoTrack.detectImpacts(localModel, localSamples,
                                                remoteModel, remoteSamples,
                                                timelineS: timeline)
        for impact in impacts {
            let isDuplicate = emitted.contains {
                $0.surface == impact.surface && abs($0.tS - impact.tS) < Self.emitDedupeS
            }
            if isDuplicate { continue }
            emitted.append((impact.tS, impact.surface))
            onEvent?(.impact(impact))
        }
    }
}
```

- [ ] **Step 4: Run tests** — focused 3/3, full suite 71/71. If the clean-case impact's
`tS` drifts beyond 0.02 s of the golden, remember the live timeline's `tLo` differs
from the batch golden's (retention pruning) — the tolerance accounts for grid shift;
diagnose before touching it.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Stereo/StereoEngine.swift ios/Tests/StereoEngineTests.swift
git commit -m "feat(stereo-swift): live owner-pumped stereo engine"
```

---

### Task 7: Wiring — RecordModel, event relay, bench status

**Files:**
- Modify: `ios/Sources/Record/RecordModel.swift` (extend the peer section),
  `ios/Sources/Peer/PeerBenchView.swift` (DEBUG stereo line)
- Test: `ios/Tests/StereoWiringTests.swift` (append)

**Interfaces:**
- Consumes: `StereoEngine`, `PeerSession` (`onCalibration`, `sendCalibration`,
  `sendControl`-backed `.event` message via a new locked `sendEvent` — see below),
  `RemoteDetectionStore` (existing), `ClockSync.remoteToLocal`.
- Produces:
  ```swift
  // PeerSession (same pattern as sendCalibration):
  func sendEvent(rallyID: UInt32, json: String)         // .ready/.live gated
  var onEvent: ((_ rallyID: UInt32, _ json: String) -> Void)?   // splits .event out of the ignored cases
  // RecordModel:
  func attachStereo(localModelJSON: String)
      // decodes own model; registers peer.onCalibration to decode the
      // remote model and construct the engine (primary role only); chains
      // engine feeding into the EXISTING peer wiring:
      //   - tracker.subscribe closure additionally calls
      //     engine.addLocalObservation(obs, frameW: peerFrameW, frameH: peerFrameH)
      //   - peer.onRemoteDetections additionally calls engine.addRemote(tuples)
      //   - the existing 20 Hz pump additionally calls
      //     engine.processIfDue(now: ClockSync.hostNow())
      // engine impacts: encode {"surface","call","margin_ft","confidence","t_s"}
      // via JSONSerialization -> peer.sendEvent + append to
      // @Published var stereoEvents: [String] (main-hopped, newest-first, cap 20)
  ```
  Secondary side: `peer.onEvent` appends the JSON to the same `stereoEvents` published
  list (mirror display). PeerBenchView (DEBUG): one `Text` row in Status showing
  `model.stereoStatusText` — wire a `@Published var stereoStatusText` into
  `PeerBenchModel` fed from... NOTE: the bench model has no RecordModel; keep the bench
  change MINIMAL: display-only `"stereo: n/a (wired in RecordView)"` placeholder is
  NOT acceptable — instead surface the LAST `.event` JSON received by its session:
  `session.onEvent = { [weak self] _, json in DispatchQueue.main.async { self?.stereoStatusText = json } }`
  set in `start()`, plus the Text row. That exercises the relay end-to-end on hardware
  without owning an engine.

- [ ] **Step 1: Write the failing tests (append to StereoWiringTests)**

```swift
    func testEventMessageRoundTripsThroughSessions() {
        let pair = LoopbackTransport.pair()
        let primary = PeerSession(transport: pair.0, isInitiator: true, now: { 0 })
        let secondary = PeerSession(transport: pair.1, isInitiator: false, now: { 0 })
        secondary.start(); primary.start()
        primary.confirmPairing(); secondary.confirmPairing()
        var t = 0.0
        for _ in 0..<40 { t += 0.1; primary.tick(now: t); secondary.tick(now: t) }

        var received: (UInt32, String)?
        secondary.onEvent = { received = ($0, $1) }
        primary.sendEvent(rallyID: 7, json: "{\"surface\":\"front_wall\"}")
        XCTAssertEqual(received?.0, 7)
        XCTAssertEqual(received?.1, "{\"surface\":\"front_wall\"}")
    }
```

- [ ] **Step 2: Run to verify failure** — `has no member 'sendEvent'`.

- [ ] **Step 3: Implement**

PeerSession: split `.event` out of the ignored-cases branch into
`case .event(let rallyID, let json): onEvent?(rallyID, json)`; add `onEvent` var and:

```swift
    func sendEvent(rallyID: UInt32, json: String) {
        stateLock.lock(); defer { stateLock.unlock() }
        guard internalPhase == .live || internalPhase == .ready else { return }
        sendControl(.event(rallyID: rallyID, json: json))
    }
```

RecordModel `attachStereo(localModelJSON:)` per the Produces block — read the current
peer section first and CHAIN, don't replace: keep existing store/batching behavior
intact, adding engine feeds alongside (the tracker.subscribe single-registration
guard `peerSubscribed` already exists — put the engine's local feed inside that same
closure, gated on `role == .primary` and `stereoEngine != nil`; remote feed goes in
the existing `onRemoteDetections` closure body; pump call in the existing timer).
Engine construction: store `localModel` on attach; in `peer.onCalibration`, decode the
remote model, `guard peer.role == .primary`, create the engine with
`remoteToLocal: { [weak peer] in peer?.clockSync.remoteToLocal($0) }`, set `onEvent`
to encode + `sendEvent(rallyID: 0, json:)` + main-hop append to `stereoEvents`
(cap 20, newest first). Increment a private `rallyID` counter per... keep `rallyID: 0`
constant in Plan B2 (rally segmentation is Phase 4 — comment it).

PeerBenchView: add the Status row `Text("Last stereo event: \(model.stereoStatusText)")`
+ `@Published var stereoStatusText = "—"` + the `session.onEvent` hookup in `start()`.

- [ ] **Step 4: Run tests** — focused 4/4 (wiring file), full suite 72/72.

- [ ] **Step 5: Commit**

```bash
git add ios/Sources/Peer/PeerSession.swift ios/Sources/Record/RecordModel.swift ios/Sources/Peer/PeerBenchView.swift ios/Tests/StereoWiringTests.swift
git commit -m "feat(stereo-swift): wire live engine into record pipeline and event relay"
```

---

## Self-review notes

- **Spec coverage (Phase 3, Swift half):** goldens extended to every confidence tier +
  pair_agreement with embedded timelines (T1); CameraModel decode/undistort/project/ray
  (T2); full StereoMath (T3) and StereoTrack (T4) parity incl. the locality guard and
  ordered surfaces; solved-model exchange (T5); live engine (T6); pipeline wiring +
  event relay + bench surfacing (T7). Remaining Phase 3 spec item NOT here, deliberate:
  the post-rally 3D mini-court replay and audible calls are Phase 4 (spec build order).
- **Numerics contract decisions:** timelines embedded in goldens (no cross-language
  arange); Swift quadratic fit = centered normal equations (documented; tolerance 1e-3 ft
  on trajectory points absorbs solver differences); algebraic tolerance 1e-7.
- **Known simplifications:** rallyID constant 0 until Phase 4 segmentation;
  `no_call` golden case records authority behavior rather than assuming it (T1 step 3
  flags divergence); APIClient method untested at unit level (codebase convention,
  reported).
- **Type consistency check:** `StereoImpact` (not Python's `Impact`) to avoid clashing
  with any future type; field names camelCased consistently; `TrackSample.tS`/`px`
  used identically across T4/T6 tests; `surfaces` array order pinned by test in T3 and
  consumed in T4; DetectionTuple field mapping in T6 matches Task 2 of Plan A
  (x, y raw pixels; ptsNs ns).
