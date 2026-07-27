import Foundation
import UIKit

struct FinishedClip: Identifiable {
    let id = UUID()
    let url: URL
    let duration: Double
}

@MainActor
final class RecordModel: ObservableObject {
    let camera = CameraController()
    let tracker: BallTracker

    @Published var trail: [BallObservation] = []
    @Published var isRecording = false
    @Published var recordingStartedAt: Date?
    @Published var errorText: String?
    /// Resolved capture exposure, shown once the court lock lands.
    @Published var exposureNote: String?

    /// The clip the last completed recording produced: non-nil presents
    /// `ResultsView`. Settable so `RecordView`'s `.sheet(item:)` can bind to
    /// it; writing anything but `nil` is refused.
    ///
    /// This used to be an owner-filtered pair (`singleCameraClip` / `liveClip`)
    /// so a live-owned rally could not open the plain judge flow and a plain
    /// recording could not be submitted as paired rally footage. With the live
    /// layer archived (archive/stereo/README.md) there is one consumer, so the
    /// arbitration went with it.
    @Published private(set) var finishedClip: FinishedClip?

    var singleCameraClip: FinishedClip? {
        get { finishedClip }
        set { if newValue == nil { finishedClip = nil } }
    }

    private static let trailLength = 15

    var detectorMissing: Bool { !tracker.isEnabled }

    enum DetectorKind: Equatable { case none, synthetic, model }

    /// `detectorMissing` only knows nil-vs-non-nil, so it cannot tell a real
    /// model from the DEBUG stand-in. The UI must never imply a real detection
    /// when the source is synthetic.
    @Published private(set) var detectorKind: DetectorKind

    /// How this model learns which way the phone is mounted: the interface
    /// orientation mapped to a capture mount, or `nil` when the interface is
    /// not resolved to either landscape — no window scene at all, or a
    /// portrait transient mid-rotation. Both readers are on the recording
    /// path: `startCamera`'s mount seed, and `applyRecording`'s start path,
    /// which refuses to start at all on `nil`.
    ///
    /// A seam rather than a direct call, because the value comes from ambient
    /// UIKit scene state that `RecordModel` would otherwise be reaching out of
    /// itself to read. Every test that starts a recording would then depend on
    /// whether the simulator's rotation had settled by the time XCTest ran it,
    /// and the ones that *wait* on `isRecording` would burn their full
    /// deadline before failing — a confusing failure for a race that has
    /// nothing to do with what they assert. `interfaceMount` below is the
    /// production implementation and the default, so the shipping path is
    /// exactly what it was.
    typealias MountResolver = @MainActor () -> CaptureSettings.CaptureOrientation?

    /// The production mount resolution, and `init`'s default: `OrientationPolicy`'s
    /// two-tier scene lookup mapped through `OrientationLock.captureOrientation(for:)`.
    ///
    /// Two-tier (foreground-active, else any scene) rather than a strict
    /// foreground-active-only check: a transient system interruption (an
    /// alert, Control Center) can leave every scene `.foregroundInactive` for
    /// a moment, and a strict check landing on `nil` there would send
    /// `startCamera`'s first seed straight to its `.landscapeRight` fallback —
    /// indistinguishable from a real landscape-right resolution — and would
    /// make `applyRecording` refuse a start the operator was entitled to.
    ///
    /// Passes `requiresKeyWindow: false`: only `interfaceOrientation` is read
    /// here, and that is available on a scene with no key window yet, unlike
    /// `OrientationPolicy.apply`'s use of the same lookup. One lookup for both
    /// readers rather than two that could quietly diverge — a divergent second
    /// lookup was already a review finding on this branch.
    static func interfaceMount() -> CaptureSettings.CaptureOrientation? {
        OrientationPolicy.activeWindowScene(requiresKeyWindow: false)
            .flatMap { OrientationLock.captureOrientation(for: $0.interfaceOrientation) }
    }

    private let mountResolver: MountResolver

    init(detector: BallDetecting? = CoreMLBallDetector(),
         captureOrientation: CaptureSettings.CaptureOrientation = .landscapeRight,
         mountResolver: @escaping MountResolver = { RecordModel.interfaceMount() }) {
        self.mountResolver = mountResolver
        // Production overwrites this in `startCamera` once the interface
        // orientation is known; the parameter exists so tests can pick a mount
        // without a live window scene.
        self.captureOrientation = captureOrientation
        if detector == nil {
            detectorKind = .none
        } else {
            #if DEBUG
            detectorKind = detector is SyntheticBallDetector ? .synthetic : .model
            #else
            detectorKind = .model
            #endif
        }
        tracker = BallTracker(detector: detector)
        tracker.subscribe { [weak self] observation in
            guard let self else { return }
            trail.append(observation)
            if trail.count > Self.trailLength { trail.removeFirst() }
        }
        // Inference must never block the capture callback: that queue also
        // feeds the AVAssetWriter, and a synchronous Core ML pass (~20-60 ms)
        // would drop recorded frames. Hop to a dedicated queue and skip
        // frames while a detection is in flight — the overlay can afford
        // missed frames, the recording cannot.
        let inferenceQueue = DispatchQueue(label: "slc.record.inference")
        let inFlight = DispatchSemaphore(value: 1)
        camera.onVideoSample = { [tracker] pixelBuffer, timestamp in
            guard inFlight.wait(timeout: .now()) == .success else { return }
            inferenceQueue.async {
                tracker.process(pixelBuffer, timestamp: timestamp)
                inFlight.signal()
            }
        }
    }

    /// The mount this session is currently using. While no recording is
    /// running, `startCamera` freely re-seeds this with an unpinned guess
    /// every time Play appears — there is nothing to protect yet, and the
    /// operator is entitled to re-mount the phone between rallies.
    /// `applyRecording`'s start path is what commits to a mount: it only
    /// assigns this AFTER `camera.startRecording()` actually succeeds, and
    /// only then does the re-seed stop touching it — a tab round trip
    /// mid-recording still re-runs `startCamera`, and would otherwise stomp
    /// the recording's committed mount with a fresh guess. Two things stop
    /// that: `startCamera` bails out whole on `isRecording`, and the re-seed
    /// itself runs on the recording funnel's own transition chain and
    /// re-checks `isRecording` when its turn comes — see `enqueueMountReseed`
    /// for why the chain, not the flag, is what makes it airtight.
    /// The preview overlay and camera-model adoption key off the
    /// `CaptureSettings` statics directly, since both mounts share one frame
    /// size and neither needed the extra indirection.
    @Published private(set) var captureOrientation: CaptureSettings.CaptureOrientation

    /// Guards the *configuration* half of `startCamera()`, and only that half.
    /// One caller now — `RecordView`'s `.task` — but it is still re-entrant:
    /// the Play tab stays alive across tab switches, so returning to it fires
    /// the task again. Configuring is expensive, and re-metering the court
    /// mid-session would change exposure under a recording, so a re-entry must
    /// not re-run it.
    ///
    /// Deliberately does NOT guard the mount seed. Those are two different
    /// concerns and conflating them is a real bug: the seed has to re-run on
    /// every appearance (see `startCamera`).
    private var cameraStarted = false

    func startCamera() async {
        // `isRecording` first, and it bails out whole: `RecordView`'s `.task`
        // re-fires this on every appearance of the Play tab,
        // including a return from Matches/Coach while a recording that
        // `applyRecording`'s start path already resolved and pinned is still
        // running. Re-seeding an unpinned guess below would overwrite the
        // mount a running recording committed to, which is exactly what
        // `captureOrientation`'s doc says stops happening once that path
        // commits. Deliberately does NOT clear `cameraStarted`: the camera is
        // running, and this is a skipped re-entry, not a failed one.
        guard !isRecording else { return }

        // Every appearance after the first re-seeds the mount, and does
        // nothing else. Session configuration is one-shot; the mount is not,
        // and freezing it at the app's first-ever camera start is a bug: the
        // preview and the court-exposure meter would keep working from a
        // mount the operator has since changed.
        //
        // No fallback on this path: a resolution that fails *now* must leave
        // whatever a previous, successful resolution established rather than
        // downgrade it to the `.landscapeRight` literal below. The first seed
        // has no such previous value to keep; this one does.
        guard !cameraStarted else {
            await enqueueMountReseed(fallback: nil).value
            return
        }
        cameraStarted = true      // cleared again in the `catch`, so a failure retries
        do {
            // The first seed, before `configure()` — `configureSession()`
            // reads `camera.orientation` to set the video connection's initial
            // rotation, so this has to have landed by then, which is what
            // awaiting the chained transition guarantees.
            //
            // Not for the preview below — a separate `AVCaptureVideoPreviewLayer`
            // connection this never touches (`CameraPreviewView.swift`), and
            // not for the court-exposure meter either, which never reads
            // `orientation` (see `lockForCourt()`). This is so the
            // recorded/streamed frame's rotation, and the mount this model
            // resolves, do not go stale.
            // Deliberately NOT pinned, here or on any later re-seed: the Play
            // tab stays at both-landscape
            // (`.landscape`) through the whole framing window, which is what
            // lets the operator flip the mount before recording starts.
            // `applyRecording`'s start path is where the mount actually gets
            // committed and pinned — see there for why record start, not
            // camera start, has to be the point of no return.
            //
            // `.landscapeRight` is a last resort only, for the genuinely-no-
            // scene case (or a portrait interface, which is not a capture
            // mode): a fallback default, not a resolved orientation. It costs
            // only a wrong seeded orientation, since nothing here pins;
            // `applyRecording`'s own resolution at record start is what has to
            // get the real mount right, and it refuses rather than falling
            // back at all.
            await enqueueMountReseed(fallback: .landscapeRight).value

            // `CameraController.configure()` is repeatable: it tears its own
            // inputs/outputs down before re-adding them, so re-entering here
            // after a mid-configuration failure below really does retry
            // rather than throwing `configurationFailed` forever.
            try await camera.configure()
            camera.start()
            // Meter the court once from the mounted position, then freeze
            // exposure/WB/focus for the session. Everything after this point
            // is shot under identical conditions, which is what keeps the
            // footage usable as training data.
            exposureNote = CaptureSettings.summary(for: try await camera.lockForCourt())
        } catch {
            cameraStarted = false      // let a retry re-enter
            errorText = error.localizedDescription
        }
    }

    // MARK: - The one recording funnel

    /// Serializes every start/stop against the camera — and every mount
    /// re-seed — in the order they were issued. Held here, not in any
    /// consumer, because the camera is here.
    private var recordingTransition: Task<Bool, Never>?

    /// Re-resolves the mount from the interface and applies it to the camera,
    /// serialized against every recording start and stop.
    ///
    /// Being *on this chain* is the whole safety argument, and a flag would
    /// not do. `applyRecording`'s start path resolves a mount, awaits
    /// `camera.updateOrientation(mount)`, opens an `AVAssetWriter` against
    /// that rotation, and only then assigns `captureOrientation` and pins the
    /// mask. `isRecording` is still `false` for all of that, and the
    /// `updateOrientation` await is a real suspension point — so a re-seed
    /// checking `isRecording` alone could interleave right there and leave the
    /// video connection rotated for one mount while the recording it is
    /// feeding was committed and pinned to the other. Chained, a re-seed runs
    /// strictly before or strictly after a transition and never inside one.
    ///
    /// `fallback` is what to use when the interface cannot be resolved to a
    /// mount: `.landscapeRight` for `startCamera`'s first seed, which has
    /// nothing better, and `nil` for every later re-seed, which does — the
    /// mount an earlier successful resolution already established.
    ///
    /// Reports `false` unconditionally. The chain is typed by the recording
    /// funnel, whose `Bool` means "did this start or stop a recording"; a
    /// re-seed never does either. Callers await it to sequence, not for the
    /// value.
    ///
    /// Not `@discardableResult`: both call sites await `.value`, so the
    /// annotation had no caller to protect.
    private func enqueueMountReseed(fallback: CaptureSettings.CaptureOrientation?) -> Task<Bool, Never> {
        let previous = recordingTransition
        let next = Task { [weak self] in
            await previous?.value
            guard let self else { return false }
            await self.applyMountReseed(fallback: fallback)
            return false
        }
        recordingTransition = next
        return next
    }

    /// Runs only from the chain above, so `isRecording` read here is the
    /// camera's state *after* everything issued earlier landed.
    private func applyMountReseed(fallback: CaptureSettings.CaptureOrientation?) async {
        // Re-checked here, at execution time, never in advance — the same rule
        // `canSetRecording` states for the recording transitions this shares a
        // chain with. A recording running by the time this gets its turn owns
        // the mount outright: it resolved one, rotated the connection to it,
        // pinned the orientation mask to it, and is writing frames under it.
        // `isRecording` also stays true across the whole stop path — it is
        // cleared only after `camera.stopRecording()` has returned — so this
        // cannot land mid-stop either, when the writer is still finishing.
        guard !isRecording else { return }
        guard let mount = mountResolver() ?? fallback else { return }
        // Routed through `updateOrientation` rather than a direct
        // `camera.orientation = ...` assignment, so `orientation` has one
        // writer, always on `sessionQueue`, instead of a direct main-actor
        // write racing `applyRecording`'s queued one. Before `configure()` has
        // run there is no connection yet, so it simply records the value and
        // cannot fail; afterwards it rotates the live connection — see
        // `updateOrientation(_:)`'s doc for both branches.
        //
        // Checked, not just awaited — same shape as `applyRecording`'s start
        // path below, and for the same reason: an unsupported angle leaves
        // the connection (and `camera.orientation`) at the previous mount,
        // and committing `captureOrientation` to the requested mount anyway
        // would describe a mount the connection does not actually have —
        // the divergence this whole change exists to prevent. `false` here
        // has no caller to report to, so it is simply not committed rather
        // than surfaced as an error: a re-seed is a background refresh, not
        // a user-initiated transition.
        guard await camera.updateOrientation(mount) else { return }
        captureOrientation = mount
    }

    /// The whole rule, written once: start only when nothing is recording,
    /// stop only when something is.
    ///
    /// `applyRecording` enforces exactly this predicate, and `RecordView` asks
    /// exactly this predicate before offering its button — so what the screen
    /// offers and what the model will do cannot drift apart.
    ///
    /// Evaluate it at the moment the transition would run, never in advance:
    /// a queued transition ahead of this one can change the answer.
    ///
    /// This used to take an `owner` (the record stage vs. the archived live
    /// layer) and arbitrate between two consumers of one camera. One consumer
    /// remains, so the predicate is just the camera's state.
    func canSetRecording(_ shouldRecord: Bool) -> Bool {
        shouldRecord != isRecording
    }

    /// The single funnel. Chains onto any in-flight transition, re-reads the
    /// camera's real state at execution time, performs the start or stop only
    /// if that is actually the needed transition, and returns whether it acted.
    ///
    /// Absolute (`shouldRecord`), never a toggle: a toggle issued against a
    /// camera state that already contradicts the intent flips it the wrong
    /// way, which is how a stop could once turn into a start.
    @discardableResult
    func setRecording(_ shouldRecord: Bool) async -> Bool {
        await enqueueSetRecording(shouldRecord).value
    }

    /// `setRecording`'s synchronous face, for callers that must fire and
    /// forget but still need the order fixed at *issue* time. Wrapping
    /// `setRecording` in an unstructured `Task` at the call site would not do:
    /// the enqueue would then happen whenever that task got scheduled, so two
    /// calls could reach the chain in the opposite order from the one they
    /// were made in.
    func enqueueSetRecording(_ shouldRecord: Bool) -> Task<Bool, Never> {
        let previous = recordingTransition
        let next = Task { [weak self] in
            await previous?.value
            guard let self else { return false }
            return await self.applyRecording(shouldRecord)
        }
        recordingTransition = next
        return next
    }

    /// Runs only from the chain above, so `isRecording` read here is the
    /// camera's state *after* everything issued earlier landed.
    ///
    /// This is also where the capture mount is committed and the orientation
    /// mask pinned — see the start path below. Both halves of that (resolve,
    /// then pin only once `startRecording()` succeeded) came from the
    /// landscape-only capture change and are unchanged by the serialization
    /// funnel wrapped around them; the only difference is that a refusal
    /// returns `false` to the caller instead of returning silently.
    private func applyRecording(_ shouldRecord: Bool) async -> Bool {
        guard canSetRecording(shouldRecord) else { return false }
        if shouldRecord {
            // Re-resolve the mount from the CURRENT interface orientation
            // here, at record start — not whatever `startCamera` last seeded —
            // because the window between camera start and record start is
            // exactly when the operator is meant to be able to flip the mount.
            // Reads `mountResolver`, the same seam `startCamera`'s seed reads,
            // rather than a second lookup of its own: a divergent second
            // lookup was already a review finding on this branch, and in a
            // test host this is also the only way to make a start deterministic
            // (see `MountResolver`).
            guard let mount = mountResolver() else {
                // The interface isn't resolved to either landscape mount
                // right now — no scene, or a portrait transient mid-rotation.
                // Pinning a guessed mount here is exactly the failure this
                // change exists to prevent, so refuse the start rather than
                // falling back to a default. `false`, so a live caller's
                // `reconcileRallyAfterStartAttempt()` turns it into a visible
                // `.failed` rally rather than a silent no-rally.
                errorText = "Hold the phone in landscape to start recording."
                return false
            }
            // Remembered so a failed attempt below can put the connection
            // back rather than leave it describing a mount no recording ever
            // committed to. Read straight from `camera`, not from
            // `captureOrientation`: this model property is intentionally not
            // updated until a recording actually starts (see below), so
            // before the first successful recording it can already differ
            // from what the connection is currently rotated to.
            let priorOrientation = camera.orientation
            // Must land, and actually apply to the live connection, before
            // startRecording() below sizes the asset writer: getting the
            // order wrong would leave the connection rotated for the old
            // mount while the writer already assumes the new one. Checked,
            // not just awaited — an unsupported angle leaves the connection
            // (and `camera.orientation`) at the previous mount, and pinning
            // or recording against that silently-still-wrong rotation would
            // be exactly the divergence this change exists to prevent.
            guard await camera.updateOrientation(mount) else {
                errorText = "Could not switch the camera to this mount."
                return false
            }
            do {
                try camera.startRecording()
                // Only commit `captureOrientation` and the pin once the
                // recording has actually started. Doing either before this
                // point and having `startRecording()` then throw would
                // strand the operator locked to this one mount with no
                // running recording to ever release it from (the stop path
                // below, and so `releaseCapturePin()`, is only reachable
                // while `isRecording` is true) — the same one-shot-per-launch
                // failure this whole change exists to remove, just relocated
                // to the failure path instead of removed.
                captureOrientation = mount
                OrientationPolicy.shared.pinForCapture(OrientationLock.pinnedMask(for: mount))
                isRecording = true
                recordingStartedAt = Date()
                errorText = nil
                return true
            } catch {
                // The recording never actually started: nothing above was
                // committed (captureOrientation/pin are untouched), but the
                // connection's rotation already changed. Put it back so
                // `camera.orientation` doesn't keep describing a mount no
                // recording committed to.
                await camera.updateOrientation(priorOrientation)
                // Nothing started, so nothing is owned. The caller learns this
                // from the `false`; the live layer turns it into a visible
                // `.failed` rally via `reconcileRallyAfterStartAttempt()`.
                errorText = error.localizedDescription
                return false
            }
        }
        // Whether the write below finishes cleanly or fails, this recording is
        // over — release the pin so the operator can re-mount before the next
        // rally. Ahead of `stopRecording()`, not after it: the pin must lift on
        // the failure path too, and `stopRecording()` throws.
        OrientationPolicy.shared.releaseCapturePin()
        do {
            let url = try await camera.stopRecording()
            let duration = recordingStartedAt.map { Date().timeIntervalSince($0) } ?? 0
            isRecording = false
            recordingStartedAt = nil
            finishedClip = FinishedClip(url: url, duration: duration)
            return true
        } catch {
            // The recording is over either way (`stopRecording()` clears the
            // writer before it can throw), so this counts as having acted —
            // there is just no clip to hand anyone.
            isRecording = false
            recordingStartedAt = nil
            errorText = error.localizedDescription
            return true
        }
    }

}
