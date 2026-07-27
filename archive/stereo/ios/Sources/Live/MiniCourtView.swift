// ARCHIVED 2026-07-27 -- two-camera stereo/peer feature.
// Excluded from ios/project.yml sources: this file is not compiled.
// Restore point: git tag archive/stereo-v1. See archive/stereo/README.md.
// ios/Sources/Live/MiniCourtView.swift
import SwiftUI
import simd

/// Court-feet → unit-square projection for the post-rally replay.
///
/// Pure, so the geometry is testable without rendering.
///
/// **Court convention** (`StereoMath.planes`, mirrored from `court_model.py`):
/// origin is the front-wall/floor seam at the left corner seen from the back
/// wall; `x` runs right 0…21 ft, `y` runs *from the front wall* 0 → 32 ft
/// toward the back, `z` runs up from the floor. `y = 0` being the **front**
/// wall is the easy thing to get backwards, and getting it backwards mirrors
/// every plot front-to-back and slopes the side-wall out line the wrong way.
///
/// Three conventions the plan and DESIGN.md §8.10 both leave open, fixed here
/// and pinned by tests:
///  1. Output is **screen space** — y grows downward — so a draw site is a
///     bare multiply by the panel size with no second flip.
///  2. `sideElevation`'s vertical extent is `0 ... outLineHeightFt` (15 ft),
///     so the out line lands exactly on the top edge. A ball above the out
///     line clamps to that edge; it is already out, so the exact height stops
///     mattering.
///  3. Both panels read front-to-back in the same direction: plan view puts
///     the front wall at the **top**, side elevation puts it at the **left**.
enum CourtProjection {
    static let maxHeightFt = StereoMath.outLineHeightFt

    /// Plan view from above: x = court width, y = depth, front wall at the top.
    static func planView(_ p: SIMD3<Double>) -> CGPoint {
        CGPoint(x: clamp01(p.x / StereoMath.courtWidthFt),
                y: clamp01(p.y / StereoMath.courtLengthFt))
    }

    /// Side elevation: x = depth (front wall at the left), y = height, flipped
    /// so the floor sits on the bottom edge.
    static func sideElevation(_ p: SIMD3<Double>) -> CGPoint {
        CGPoint(x: clamp01(p.y / StereoMath.courtLengthFt),
                y: clamp01(1.0 - p.z / maxHeightFt))
    }

    /// Unresolved impacts carry raw, unsnapped triangulation that can sit
    /// outside the court entirely — fold it onto the edge rather than let it
    /// draw outside the panel.
    private static func clamp01(_ v: Double) -> CGFloat {
        CGFloat(min(1.0, max(0.0, v)))
    }
}

/// Post-rally replay — DESIGN.md §8.10's court family, extended to a
/// time-based trajectory.
///
/// Two panels, plan view over side elevation. The trajectory is `--dim`; the
/// impact marker takes the verdict colour from `CallPresentation`, so the
/// replay always agrees with the call that was made.
///
/// Per §16 it is never visible before a call — the host reserves its footprint
/// and shows it once a rally resolves, so nothing shifts (§0.9).
struct MiniCourtView: View {
    /// The footprint a host reserves for this view, filled or blank (§0.9).
    ///
    /// Published as a constant rather than left to each host to guess: the
    /// reserved space and the rendered view have to be the *same* height or
    /// the replay draws past its slot and over whatever sits below it. The
    /// value is this view's natural height at the default Dynamic Type size —
    /// two panels of `.caption` (16) + 4 spacing + canvas (92) = 112, plus the
    /// 8 pt gap between them.
    ///
    /// It is not a promise the panels depend on: the canvases are flexible
    /// (`maxHeight: .infinity`), so whatever height a host actually imposes is
    /// divided between the two panels and the labels take the rest. That is
    /// what keeps reserved and rendered identical at *every* type size, not
    /// just the default one — a fixed 92 pt canvas plus a caption that grows
    /// with Dynamic Type cannot do that.
    static let reservedHeight: CGFloat = 232

    let track: [TrackPoint3D]
    let impact: StereoImpact?

    init(track: [TrackPoint3D], impact: StereoImpact?) {
        self.track = track
        self.impact = impact
    }

    private var markerColor: Color {
        impact.map { CallPresentation.from($0).color } ?? Theme.unknown
    }

    var body: some View {
        VStack(spacing: 8) {
            panel(projection: CourtProjection.planView, label: "From above")
            panel(projection: CourtProjection.sideElevation, label: "From the side")
        }
    }

    private func panel(projection: @escaping (SIMD3<Double>) -> CGPoint,
                       label: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            // §12: every chart pairs with a text summary — the panel is never
            // the only representation of what happened.
            Text(label)
                .font(.caption)
                .foregroundStyle(Theme.dim)
            Canvas { context, size in
                let rect = CGRect(origin: .zero, size: size)
                context.stroke(Path(rect), with: .color(Theme.line), lineWidth: 1)

                func place(_ p: SIMD3<Double>) -> CGPoint {
                    let unit = projection(p)
                    return CGPoint(x: rect.minX + unit.x * rect.width,
                                   y: rect.minY + unit.y * rect.height)
                }

                if track.count > 1 {
                    var path = Path()
                    path.move(to: place(track[0].pointFt))
                    for point in track.dropFirst() { path.addLine(to: place(point.pointFt)) }
                    context.stroke(path, with: .color(Theme.dim), lineWidth: 2)
                }

                if let impact {
                    let centre = place(impact.pointFt)
                    let dot = CGRect(x: centre.x - 5, y: centre.y - 5, width: 10, height: 10)
                    context.fill(Path(ellipseIn: dot), with: .color(markerColor))
                }
            }
            // Takes whatever the host reserved, minus the label row — see
            // `reservedHeight`. `Canvas` already accepts the size proposed to
            // it; the explicit `maxHeight` says so at the call site, so this
            // panel is never mistaken for a fixed 92 pt box that a too-small
            // reservation would silently overflow.
            .frame(maxHeight: .infinity)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 8))
        }
    }
}
