"""Parametric STL generator for the squash fin mount — v2 (tripod-clamp revision).

Source of truth: docs/mount-spec.md + docs/superpowers/specs/
2026-07-23-mount-v2-clamp-redesign-design.md. All dimensions mm.

v2 replaces the custom phone cradle with the Printables "Tripod Mobile Phone
Clamp V2" (Stamos, CC BY-NC-SA, hardware/third_party/): the clamp's LOWER part
is remixed — its tripod foot underside (arca bevels + metal-nut cavity) is
swallowed by a grafted block carrying a T-slot that slides onto a T-rail on the
wedge face (end stop + tapered peg, same UX as the saddle/mast joints that
passed the 2026-07-23 hardware test). Wedge faces are re-derived: the clamp bar
mounts perpendicular to the face, so face inclination = optical pitch directly
and the face descends toward the wall.

Field-test fixes baked in:
- wall-stop bosses / gap fillers deleted from the saddle (position by hand +
  screw friction);
- all rail/stop unions embedded >=2 mm (the v1 0.05 mm embedment made the
  wedge-face dovetails degenerate);
- clamp screw ball tip replaced by a threaded-on tip disc (glued = captive);
- every peg bore modeled open and asserted open by containment sampling
  (the saddle cap bore stays blind BY DESIGN — it must never reach the glass);
- mast height / mast position / per-wedge rail position solved against the
  WSF court-plane rule and the edge-occlusion ratio, then asserted.

Every STL is exported ALREADY POSED FOR PRINTING — no rotation in the slicer.

    python3 hardware/generate_mount_stls.py

Outputs to hardware/stl/. Requires: trimesh, manifold3d, shapely, numpy,
scipy, networkx, rtree.
"""
import numpy as np
import trimesh
from trimesh.creation import box as _box, cylinder as _cyl, extrude_polygon
from trimesh.transformations import rotation_matrix
from shapely.geometry import Polygon
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "stl"
THIRD = HERE / "third_party"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- parameters
T_F = 12.0        # fin/wall glass thickness
WALL_T_GLASS = 12.0   # back-wall glass thickness (court-side face offset)
CHANNEL = T_F + 3.5          # 15.5 between PETG jaw faces
JAW_WALL = 8.0
JAW_DEPTH = 80.0
SADDLE_LEN = 130.0
CAP_T = 15.0
CAP_HALF_W = 20.0
SEAT_PROUD = 0.5
SEAT_T, SEAT_RECESS = 3.0, 2.5            # TPU seat strip, 0.5 proud
PAD_T, PAD_RECESS = 4.0, 2.5              # TPU jaw pads, 1.5 proud
SCREW_X, SCREW_Z = 65.0, -45.0
# printed clamp thread (coarse prints anywhere)
THR_MAJ, THR_MIN, THR_PITCH = 6.0, 4.75, 2.5   # radii: M12x2.5-ish
THR_CLEAR = 0.4                                 # radial nut clearance
DT_CLEAR = 0.4                                  # dovetail clearance (relaxed)
PIN_R = 2.6                                     # peg bore radius (pin max 2.6->2.3 taper)
WEDGE_ANGLES = {"A_uw169_40deg": 40.0, "B_uw43_32deg": 32.0}

# T-joint wedge<->clamp (all faces vertical or small designed bridges)
T_STEM_W, T_STEM_H = 14.0, 3.3      # rail stem width (slope dir) x height
T_HEAD_W, T_HEAD_H = 26.0, 3.0      # rail head width x height
T_EMBED = 2.0                        # rail/stop embedment into the wedge body
RAIL_Y0, RAIL_Y1 = -15.0, 9.0        # rail span across the wedge (wedge y)
STOP_Y0, STOP_Y1 = 15.0, 19.0        # end-stop tab span
WEDGE_HALF_W = 19.0                  # wedge body y half-width (was 15 in v1)
PIN_YW = -17.0                       # wedge-clamp peg, beside the rail start

# clamp bar facts measured from the Printables lower mesh (asserted at load)
BAR_END_Y = 30.0          # screw tip / jaw pocket floor plane
FOOT_Y1 = 37.75           # stock foot underside extreme
BLOCK_X0, BLOCK_X1 = -19.0, 15.0     # graft block, maps 1:1 onto wedge y
BLOCK_Y0 = 33.0                      # graft overlap start (inside solid foot)
BLOCK_Z0, BLOCK_Z1 = 0.0, 34.0       # graft block depth (print z)
SLOT_ZC = 17.0                       # T-slot center (print z)
CEIL_T = 4.0                         # slot ceiling beyond stock foot
D_STEM = T_STEM_H - 0.3   # slot stem depth: SHALLOWER than the rail stem, so
D_HEAD = T_HEAD_H + 0.7   # the head's underside clears the block shoulder;
                          # crest gap 0.4 (mouth face seats on the wedge face)
SLOT_MOUTH_Y = FOOT_Y1 + CEIL_T + D_STEM + D_HEAD    # 48.45
PHONE_BACK_Z = 4.1        # phone screen rests on the bar top
PHONE_T_MIN = 8.0         # thinnest phone (worst case for edge occlusion)
PW_MIN, PW_MAX = 74.0, 82.0   # phone width across the jaws (site-measure, spec s5)
LENS_FROM_TOP = 22.0      # ultrawide lens center from phone top edge (spec E)
JAW_OVER = 12.0           # upper-jaw hardware beyond the phone top edge
HW_FWD_Z = 27.3           # most forward hardware depth (thicker-variant jaw hooks)

# site / rules constants (spec section 8)
COURT_MARGIN = 3.0        # topmost hardware stays >=3 outboard of the wall OUTER face
OCCLUSION_RATIO = 4.92    # lens height / setback-from-court-side-face (tan 78.5deg)

HALF_CH = CHANNEL / 2.0
CAP_TOP = SEAT_PROUD + CAP_T             # 15.5
BOSS_Y = HALF_CH + JAW_WALL              # outer jaw face


def box(x0, x1, y0, y1, z0, z1):
    b = _box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return b


def cyl(radius, p0, p1, sections=64):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    vec = p1 - p0
    h = np.linalg.norm(vec)
    c = _cyl(radius=radius, height=h, sections=sections)
    c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], vec / h))
    c.apply_translation((p0 + p1) / 2)
    return c


def prism_xz(points, y0, y1):
    """Extrude an XZ polygon along Y."""
    p = extrude_polygon(Polygon(points), y1 - y0)
    p.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, y0], [0, 1, 0, 0],
                                [0, 0, 0, 1]], float))
    return p


def prism_yz(points, x0, x1):
    """Extrude a YZ polygon along X."""
    p = extrude_polygon(Polygon(points), x1 - x0)   # built in XY, extruded +Z
    p.apply_transform(np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 0, 1]], float))
    return p


def threaded_rod(r_maj, r_min, pitch, length, sections=64, rows_per_pitch=8):
    """Watertight helical-thread rod along +Z from z=0 (triangular profile)."""
    turns = length / pitch
    m = max(int(turns * rows_per_pitch), 4)
    zs = np.linspace(0, length, m + 1)
    th = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    verts, faces = [], []
    for z in zs:
        t = (z / pitch - th / (2 * np.pi)) % 1.0
        r = r_min + (r_maj - r_min) * (1 - np.abs(2 * t - 1))
        verts.append(np.column_stack([r * np.cos(th), r * np.sin(th),
                                      np.full(sections, z)]))
    verts = np.vstack(verts)
    for i in range(m):
        for j in range(sections):
            a = i * sections + j
            b = i * sections + (j + 1) % sections
            c = (i + 1) * sections + j
            d = (i + 1) * sections + (j + 1) % sections
            faces += [[a, b, d], [a, d, c]]
    lo = len(verts); verts = np.vstack([verts, [[0, 0, 0]]])
    hi = len(verts); verts = np.vstack([verts, [[0, 0, length]]])
    for j in range(sections):
        faces.append([lo, (j + 1) % sections, j])
        faces.append([hi, m * sections + j, m * sections + (j + 1) % sections])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    assert mesh.is_watertight
    return mesh


def union(parts):
    return trimesh.boolean.union(parts, engine="manifold")


def diff(base, cuts):
    return trimesh.boolean.difference([base] + cuts, engine="manifold")


def lie_on_minus_y(mesh):
    """Print pose for parts whose features are all extrusions along Y."""
    m = mesh.copy()
    m.apply_transform(rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def flip_upside_down(mesh):
    m = mesh.copy()
    m.apply_transform(rotation_matrix(np.pi, [1, 0, 0]))
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def drop_to_plate(mesh):
    m = mesh.copy()
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def keep_pose(mesh):
    return mesh.copy()


def save(mesh, name, orient=drop_to_plate):
    mesh = orient(mesh)
    assert mesh.is_watertight, f"{name} not watertight"
    assert mesh.volume > 0, f"{name} zero volume"
    assert abs(mesh.bounds[0][2]) < 1e-6, f"{name} not on the build plate"
    mesh.export(OUT / f"{name}.stl")
    ext = np.round(mesh.extents, 1)
    print(f"{name:22s} extents {ext} mm  volume {mesh.volume/1000:.1f} cm3")
    return mesh


def assert_bore_open(mesh, p0, p1, name, samples=15):
    """Every sampled point along the bore axis must be OUTSIDE the solid."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    ts = np.linspace(0.05, 0.95, samples)[:, None]
    pts = p0 + ts * (p1 - p0)
    inside = mesh.contains(pts)
    assert not inside.any(), (
        f"{name}: bore {p0}->{p1} blocked at t="
        f"{ts[inside.nonzero()[0][0], 0]:.2f}")


# ------------------------------------------------- placement solver (spec s8)
# World frame: X along the fin, 0 at the wall OUTER face, +X away from the
# wall (court side is X < -WALL_T_GLASS behind the court-side face at -12).
# Z vertical, 0 at the fin/wall top edge (2130 AFF). Saddle slid flush: X 0..130.
#
# Wedge local x=0 sits at world X = x_s + 5 (mast base at saddle x_s, wedge
# local = mast local - 5). Wedge face: F(s) = (x_s+5 + s cos(th),
# CAP_TOP + MAST_H + 10 + s sin(th)), s = slope coordinate from the wall-side
# (low) corner, ascending away from the wall. The clamp bar mounts with
# print +z pointing DOWN-slope (camera direction) and print -y up the bar.
# A clamp print point (y_p, z_p) maps to world:
#   h = SLOT_MOUTH_Y - y_p                (offset along the face normal)
#   s = s_lo + (BLOCK_Z1 - z_p)           (block spans s_lo .. s_lo+34)
#   X = x_s+5 + s cos(th) - h sin(th)
#   Z = CAP_TOP + MAST_H + 10 + s sin(th) + h cos(th)

def clamp_point_world(x_s, mast_h, s_lo, th_deg, y_p, z_p):
    th = np.radians(th_deg)
    h = SLOT_MOUTH_Y - y_p
    s = s_lo + (BLOCK_Z1 - z_p)
    X = x_s + 5 + s * np.cos(th) - h * np.sin(th)
    Z = CAP_TOP + mast_h + 10 + s * np.sin(th) + h * np.cos(th)
    return X, Z


def placement_metrics(x_s, mast_h, s_lo, th_deg):
    """Worst-case court-plane margin and occlusion ratio over the phone range."""
    # topmost hardware: upper-jaw structure JAW_OVER beyond the widest phone's
    # top edge, at the most forward hardware depth (jaw hooks)
    y_top = BAR_END_Y - PW_MAX - JAW_OVER
    Xc, _ = clamp_point_world(x_s, mast_h, s_lo, th_deg, y_top, HW_FWD_Z)
    margin = Xc - 0.0                       # vs wall OUTER face plane at X=0
    # occlusion: worst at the narrowest, thinnest phone (lens lowest, farthest out)
    y_lens = BAR_END_Y - PW_MIN + LENS_FROM_TOP
    Xl, Zl = clamp_point_world(x_s, mast_h, s_lo, th_deg, y_lens,
                               PHONE_BACK_Z + PHONE_T_MIN)
    ratio = Zl / (Xl + WALL_T_GLASS)        # vs court-side top corner
    return margin, ratio


def solve_placement():
    """Min mast height satisfying both constraints for both wedges."""
    best = None
    s_max = {n: 42.0 / np.cos(np.radians(a)) - (BLOCK_Z1 - BLOCK_Z0)
             for n, a in WEDGE_ANGLES.items()}
    for mast_h in np.arange(60, 180, 2.0):
        for x_s in np.arange(20, 70.5, 1.0):
            ok, s_los = True, {}
            for name, ang in WEDGE_ANGLES.items():
                good = None
                for s_lo in np.arange(0, s_max[name] + 1e-6, 0.5):
                    m, r = placement_metrics(x_s, mast_h, s_lo, ang)
                    if m >= COURT_MARGIN and r >= OCCLUSION_RATIO:
                        good = s_lo
                        break           # smallest s_lo -> most margin later
                if good is None:
                    ok = False
                    break
                s_los[name] = good
            if ok:
                best = (x_s, mast_h, s_los)
                break
        if best:
            break
    assert best, "no feasible placement — check spec s8 parameters"
    x_s, mast_h, s_los = best
    print(f"placement: mast_h={mast_h:.0f}  x_s={x_s:.1f}")
    for name, ang in WEDGE_ANGLES.items():
        m, r = placement_metrics(x_s, mast_h, s_los[name], ang)
        print(f"  wedge {name}: s_lo={s_los[name]:.2f}  "
              f"court-margin={m:.1f}mm  occlusion h/s={r:.2f}")
        assert m >= COURT_MARGIN and r >= OCCLUSION_RATIO
    return x_s, mast_h, s_los


X_S, MAST_H, S_LO = solve_placement()
RAIL_S = {n: S_LO[n] + (BLOCK_Z1 - SLOT_ZC) for n in WEDGE_ANGLES}  # rail center


# ------------------------------------------------------- 1. saddle
# Design frame: X along fin (0 = wall end), Y across fin, Z from fin top edge.
# PRINTS UPSIDE-DOWN (cap top on the plate, jaws opening up). v2: no wall
# bosses / stops / wall pads (field decision F4) — plain channel, position by
# hand, screw friction holds. Cap top carries the mast slot + peg bore (bore
# blind BY DESIGN: it must never break into the glass channel).
def make_saddle():
    cap = box(0, SADDLE_LEN, -CAP_HALF_W, CAP_HALF_W, SEAT_PROUD, CAP_TOP)
    jaw_n = box(0, SADDLE_LEN, -HALF_CH - JAW_WALL, -HALF_CH, -JAW_DEPTH, SEAT_PROUD)
    jaw_p = box(0, SADDLE_LEN, HALF_CH, BOSS_Y, -JAW_DEPTH, SEAT_PROUD)
    screw_boss = cyl(13, [SCREW_X, BOSS_Y - 0.1, SCREW_Z],
                     [SCREW_X, BOSS_Y + 14, SCREW_Z])
    # true 45-degree teardrop roof so the flipped print is support-free
    td = 13 * 1.4142
    teardrop = prism_xz([(SCREW_X - td, SCREW_Z), (SCREW_X + td, SCREW_Z),
                         (SCREW_X, SCREW_Z + td)], BOSS_Y - 0.1, BOSS_Y + 14)
    body = union([cap, jaw_n, jaw_p, screw_boss, teardrop])

    nut = threaded_rod(THR_MAJ + THR_CLEAR, THR_MIN + THR_CLEAR, THR_PITCH, 24)
    nut.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
    nut.apply_translation([SCREW_X, BOSS_Y + 14.1 - 24, SCREW_Z])
    # mast slot in cap top: mouth 20.8 at the surface flaring to 26.8 at -6.3,
    # along Y, stop at +Y end; centered on the solved mast position X_S
    slot = prism_xz([(X_S + 17.1, CAP_TOP + 0.1), (X_S + 37.9, CAP_TOP + 0.1),
                     (X_S + 40.9, CAP_TOP - 6.3), (X_S + 14.1, CAP_TOP - 6.3)],
                    -CAP_HALF_W - 0.1, 15.0)
    pin_top, pin_bot = CAP_TOP + 0.1, 6.0    # blind: never reaches the channel
    cuts = [
        box(9.75, 120.25, -6.2, 6.2, SEAT_PROUD, SEAT_PROUD + SEAT_RECESS),
        box(10, 120, -HALF_CH - PAD_RECESS, -HALF_CH + 0.1, -70, -10),
        box(10, 120, HALF_CH - 0.1, HALF_CH + PAD_RECESS, -70, -10),
        nut,
        cyl(10, [SCREW_X, HALF_CH - 0.1, SCREW_Z], [SCREW_X, HALF_CH + 6, SCREW_Z]),
        slot,
        cyl(PIN_R, [X_S + 55, 0, pin_bot], [X_S + 55, 0, pin_top]),  # mast peg
    ]
    m = diff(body, cuts)
    assert_bore_open(m, [X_S + 55, 0, pin_bot + 0.2], [X_S + 55, 0, CAP_TOP],
                     "saddle mast-peg bore")
    # thread bore open all the way through the boss
    assert_bore_open(m, [SCREW_X, HALF_CH + 1, SCREW_Z],
                     [SCREW_X, BOSS_Y + 13.5, SCREW_Z], "saddle thread bore")
    return save(m, "saddle", orient=flip_upside_down)


# ------------------------------------------------------- 2. mast
# Base rail slides into the cap slot (pegged); column rises to the wedge rail.
# v2: column height solved by placement (MAST_H); wedge-peg pocket moved to
# x=10.5 (beside the new low side of the wedge). PRINTS LYING ON ITS SIDE.
def make_mast():
    top = MAST_H                     # column top face, cap-top-relative
    base = box(0, 60, -15, 15, 0, 8)
    under_rail = prism_xz([(17.5, 0.1), (37.5, 0.1), (40.5, -6.0), (14.5, -6.0)],
                          -15, 15)
    col = box(5, 50, -15, 15, 8, top)
    # male rail root 20 (17.5..37.5), crest 26 (14.5..40.5), embedded T_EMBED
    top_rail = prism_xz([(17.5, top - T_EMBED), (37.5, top - T_EMBED),
                         (37.5, top), (40.5, top + 6), (14.5, top + 6),
                         (17.5, top)], -15, 15)
    # end stop overlaps the rail 2 mm in y (v1 attached it face-on-face)
    stop = prism_xz([(14.5, top - T_EMBED), (40.5, top - T_EMBED),
                     (40.5, top + 6), (14.5, top + 6)], 13, 17)
    body = union([base, under_rail, col, top_rail, stop])
    cuts = [
        cyl(PIN_R, [10.5, 0, top - 16], [10.5, 0, top + 0.1]),  # wedge peg pocket
        cyl(PIN_R, [55, 0, -6.1], [55, 0, 8.1]),                # saddle peg (through)
    ]
    m = diff(body, cuts)
    assert_bore_open(m, [55, 0, -5.9], [55, 0, 7.9], "mast saddle-peg bore")
    assert_bore_open(m, [10.5, 0, top - 15.8], [10.5, 0, top - 0.2],
                     "mast wedge-peg pocket")
    return save(m, "mast", orient=lie_on_minus_y)


# ------------------------------------------------------- 3. wedges
# v2 face: inclination = optical pitch, DESCENDING toward the wall (local x=0).
# The clamp bar mounts perpendicular to the face, so the camera axis runs
# down-slope at exactly the pitch angle. Face carries a T-rail (stem 14x3.3,
# head 26x3.0, embedded 2) + end-stop tab + two peg bores. PRINTS LYING ON
# ITS SIDE (every feature is a Y-extrusion), support-free.
def make_wedge(name, pitch_deg):
    th = np.radians(pitch_deg)
    e_s = np.array([np.cos(th), 0, np.sin(th)])       # up-slope, away from wall
    e_n = np.array([-np.sin(th), 0, np.cos(th)])      # face normal
    f0 = np.array([0.0, 0, 10.0])                     # low (wall-side) corner
    h_hi = 10.0 + 42.0 * np.tan(th)
    body = prism_xz([(0, 0), (42, 0), (42, h_hi), (0, 10)],
                    -WEDGE_HALF_W, WEDGE_HALF_W)
    # bottom slot rides the mast rail (crest cx 22.5 local), cut through
    c, hh = DT_CLEAR, 6 + DT_CLEAR
    slot = prism_xz([(22.5 - 10 - c, -0.1), (22.5 + 10 + c, -0.1),
                     (22.5 + 13 + c, -0.1 + hh), (22.5 - 13 - c, -0.1 + hh)],
                    -WEDGE_HALF_W - 0.2, WEDGE_HALF_W + 0.2)

    def face_quad(s0, s1, n0, n1):
        pts = [f0 + s0 * e_s + n0 * e_n, f0 + s1 * e_s + n0 * e_n,
               f0 + s1 * e_s + n1 * e_n, f0 + s0 * e_s + n1 * e_n]
        return [(p[0], p[2]) for p in pts]

    sc = RAIL_S[name]
    stem = prism_xz(face_quad(sc - T_STEM_W / 2, sc + T_STEM_W / 2,
                              -T_EMBED, T_STEM_H), RAIL_Y0, RAIL_Y1)
    head = prism_xz(face_quad(sc - T_HEAD_W / 2, sc + T_HEAD_W / 2,
                              T_STEM_H - 0.1, T_STEM_H + T_HEAD_H),
                    RAIL_Y0, RAIL_Y1)
    stop = prism_xz(face_quad(sc - T_HEAD_W / 2 - 2, sc + T_HEAD_W / 2 + 2,
                              -T_EMBED, 8.0), STOP_Y0, STOP_Y1)
    body = union([body, stem, head, stop])

    # mast peg: the wedge sits shifted -6 on the mast (its +y face rides to the
    # mast stop at y 13), so the mast-frame y=0 pocket is wedge-local y=+6
    p_pin = f0 + sc * e_s + np.array([0, PIN_YW, 0])
    h_at = 10 + 5.5 * np.tan(th)
    cuts = [slot,
            cyl(PIN_R, p_pin + 8 * e_n, p_pin - 12 * e_n),   # clamp peg bore
            cyl(PIN_R, [5.5, 6, -0.1], [5.5, 6, h_at + 0.1])]
    m = diff(body, cuts)
    assert_bore_open(m, p_pin + 0.2 * e_n, p_pin - 11.8 * e_n,
                     f"wedge {name} clamp-peg bore")
    assert_bore_open(m, [5.5, 6, 0.2], [5.5, 6, h_at - 0.2],
                     f"wedge {name} mast-peg bore")
    return save(m, f"wedge_{name}", orient=lie_on_minus_y), m


# ------------------------------------------------------- 4. clamp lower mount
# Remix of the Printables lower part (CC BY-NC-SA, Stamos): a solid block is
# grafted over the tripod foot's underside — swallowing the arca bevels, the
# 1/4-20 hole and the metal-nut cavity (no non-printable parts) — and carries
# the T-slot + peg bore. The foot plate above the graft is untouched: it is
# the lower jaw's backing. Prints in the stock orientation (flat, z up);
# slot ceilings are 3.2-3.5 mm designed bridges.
def make_clamp_lower_mount(thicker=False):
    src = THIRD / ("updated_phone_clamp_lower_thicker.stl" if thicker
                   else "updated_phone_clamp_lower.stl")
    lower = trimesh.load_mesh(src)
    b = lower.bounds
    assert abs(b[1][1] - FOOT_Y1) < 0.1 and abs(b[0][1] + 29.5) < 0.1, \
        "unexpected clamp lower mesh — re-measure BAR/FOOT constants"

    blockx = box(BLOCK_X0, BLOCK_X1, BLOCK_Y0, SLOT_MOUTH_Y, BLOCK_Z0, BLOCK_Z1)
    body = union([lower, blockx])

    # T-slot: axis along print X (through), mouth on the y+ face, flare in Z
    sw, hw = T_STEM_W / 2 + DT_CLEAR, T_HEAD_W / 2 + DT_CLEAR
    y_stem = SLOT_MOUTH_Y - D_STEM                    # 45.45
    y_head = y_stem - D_HEAD                          # 41.75
    slot_pts = [(y_stem, SLOT_ZC - sw), (SLOT_MOUTH_Y + 0.1, SLOT_ZC - sw),
                (SLOT_MOUTH_Y + 0.1, SLOT_ZC + sw), (y_stem, SLOT_ZC + sw),
                (y_stem, SLOT_ZC + hw), (y_head, SLOT_ZC + hw),
                (y_head, SLOT_ZC - hw), (y_stem, SLOT_ZC - hw)]
    slot = prism_yz(slot_pts, BLOCK_X0 - 0.2, BLOCK_X1 + 0.2)
    pin = cyl(PIN_R, [PIN_YW, BLOCK_Y0 - 3, SLOT_ZC],
              [PIN_YW, SLOT_MOUTH_Y + 0.1, SLOT_ZC])
    m = diff(body, [slot, pin])
    assert_bore_open(m, [PIN_YW, BLOCK_Y0 + 0.2, SLOT_ZC],
                     [PIN_YW, y_head - 0.2, SLOT_ZC], "clamp peg bore")
    name = "clamp_lower_mount" + ("_thicker" if thicker else "")
    return save(m, name, orient=keep_pose), m


# ------------------------------------------------------- 5. saddle clamp screw
# v2: no ball tip / snap cap. The thread runs to the end; a Ø18 tip disc with
# an internal printed thread screws on from inside the channel and is CA-glued
# (captive — field issue F5). TPU pad glues into the disc face pocket.
def make_clamp_screw():
    knob = cyl(16, [0, 0, 0], [0, 0, 10])
    notches = [cyl(3.5, [16 * np.cos(a), 16 * np.sin(a), -0.1],
                   [16 * np.cos(a), 16 * np.sin(a), 10.1])
               for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    collar = cyl(7, [0, 0, 10], [0, 0, 14])
    thread = threaded_rod(THR_MAJ, THR_MIN, THR_PITCH, 34)
    thread.apply_translation([0, 0, 14])
    body = diff(union([knob, collar, thread]), notches)
    return save(body, "clamp_screw")          # knob down, thread vertical


def make_screw_tip_disc():
    disc = cyl(9, [0, 0, 0], [0, 0, 6])
    pocket = cyl(7.6, [0, 0, -0.1], [0, 0, 1.5])      # TPU pad pocket (face)
    nut = threaded_rod(THR_MAJ + THR_CLEAR, THR_MIN + THR_CLEAR, THR_PITCH, 4.6)
    nut.apply_translation([0, 0, 1.5])                 # threads on 4.5 mm
    m = diff(disc, [pocket, nut])
    return save(m, "screw_tip_disc")           # face down, thread bore up


def make_pins():
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    r0, r1, hgt = 2.6, 2.3, 20.0
    v = np.vstack([np.column_stack([r0 * np.cos(th), r0 * np.sin(th), np.zeros(48)]),
                   np.column_stack([r1 * np.cos(th), r1 * np.sin(th), np.full(48, hgt)]),
                   [[0, 0, 0], [0, 0, hgt]]])
    f = []
    for j in range(48):
        a, b = j, (j + 1) % 48
        f += [[a, b, 48 + b], [a, 48 + b, 48 + a], [96, b, a], [97, 48 + a, 48 + b]]
    frustum = trimesh.Trimesh(v, f, process=True)
    head = cyl(4.5, [0, 0, -3], [0, 0, 0])
    return save(union([frustum, head]), "pin_x3")   # head down; print 3 + spares


# ------------------------------------------------------- 6. TPU 68A parts
def make_tpu():
    save(box(0, 110.5, 0, 12.2, 0, SEAT_T), "tpu_seat_strip")
    save(box(0, 110, 0, 60, 0, PAD_T), "tpu_jaw_pad_x2")
    save(cyl(7.4, [0, 0, 0], [0, 0, 2.5]), "tpu_screw_pad")


# ------------------------------------------------------- assembly verification
def pose_clamp_on_wedge(clamp, name, pitch_deg):
    """Clamp print frame -> wedge local frame at the assembled position."""
    th = np.radians(pitch_deg)
    s34 = S_LO[name] + BLOCK_Z1
    M = np.eye(4)
    M[:3, 0] = [0, 1, 0]                                   # x_print -> y_wedge
    M[:3, 1] = [np.sin(th), 0, -np.cos(th)]                # y_print -> -normal
    M[:3, 2] = [-np.cos(th), 0, -np.sin(th)]               # z_print -> -slope
    M[:3, 3] = [s34 * np.cos(th) - SLOT_MOUTH_Y * np.sin(th), 0,
                10 + s34 * np.sin(th) + SLOT_MOUTH_Y * np.cos(th)]
    c = clamp.copy()
    c.apply_transform(M)
    return c


def verify_t_joint(wedge_mesh, clamp_mesh, name, pitch_deg):
    posed = pose_clamp_on_wedge(clamp_mesh, name, pitch_deg)
    inter = trimesh.boolean.intersection([wedge_mesh, posed], engine="manifold")
    vol = 0.0 if inter.is_empty else inter.volume
    assert vol < 5.0, f"T-joint interference on {name}: {vol:.1f} mm3"
    print(f"T-joint {name}: interference {vol:.2f} mm3 (clearance OK)")
    return posed


if __name__ == "__main__":
    make_saddle()
    make_mast()
    wedges = {}
    for name, ang in WEDGE_ANGLES.items():
        _, wedges[name] = make_wedge(name, ang)
    _, clamp_m = make_clamp_lower_mount(thicker=False)
    make_clamp_lower_mount(thicker=True)
    make_clamp_screw()
    make_screw_tip_disc()
    make_pins()
    make_tpu()
    for name, ang in WEDGE_ANGLES.items():
        verify_t_joint(wedges[name], clamp_m, name, ang)
    print("\nAll STLs written to", OUT, "- posed for printing, no rotation needed")
