"""Parametric STL generator for the squash fin mount — v3 (own clamp, old mast).

Source of truth: docs/mount-spec.md + docs/superpowers/specs/
2026-07-23-mount-v2-clamp-redesign-design.md (v3 addendum). All dimensions mm.

v3 keeps the v1 architecture the team already printed and validated — the
mast is emitted BYTE-COMPATIBLE with v1 (do not reprint it), the saddle keeps
its v1 slot position, and the wedges keep their v1 steep-face geometry
(face angle = 90 - pitch; the phone plane lies parallel to the face exactly
like the v1 cradle). The custom cradle is replaced by a phone clamp of our
own design that BORROWS THE MECHANISM of the Printables tripod clamp
(fixed jaw + rail-guided sliding jaw + screw drive) but is built from our
proven printed M12x2.5 thread and mounts through the same female-dovetail
back boss + tapered peg the v1 cradle used. No third-party files needed.

Field-test fixes carried over from v2:
- wall-stop bosses / gap fillers deleted from the saddle;
- all NEW rail/stop unions embedded >= 2 mm (the v1 0.05 mm embedment made
  the wedge-face dovetails degenerate — root cause of the unmountable
  cradle. The v1 wedge cradle-peg bore was also misaligned with the cradle:
  offset along the slope on the wedge but along the wall on the cradle);
- saddle screw ball tip replaced by a threaded-on tip disc (glued = captive);
  the phone-jaw screw uses the same disc to hold its carriage captive;
- every peg bore modeled open and asserted open by containment sampling
  (the saddle cap bore stays blind BY DESIGN — it must never reach glass);
- optics: with the phone stack ~9 mm thicker than the v1 cradle and the mast
  kept at 48, the legacy h/s >= 4.92 ratio (derived for a 90-mm-high lens)
  does not hold, but the FUNCTIONAL spec s10 acceptance — floor visible to
  <= 0.6 m of the back wall — is computed and asserted for both wedges,
  along with the WSF court-plane margin (>= 3 mm from the court-side face).

Every STL is exported ALREADY POSED FOR PRINTING — no rotation in the slicer.
phone_clamp_body is the ONE supported part (boss-down, supports under the
plate, 8 mm gap — same deal as the v1 cradle).

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

OUT = Path(__file__).parent / "stl"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- parameters
T_F = 12.0        # fin/wall glass thickness
WALL_T_GLASS = 12.0
CHANNEL = T_F + 3.5          # 15.5 between PETG jaw faces
JAW_WALL = 8.0
JAW_DEPTH = 80.0
SADDLE_LEN = 130.0
CAP_T = 15.0
CAP_HALF_W = 20.0
SEAT_PROUD = 0.5
SEAT_T, SEAT_RECESS = 3.0, 2.5
PAD_T, PAD_RECESS = 4.0, 2.5
SCREW_X, SCREW_Z = 65.0, -45.0
THR_MAJ, THR_MIN, THR_PITCH = 6.0, 4.75, 2.5   # radii: M12x2.5-ish
THR_CLEAR = 0.4
DT_CLEAR = 0.4
PIN_R = 2.6
T_EMBED = 2.0                # embedment for all new rail/stop unions
WEDGE_ANGLES = {"A_uw169_40deg": 40.0, "B_uw43_32deg": 32.0}   # optical pitch
MAST_H = 48.0                # v1 mast column top — KEPT (part already printed)

# phone clamp (own design, borrowed screw-drive mechanism)
PLATE_W = 70.0               # x, across the wall direction
PLATE_T = 6.0                # z, back 0 .. front 6
PLATE_L = 159.0              # y, 0 = top (hook end, up-slope) .. screw boss end
BOSS_H = 8.0                 # back dovetail boss height (v1 cradle value)
BOSS_YC = 64.0               # boss center along the plate
HOOK_Y = 10.0                # fixed top hook: phone top-edge contact plane
PW_MIN, PW_MAX = 68.0, 95.0  # phone width range across the jaws
PHONE_T_MIN, PHONE_T_MAX = 8.0, 11.0
LINER = 1.5                  # TPU proud of pockets on every phone contact
RAIL_Y0, RAIL_Y1 = 70.0, 145.0     # carriage rail span on the plate front
CARR_L = 40.0                # carriage body length along y
SCREW_AXIS_Z = 22.0          # phone-jaw screw axis height above plate back
LENS_FROM_TOP = 22.0         # ultrawide lens center below the phone top edge
COURT_MARGIN = 3.0           # min distance outboard of the court-side face
BLIND_MAX = 600.0            # spec s10: floor visible to <= 0.6 m of the wall
EDGE_AFF = 2130.0

HALF_CH = CHANNEL / 2.0
CAP_TOP = SEAT_PROUD + CAP_T
BOSS_Y = HALF_CH + JAW_WALL


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
    p = extrude_polygon(Polygon(points), x1 - x0)
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
    p0, p1 = np.array(p0, float), np.array(p1, float)
    ts = np.linspace(0.05, 0.95, samples)[:, None]
    pts = p0 + ts * (p1 - p0)
    inside = mesh.contains(pts)
    assert not inside.any(), (
        f"{name}: bore {p0}->{p1} blocked at t="
        f"{ts[inside.nonzero()[0][0], 0]:.2f}")


# shared dovetail profile: root 20 wide, crest 26, height 6 (v1 family)
def male_rail_pts(cx, z0, embed=T_EMBED):
    """Hexagon: straight root buried `embed` deep, then the v1 flare."""
    return [(cx - 10, z0 - embed), (cx + 10, z0 - embed), (cx + 10, z0),
            (cx + 13, z0 + 6), (cx - 13, z0 + 6), (cx - 10, z0)]


# ------------------------------------------------- placement (v1 frame, fixed)
# World: X along the fin, 0 at the wall OUTER face; Z vertical, 0 at the
# fin/wall top edge. Saddle X 0..130 (slid flush). Mast local == saddle X
# (v1 slot position), wedge local x = mast x - 5, wedge sits at CAP_TOP+48.
# Face angle from horizontal th_f = 90 - pitch, DESCENDING away from the wall
# (v1 geometry): face top corner f0 = (5, CAP_TOP+48+h_hi) in world (X, Z),
# down-slope dir d = (cos, -sin), outward normal n = (sin, cos).
# The clamp body lies parallel: back plane at n=BOSS_H; body y_b maps to the
# slope coord t = T0 + y_b (T0 = t of the plate's top end, may be negative =
# overhanging the face's top corner toward the wall).

def wedge_frame(pitch_deg):
    th_f = np.radians(90.0 - pitch_deg)
    h_hi = 10.0 + 42.0 * np.tan(th_f)
    L = 42.0 / np.cos(th_f)
    d = np.array([np.cos(th_f), -np.sin(th_f)])
    n = np.array([np.sin(th_f), np.cos(th_f)])
    return th_f, h_hi, L, d, n


def placement(pitch_deg):
    """Per-wedge rail position on the face: bind the court-plane margin."""
    th_f, h_hi, L, d, n = wedge_frame(pitch_deg)
    # most court-side hardware = plate top BACK corner (y_b=0, n=BOSS_H):
    # X = 5 + T0*d_x + BOSS_H*n_x >= -WALL_T_GLASS + COURT_MARGIN
    T0_bind = ((-WALL_T_GLASS + COURT_MARGIN) - 5 - BOSS_H * n[0]) / d[0]
    t_rail = np.clip(T0_bind + BOSS_YC, 15.0, L - 15.0)
    T0 = t_rail - BOSS_YC
    f0 = np.array([5.0, CAP_TOP + MAST_H + h_hi])

    def world(y_b, n_off):
        p = f0 + (T0 + y_b) * d + n_off * n
        return p[0], p[1]

    margin = world(0.0, BOSS_H)[0] + WALL_T_GLASS
    # worst case for the blind zone and h/s: thickest phone (lens farthest out)
    n_lens = BOSS_H + PLATE_T + LINER + PHONE_T_MAX
    Xl, Zl = world(HOOK_Y + LENS_FROM_TOP, n_lens)
    s_occ = Xl + WALL_T_GLASS
    ratio = Zl / s_occ
    blind = (EDGE_AFF + Zl) * s_occ / Zl - (s_occ)   # floor hidden, from court face
    return dict(t_rail=float(t_rail), T0=float(T0), margin=float(margin),
                ratio=float(ratio), blind=float(blind), Xl=float(Xl),
                Zl=float(Zl))


PLACE = {name: placement(ang) for name, ang in WEDGE_ANGLES.items()}
for name, p in PLACE.items():
    print(f"placement {name}: rail at t={p['t_rail']:.1f}  "
          f"court-margin={p['margin']:.1f}mm  lens h={p['Zl']:.0f} "
          f"s={p['Xl']+WALL_T_GLASS:.0f}  h/s={p['ratio']:.2f}  "
          f"floor-blind={p['blind']:.0f}mm (max {BLIND_MAX:.0f})")
    assert p['margin'] >= COURT_MARGIN - 0.1, f"{name}: court-plane margin"
    assert p['blind'] <= BLIND_MAX, f"{name}: near-wall floor blind zone"


# ------------------------------------------------------- 1. saddle
# v1 slot position (mast at the saddle's wall end — the printed mast drops
# straight back in). v2 simplifications kept: no wall bosses/stops/pads;
# screw tip is the glued-on threaded disc.
def make_saddle():
    cap = box(0, SADDLE_LEN, -CAP_HALF_W, CAP_HALF_W, SEAT_PROUD, CAP_TOP)
    jaw_n = box(0, SADDLE_LEN, -HALF_CH - JAW_WALL, -HALF_CH, -JAW_DEPTH, SEAT_PROUD)
    jaw_p = box(0, SADDLE_LEN, HALF_CH, BOSS_Y, -JAW_DEPTH, SEAT_PROUD)
    screw_boss = cyl(13, [SCREW_X, BOSS_Y - 0.1, SCREW_Z],
                     [SCREW_X, BOSS_Y + 14, SCREW_Z])
    td = 13 * 1.4142
    teardrop = prism_xz([(SCREW_X - td, SCREW_Z), (SCREW_X + td, SCREW_Z),
                         (SCREW_X, SCREW_Z + td)], BOSS_Y - 0.1, BOSS_Y + 14)
    body = union([cap, jaw_n, jaw_p, screw_boss, teardrop])

    nut = threaded_rod(THR_MAJ + THR_CLEAR, THR_MIN + THR_CLEAR, THR_PITCH, 24)
    nut.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
    nut.apply_translation([SCREW_X, BOSS_Y + 14.1 - 24, SCREW_Z])
    slot = prism_xz([(17.1, CAP_TOP + 0.1), (37.9, CAP_TOP + 0.1),
                     (40.9, CAP_TOP - 6.3), (14.1, CAP_TOP - 6.3)],
                    -CAP_HALF_W - 0.1, 15.0)
    cuts = [
        box(9.75, 120.25, -6.2, 6.2, SEAT_PROUD, SEAT_PROUD + SEAT_RECESS),
        box(10, 120, -HALF_CH - PAD_RECESS, -HALF_CH + 0.1, -70, -10),
        box(10, 120, HALF_CH - 0.1, HALF_CH + PAD_RECESS, -70, -10),
        nut,
        cyl(10, [SCREW_X, HALF_CH - 0.1, SCREW_Z], [SCREW_X, HALF_CH + 6, SCREW_Z]),
        slot,
        cyl(PIN_R, [55, 0, 6.0], [55, 0, CAP_TOP + 0.1]),  # blind BY DESIGN
    ]
    m = diff(body, cuts)
    assert_bore_open(m, [55, 0, 6.2], [55, 0, CAP_TOP], "saddle mast-peg bore")
    assert_bore_open(m, [SCREW_X, HALF_CH + 1, SCREW_Z],
                     [SCREW_X, BOSS_Y + 13.5, SCREW_Z], "saddle thread bore")
    return save(m, "saddle", orient=flip_upside_down)


# ------------------------------------------------------- 2. mast — v1 VERBATIM
# The team's printed mast fits perfectly; this reproduces the v1 geometry
# byte-for-byte so the file matches the part in hand. Do not "improve".
def make_mast():
    base = box(0, 60, -15, 15, 0, 8)
    under_rail = prism_xz([(17.5, 0.1), (37.5, 0.1), (40.5, -6.0), (14.5, -6.0)],
                          -15, 15)
    col = box(5, 50, -15, 15, 8, 48)
    top_rail = prism_xz([(17.5, 48), (37.5, 48), (40.5, 54), (14.5, 54)], -15, 15)
    stop = prism_xz([(14.5, 48), (40.5, 48), (40.5, 54), (14.5, 54)], 15, 17)
    body = union([base, under_rail, col, top_rail, stop])
    cuts = [
        cyl(PIN_R, [44.5, 0, 38], [44.5, 0, 48.1]),    # wedge pin hole
        cyl(PIN_R, [55, 0, -6.1], [55, 0, 8.1]),       # saddle pin hole (through)
    ]
    m = diff(body, cuts)
    assert_bore_open(m, [55, 0, -5.9], [55, 0, 7.9], "mast saddle-peg bore")
    assert_bore_open(m, [44.5, 0, 38.2], [44.5, 0, 47.9], "mast wedge-peg pocket")
    return save(m, "mast", orient=lie_on_minus_y)


# ------------------------------------------------------- 3. wedges (v1 + fixes)
# v1 steep-face geometry (face = 90 - pitch, descending away from the wall);
# fixes: rail/stop embedded 2 mm (v1's 0.05 mm overlap corrupted the
# booleans), rail position per-wedge from placement(), and the clamp-peg
# bore moved to (t_rail, wall -10) — v1 had it offset along the SLOPE while
# the cradle's bore was offset along the WALL, so they never lined up.
def make_wedge(name, pitch_deg):
    th_f, h_hi, L, d2, n2 = wedge_frame(pitch_deg)
    body = prism_xz([(0, 0), (42, 0), (42, 10), (0, h_hi)], -15, 15)
    c, hh = DT_CLEAR, 6 + DT_CLEAR
    slot = prism_xz([(22.5 - 10 - c, -0.1), (22.5 + 10 + c, -0.1),
                     (22.5 + 13 + c, -0.1 + hh), (22.5 - 13 - c, -0.1 + hh)],
                    -15.2, 15.2)   # mast rail cx 27.5 -> local 22.5

    f0 = np.array([0.0, h_hi])                      # face top corner (x, z)
    d3 = np.array([d2[0], 0, d2[1]])
    n3 = np.array([n2[0], 0, n2[1]])
    tr = PLACE[name]['t_rail']

    def face_pt(t, n_off):
        p = f0 + t * d2 + n_off * n2
        return (p[0], p[1])

    rail = prism_xz([face_pt(tr - 10, -T_EMBED), face_pt(tr + 10, -T_EMBED),
                     face_pt(tr + 10, 0), face_pt(tr + 13, 6),
                     face_pt(tr - 13, 6), face_pt(tr - 10, 0)], -15, 15)
    # stop tab caps the rail's +y end (fully on the wedge body): the clamp
    # boss (40 wide) rides to it, so the body sits at lateral -9 and its
    # x=-1 peg bore lands on this wedge's y=-10 bore (v1 had these crossed)
    stop = prism_xz([face_pt(tr - 15, -T_EMBED), face_pt(tr + 15, -T_EMBED),
                     face_pt(tr + 15, 6), face_pt(tr - 15, 6)], 11, 15)
    body = union([body, rail, stop])

    p_pin = f0[0] * np.array([1, 0, 0]) + np.array([0, 0, f0[1]]) + tr * d3
    p_pin = np.array([p_pin[0], -10.0, p_pin[2]])   # wall offset -10
    cuts = [slot,
            cyl(PIN_R, p_pin + 8 * n3, p_pin - 12 * n3),   # clamp peg bore
            cyl(PIN_R, [37.5, 0, -0.1], [37.5, 0, 12])]     # mast peg (v1)
    m = diff(body, cuts)
    assert_bore_open(m, p_pin + 6.2 * n3, p_pin - 11.8 * n3,
                     f"wedge {name} clamp-peg bore", samples=25)
    assert_bore_open(m, [37.5, 0, 0.2], [37.5, 0, 11.8],
                     f"wedge {name} mast-peg bore")
    return save(m, f"wedge_{name}", orient=lie_on_minus_y), m


# ------------------------------------------------------- 4. phone clamp body
# Own design, borrowed mechanism. Plate with: fixed top hook (up-slope,
# low-profile), male dovetail rail on the front for the jaw carriage,
# M12x2.5 nut boss at the bottom end for the drive screw, and on the BACK
# the v1-cradle female-dovetail boss (slot along the wall, peg bore at
# wall -10) that rides the wedge face rail. Prints boss-down WITH SUPPORTS
# under the plate (8 mm gap) — the one supported part, like the v1 cradle.
def make_phone_clamp_body():
    plate = box(-PLATE_W / 2, PLATE_W / 2, 0, PLATE_L, 0, PLATE_T)
    hook_wall = box(-PLATE_W / 2, PLATE_W / 2, 0, HOOK_Y, PLATE_T,
                    PLATE_T + LINER + PHONE_T_MAX + 3.0)   # z to 21.5
    lip_z0 = PLATE_T + LINER + PHONE_T_MAX
    # lip with 45-deg underside: wedges thin phones toward the plate
    lip = prism_yz([(HOOK_Y, lip_z0 + 3.0), (HOOK_Y + 5.0, lip_z0 + 3.0),
                    (HOOK_Y + 2.0, lip_z0), (HOOK_Y, lip_z0)],
                   -PLATE_W / 2, PLATE_W / 2)
    rail = prism_xz(male_rail_pts(0.0, PLATE_T), RAIL_Y0, RAIL_Y1)
    # screw boss at the bottom end
    sboss = box(-15, 15, PLATE_L - 14, PLATE_L, 0, SCREW_AXIS_Z + 9)
    # back dovetail boss (v1 cradle values: 40 x 34 footprint, 8 tall)
    bboss = box(-20, 20, BOSS_YC - 17, BOSS_YC + 17, -BOSS_H, 0.1)
    body = union([plate, hook_wall, lip, rail, sboss, bboss])

    # female slot in the back boss: along X, mouth 20.8 flaring 26.8 (v1)
    ch = 6 + DT_CLEAR
    slot = prism_yz([(BOSS_YC - 10 - DT_CLEAR, -BOSS_H - 0.1),
                     (BOSS_YC + 10 + DT_CLEAR, -BOSS_H - 0.1),
                     (BOSS_YC + 13 + DT_CLEAR, -BOSS_H - 0.1 + ch),
                     (BOSS_YC - 13 - DT_CLEAR, -BOSS_H - 0.1 + ch)],
                    -20.2, 20.2)
    nut = threaded_rod(THR_MAJ + THR_CLEAR, THR_MIN + THR_CLEAR, THR_PITCH, 14.2)
    nut.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, -1, 0]))
    nut.apply_translation([0, PLATE_L + 0.1, SCREW_AXIS_Z])
    cuts = [
        slot,
        nut,
        # peg bore at x=-1 (lands on the wedge bore at wall -10 given the
        # body's -9 lateral seat), through plate + boss ceiling + slot cavity
        cyl(PIN_R, [-1, BOSS_YC, PLATE_T + 0.1], [-1, BOSS_YC, -BOSS_H - 0.1]),
        # TPU pockets: hook face (1.5 deep) + two back strips on the plate front
        box(-30, 30, HOOK_Y - 1.5, HOOK_Y + 0.1, PLATE_T + 1, PLATE_T + 9),
        box(-30, 30, 20, 30, PLATE_T - 1.5, PLATE_T + 0.1),
        box(-30, 30, 50, 60, PLATE_T - 1.5, PLATE_T + 0.1),
    ]
    m = diff(body, cuts)
    assert_bore_open(m, [-1, BOSS_YC, PLATE_T - 0.1],
                     [-1, BOSS_YC, -BOSS_H + 0.4], "clamp body peg bore")
    assert_bore_open(m, [0, PLATE_L - 0.5, SCREW_AXIS_Z],
                     [0, PLATE_L - 13.5, SCREW_AXIS_Z], "clamp body nut bore")
    # print pose: the boss is already the lowest feature — drop_to_plate IS
    # boss-down; supports go under the plate (8 mm gap), like the v1 cradle
    return save(m, "phone_clamp_body"), m


# ------------------------------------------------------- 5. jaw carriage
# Rides the body rail (female slot on its back), bottom-jaw shelf + 45-deg
# lip on its top end, and a coin-slot pocket at its bottom end that captures
# the screw tip disc (screw threads through the boss nut into the disc;
# CA-glue = carriage captive both ways). Prints slot-mouth-down (bridges).
def make_jaw_carriage():
    zb = PLATE_T + DT_CLEAR              # carriage back plane over plate front
    body = box(-22, 22, 0, CARR_L, zb, zb + 8)
    # jaw shelf + hook at the TOP end (contacts the phone bottom edge)
    shelf = box(-30, 30, 0, 8, zb, PLATE_T + LINER + PHONE_T_MAX + 3.0)
    lip = prism_yz([(0.0, PLATE_T + LINER + PHONE_T_MAX + 3.0),
                    (0.0, PLATE_T + LINER + PHONE_T_MAX),
                    (-2.0, PLATE_T + LINER + PHONE_T_MAX),
                    (-5.0, PLATE_T + LINER + PHONE_T_MAX + 3.0)],
                   -30, 30)
    # tower at the bottom end around the disc pocket
    tower = box(-14, 14, CARR_L - 12, CARR_L, zb, SCREW_AXIS_Z + 11)
    body = union([body, shelf, lip, tower])
    # female dovetail slot on the back, through (rides the body rail)
    ch = 6 + DT_CLEAR
    slot = prism_xz([(-10 - DT_CLEAR, zb - 0.1 - 20), (10 + DT_CLEAR, zb - 0.1 - 20),
                     (10 + DT_CLEAR, zb - 0.1), (13 + DT_CLEAR, zb - 0.1 + ch),
                     (-13 - DT_CLEAR, zb - 0.1 + ch), (-10 - DT_CLEAR, zb - 0.1)],
                    -0.2, CARR_L + 0.2)
    # slot is the male-shaped void: cut = rail shape + entry below
    coin = box(-9.6, 9.6, CARR_L - 10.4, CARR_L - 3.8,
               SCREW_AXIS_Z - 9.6, SCREW_AXIS_Z + 11.1)   # disc drops in from top
    tip_clear = cyl(7, [0, CARR_L - 10.4 - 9, SCREW_AXIS_Z],
                    [0, CARR_L - 3, SCREW_AXIS_Z])
    m = diff(body, [slot, coin, tip_clear])
    # pocket for the bottom-jaw TPU strip (same strip as the hook pocket)
    m = diff(m, [box(-30, 30, -0.1, 1.5, zb + 1, zb + 9)])
    def slot_down(mesh):
        mm = mesh.copy()
        mm.apply_translation([0, 0, -mm.bounds[0][2]])
        return mm
    return save(m, "phone_jaw_carriage", orient=slot_down), m


# ------------------------------------------------------- 6. screws & discs
def make_clamp_screw():
    """Saddle thumbscrew (v2: thread to the tip, disc threads on + glued)."""
    knob = cyl(16, [0, 0, 0], [0, 0, 10])
    notches = [cyl(3.5, [16 * np.cos(a), 16 * np.sin(a), -0.1],
                   [16 * np.cos(a), 16 * np.sin(a), 10.1])
               for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    collar = cyl(7, [0, 0, 10], [0, 0, 14])
    thread = threaded_rod(THR_MAJ, THR_MIN, THR_PITCH, 34)
    thread.apply_translation([0, 0, 14])
    body = diff(union([knob, collar, thread]), notches)
    return save(body, "clamp_screw")


def make_phone_jaw_screw():
    """Phone-jaw drive screw: knob + 52 mm thread; tip disc captures carriage."""
    knob = cyl(14, [0, 0, 0], [0, 0, 10])
    notches = [cyl(3.0, [14 * np.cos(a), 14 * np.sin(a), -0.1],
                   [14 * np.cos(a), 14 * np.sin(a), 10.1])
               for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    collar = cyl(7, [0, 0, 10], [0, 0, 13])
    thread = threaded_rod(THR_MAJ, THR_MIN, THR_PITCH, 52)
    thread.apply_translation([0, 0, 13])
    body = diff(union([knob, collar, thread]), notches)
    return save(body, "phone_jaw_screw")


def make_screw_tip_disc():
    """Ø18 disc, internal thread; used on BOTH screws (print 2). Face pocket
    takes the TPU pad on the saddle screw; harmless empty on the jaw screw."""
    disc = cyl(9, [0, 0, 0], [0, 0, 6])
    pocket = cyl(7.6, [0, 0, -0.1], [0, 0, 1.5])
    nut = threaded_rod(THR_MAJ + THR_CLEAR, THR_MIN + THR_CLEAR, THR_PITCH, 4.6)
    nut.apply_translation([0, 0, 1.5])
    return save(diff(disc, [pocket, nut]), "screw_tip_disc")


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
    return save(union([frustum, head]), "pin_x3")


# ------------------------------------------------------- 7. TPU 68A parts
def make_tpu():
    save(box(0, 110.5, 0, 12.2, 0, SEAT_T), "tpu_seat_strip")
    save(box(0, 110, 0, 60, 0, PAD_T), "tpu_jaw_pad_x2")
    save(cyl(7.4, [0, 0, 0], [0, 0, 2.5]), "tpu_screw_pad")
    save(box(0, 59.5, 0, 7.8, 0, 3.0), "tpu_phone_jaw_x2")    # hook + carriage
    save(box(0, 59.5, 0, 9.8, 0, 3.0), "tpu_phone_back_x2")   # plate front


# ------------------------------------------------------- assembly verification
def pose_body_on_wedge(body_mesh, name, pitch_deg):
    """Clamp-body frame -> wedge local frame at the assembled position."""
    th_f, h_hi, L, d2, n2 = wedge_frame(pitch_deg)
    T0 = PLACE[name]['T0']
    M = np.eye(4)
    M[:3, 0] = [0, 1, 0]                          # body x -> wedge y (wall)
    M[:3, 1] = [d2[0], 0, d2[1]]                  # body y -> down-slope
    M[:3, 2] = [n2[0], 0, n2[1]]                  # body z -> +normal
    org = (np.array([0.0, h_hi]) + T0 * d2 + BOSS_H * n2)   # body (0,0,0)
    M[:3, 3] = [org[0], -9.0, org[1]]             # boss rides to the stop at 11
    b = body_mesh.copy()
    b.apply_transform(M)
    return b


def verify_mount(wedge_mesh, body_mesh, name, pitch_deg):
    posed = pose_body_on_wedge(body_mesh, name, pitch_deg)
    inter = trimesh.boolean.intersection([wedge_mesh, posed], engine="manifold")
    vol = 0.0 if inter.is_empty else inter.volume
    assert vol < 5.0, f"body-wedge interference on {name}: {vol:.1f} mm3"
    print(f"mount {name}: body/wedge interference {vol:.2f} mm3 (OK)")


def verify_carriage(body_mesh, carr_mesh):
    for jaw_y in (HOOK_Y + PW_MIN, HOOK_Y + PW_MAX):
        c = carr_mesh.copy()
        c.apply_translation([0, jaw_y, 0])
        inter = trimesh.boolean.intersection([body_mesh, c], engine="manifold")
        vol = 0.0 if inter.is_empty else inter.volume
        assert vol < 5.0, f"carriage-body interference at jaw {jaw_y}: {vol:.1f}"
        print(f"carriage at jaw y={jaw_y:.0f}: interference {vol:.2f} mm3 (OK)")


if __name__ == "__main__":
    make_saddle()
    make_mast()
    wedges = {}
    for name, ang in WEDGE_ANGLES.items():
        _, wedges[name] = make_wedge(name, ang)
    _, body_m = make_phone_clamp_body()
    _, carr_m = make_jaw_carriage()
    make_clamp_screw()
    make_phone_jaw_screw()
    make_screw_tip_disc()
    make_pins()
    make_tpu()
    for name, ang in WEDGE_ANGLES.items():
        verify_mount(wedges[name], body_m, name, ang)
    verify_carriage(body_m, carr_m)
    print("\nAll STLs written to", OUT, "- posed for printing; only "
          "phone_clamp_body needs supports (under the plate, boss-down)")
