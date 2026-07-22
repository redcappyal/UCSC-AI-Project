"""Parametric STL generator for the squash fin mount — fully-printed revision.

Source of truth: docs/mount-spec.md. All dimensions mm. Iteration-1 design
goals: NO purchased hardware (printed threads, dovetails, tapered pins), TPU
68A for all soft parts, relaxed tolerances (usable, not precise).

Every STL is exported ALREADY POSED FOR PRINTING — load into the slicer and
print without rotating. Overhangs are designed out (45-degree chamfers,
teardrop boss, side-lying exports); only the cradle needs supports (under its
back-plate wings around the dovetail boss).

Edit the PARAMS block (site-measured values marked TODO) and re-run:

    python3 hardware/generate_mount_stls.py

Outputs to hardware/stl/. Requires: trimesh, manifold3d, shapely, numpy.
Draft solids: no fillets (spec 7.1 wants >=2 near glass — acceptable for the
first test print).
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
G = 5.0           # TODO site: fin-to-wall gap at junction (spec 5: 2-8)
CHANNEL = T_F + 3.5          # 15.5 between PETG jaw faces
JAW_WALL = 8.0
JAW_DEPTH = 80.0
SADDLE_LEN = 130.0
CAP_T = 15.0
CAP_HALF_W = 20.0
SEAT_PROUD = 0.5
# TPU 68A pads: deeper pockets (soft pads need wall support), thicker sections
SEAT_T, SEAT_RECESS = 3.0, 2.5            # 0.5 proud
PAD_T, PAD_RECESS = 4.0, 2.5              # jaw pads, 1.5 proud
WALL_T, WALL_RECESS = 3.0, 2.0            # wall stops, 1 proud
SCREW_X, SCREW_Z = 65.0, -45.0
# printed clamp thread (coarse prints anywhere)
THR_MAJ, THR_MIN, THR_PITCH = 6.0, 4.75, 2.5   # radii: M12x2.5-ish
THR_CLEAR = 0.4                                 # radial nut clearance
DT_CLEAR = 0.4                                  # dovetail clearance (relaxed)
WEDGE_ANGLES = {"A_uw169_40deg": 40.0, "B_uw43_32deg": 32.0}
# phone + case envelope — TODO site: measure actual phone/case (spec 5: P, E, M)
PHONE_L, PHONE_H, PHONE_T = 152.0, 77.0, 10.0
LINER = 1.5

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


def save(mesh, name, orient=drop_to_plate):
    mesh = orient(mesh)
    assert mesh.is_watertight, f"{name} not watertight"
    assert mesh.volume > 0, f"{name} zero volume"
    assert abs(mesh.bounds[0][2]) < 1e-6, f"{name} not on the build plate"
    mesh.export(OUT / f"{name}.stl")
    ext = np.round(mesh.extents, 1)
    print(f"{name:22s} extents {ext} mm  volume {mesh.volume/1000:.1f} cm3")
    return mesh


# shared dovetail profile: root 20 wide, crest 26, height 6
def male_rail_xz(cx, z0):
    return [(cx - 10, z0), (cx + 10, z0), (cx + 13, z0 + 6), (cx - 13, z0 + 6)]


# ------------------------------------------------------- 1. saddle
# Design frame: X along fin (0 = wall end), Y across fin, Z from fin top edge.
# PRINTS UPSIDE-DOWN (cap top on the plate, jaws opening up): the datum-A seat
# pocket and both jaw-pad pockets print crisp; wall bosses get 45-degree
# chamfers and the screw boss a teardrop roof so the flipped part is
# support-free. Cap top carries a female dovetail slot + pin hole for the mast.
def make_saddle():
    cap = box(0, SADDLE_LEN, -CAP_HALF_W, CAP_HALF_W, SEAT_PROUD, CAP_TOP)
    jaw_n = box(0, SADDLE_LEN, -HALF_CH - JAW_WALL, -HALF_CH, -JAW_DEPTH, SEAT_PROUD)
    jaw_p = box(0, SADDLE_LEN, HALF_CH, BOSS_Y, -JAW_DEPTH, SEAT_PROUD)
    boss_hi = box(-(G - 1), 0, -10, 10, -30, -10)
    boss_lo = box(-(G - 1), 0, -10, 10, -80, -60)
    # 45-degree chamfers above each wall boss (self-supporting when flipped)
    cham_hi = prism_xz([(-(G - 1), -10), (0, -10), (0, -10 + (G - 1))], -10, 10)
    cham_lo = prism_xz([(-(G - 1), -60), (0, -60), (0, -60 + (G - 1))], -10, 10)
    screw_boss = cyl(13, [SCREW_X, BOSS_Y - 0.1, SCREW_Z],
                     [SCREW_X, BOSS_Y + 14, SCREW_Z])
    # true 45-degree teardrop: sides tangent to the boss circle, so the
    # flipped print has no bare cylinder-top slivers
    td = 13 * 1.4142
    teardrop = prism_xz([(SCREW_X - td, SCREW_Z), (SCREW_X + td, SCREW_Z),
                         (SCREW_X, SCREW_Z + td)], BOSS_Y - 0.1, BOSS_Y + 14)
    body = union([cap, jaw_n, jaw_p, boss_hi, boss_lo, cham_hi, cham_lo,
                  screw_boss, teardrop])

    nut = threaded_rod(THR_MAJ + THR_CLEAR, THR_MIN + THR_CLEAR, THR_PITCH, 24)
    nut.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
    nut.apply_translation([SCREW_X, BOSS_Y + 14.1 - 24, SCREW_Z])
    # mast slot in cap top: mouth 20.8 at the surface flaring to 26.8 at -6.3,
    # along Y, stop at +Y end
    slot = prism_xz([(17.1, CAP_TOP + 0.1), (37.9, CAP_TOP + 0.1),
                     (40.9, CAP_TOP - 6.3), (14.1, CAP_TOP - 6.3)],
                    -CAP_HALF_W - 0.1, 15.0)
    cuts = [
        box(9.75, 120.25, -6.2, 6.2, SEAT_PROUD, SEAT_PROUD + SEAT_RECESS),
        box(10, 120, -HALF_CH - PAD_RECESS, -HALF_CH + 0.1, -70, -10),
        box(10, 120, HALF_CH - 0.1, HALF_CH + PAD_RECESS, -70, -10),
        box(-(G - 1) - 0.1, -(G - 1) + WALL_RECESS, -8, 8, -28, -12),
        box(-(G - 1) - 0.1, -(G - 1) + WALL_RECESS, -8, 8, -78, -62),
        nut,
        cyl(10, [SCREW_X, HALF_CH - 0.1, SCREW_Z], [SCREW_X, HALF_CH + 6, SCREW_Z]),
        slot,
        cyl(2.6, [55, 0, 7.0], [55, 0, CAP_TOP + 0.1]),   # mast pin hole
    ]
    return save(diff(body, cuts), "saddle", orient=flip_upside_down)


# ------------------------------------------------------- 2. mast
# Separate part again (the merged version had no printable orientation).
# Base carries a male rail underneath (slides into the cap slot, pinned);
# column rises to the wedge rail. Base width = column width so the whole -Y
# side is coplanar: PRINTS LYING ON ITS SIDE, fully support-free (every
# feature is an extrusion along Y).
def make_mast():
    base = box(0, 60, -15, 15, 0, 8)
    under_rail = prism_xz([(17.5, 0.1), (37.5, 0.1), (40.5, -6.0), (14.5, -6.0)],
                          -15, 15)
    col = box(5, 50, -15, 15, 8, 48)
    top_rail = prism_xz(male_rail_xz(27.5, 48), -15, 15)
    stop = prism_xz([(14.5, 48), (40.5, 48), (40.5, 54), (14.5, 54)], 15, 17)
    body = union([base, under_rail, col, top_rail, stop])
    cuts = [
        cyl(2.6, [44.5, 0, 38], [44.5, 0, 48.1]),    # wedge pin hole
        cyl(2.6, [55, 0, -6.1], [55, 0, 8.1]),       # saddle pin hole (through)
    ]
    return save(diff(body, cuts), "mast", orient=lie_on_minus_y)


# ------------------------------------------------------- 3. wedges
# Female slot below rides the mast rail; the inclined face carries its own
# male rail + end stop for the cradle. PRINTS LYING ON ITS SIDE (all features
# are Y-extrusions), support-free.
def make_wedge(name, pitch_deg):
    slope = np.radians(90.0 - pitch_deg)
    x0, x1, h_low = 0.0, 42.0, 10.0
    h_high = h_low + (x1 - x0) * np.tan(slope)
    body = prism_xz([(x0, 0), (x1, 0), (x1, h_low), (x0, h_high)], -15, 15)
    c, h = DT_CLEAR, 6 + DT_CLEAR
    slot = prism_xz([(22.5 - 10 - c, -0.1), (22.5 + 10 + c, -0.1),
                     (22.5 + 13 + c, -0.1 + h), (22.5 - 13 - c, -0.1 + h)],
                    -15.2, 15.2)   # mast rail cx 27.5 -> local 22.5

    n = np.array([np.sin(np.radians(pitch_deg)), 0, np.cos(np.radians(pitch_deg))])
    fx = np.array([np.cos(np.radians(pitch_deg)), 0, -np.sin(np.radians(pitch_deg))])
    mid = np.array([(x0 + x1) / 2, 0, (h_low + h_high) / 2])
    rail = prism_xz(male_rail_xz(0, 0), -15, 15)
    stop = prism_xz([(-13, 0), (13, 0), (13, 6), (-13, 6)], 15, 17)
    R = rotation_matrix(np.radians(pitch_deg), [0, 1, 0])
    for part in (rail, stop):
        part.apply_transform(R)
        part.apply_translation(mid - 0.05 * n)
    body = union([body, rail, stop])
    cuts = [slot,
            cyl(2.6, mid - 10 * fx + 7 * n, mid - 10 * fx - 12 * n),  # cradle pin
            cyl(2.6, [37.5, 0, -0.1], [37.5, 0, 12])]                  # mast pin
    return save(diff(body, cuts), f"wedge_{name}", orient=lie_on_minus_y)


# ------------------------------------------------------- 4. cradle
# PRINTS BACK-BOSS-DOWN — the one part that needs supports (under the
# back-plate wings around the dovetail boss, 8 mm gap; scars are cosmetic).
def make_cradle():
    L = PHONE_L + 2 * (LINER + 0.4) + 12
    H = PHONE_H + LINER + 0.4 + 6
    rail_top = 5 + LINER + PHONE_T + 5
    plate = box(0, L, 0, H, 0, 5)
    rail_b = box(0, L, 0, 6, 5, rail_top)
    rail_l = box(0, 6, 0, H, 5, rail_top)
    rail_r = box(L - 6, L, 0, H, 5, rail_top)
    prow = extrude_polygon(Polygon([(0, rail_top), (6, rail_top), (0, rail_top + 5)]),
                           L - 40)
    prow.apply_transform(np.array([[0, 0, 1, 20], [1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 0, 1]], float))
    tabs = [box(30, 42, 0, 11, 5 + LINER + PHONE_T, rail_top),
            box(L - 42, L - 30, 0, 11, 5 + LINER + PHONE_T, rail_top),
            box(0, 11, 34, 46, 5 + LINER + PHONE_T, rail_top),
            box(L - 11, L, 20, 32, 5 + LINER + PHONE_T, rail_top)]
    cx, cy = L / 2, H / 2 + 3
    boss = box(cx - 20, cx + 20, cy - 17, cy + 17, -8, 0)
    body = union([plate, rail_b, rail_l, rail_r, prow, boss] + tabs)
    c, h = DT_CLEAR, 6 + DT_CLEAR
    slot = extrude_polygon(Polygon([(cy - 10 - c, -8.1), (cy + 10 + c, -8.1),
                                    (cy + 13 + c, -8.1 + h), (cy - 13 - c, -8.1 + h)]),
                           40.1)
    slot.apply_transform(np.array([[0, 0, 1, cx - 26.2], [1, 0, 0, 0],
                                   [0, 1, 0, 0], [0, 0, 0, 1]], float))
    cuts = [
        box(24, L - 24, 14, H - 8, -0.1, 5.1),
        box(-0.1, 60, H - 26, H + 0.1, -0.1, rail_top + 6),
        box(cx - 8, cx + 8, -0.1, 6.1, 4.9, rail_top + 6),
        box(52, 61, -0.1, H + 0.1, -0.1, 2),
        box(L - 61, L - 52, -0.1, H + 0.1, -0.1, 2),
        slot,
        cyl(2.6, [cx - 10, cy, -8.1], [cx - 10, cy, 5.1]),
    ]
    return save(diff(body, cuts), "cradle")   # drop_to_plate: boss face at z0


# ------------------------------------------------------- 5. printed clamp
def make_clamp_screw():
    knob = cyl(16, [0, 0, 0], [0, 0, 10])
    notches = [cyl(3.5, [16 * np.cos(a), 16 * np.sin(a), -0.1],
                   [16 * np.cos(a), 16 * np.sin(a), 10.1])
               for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    collar = cyl(7, [0, 0, 10], [0, 0, 14])
    thread = threaded_rod(THR_MAJ, THR_MIN, THR_PITCH, 30)
    thread.apply_translation([0, 0, 14])
    tip = cyl(3.5, [0, 0, 44], [0, 0, 48])
    ball = trimesh.creation.icosphere(radius=4, subdivisions=3)
    ball.apply_translation([0, 0, 49])
    body = diff(union([knob, collar, thread, tip, ball]), notches)
    return save(body, "clamp_screw")          # knob down, thread vertical


def make_pad_cap():
    disc = cyl(9, [0, 0, 0], [0, 0, 4])
    face_pocket = cyl(7.6, [0, 0, -0.1], [0, 0, 1.5])
    socket = trimesh.creation.icosphere(radius=4.3, subdivisions=3)
    socket.apply_translation([0, 0, 4.6])
    entry = cyl(3.4, [0, 0, 3.9], [0, 0, 8.1])
    return save(diff(disc, [face_pocket, socket, entry]), "pad_cap")  # face down


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
    return save(union([frustum, head]), "pin_x2")   # head down


# ------------------------------------------------------- 6. TPU 68A parts
def make_tpu():
    save(box(0, 110.5, 0, 12.2, 0, SEAT_T), "tpu_seat_strip")
    save(box(0, 110, 0, 60, 0, PAD_T), "tpu_jaw_pad_x2")
    save(box(0, 16, 0, 16, 0, WALL_T), "tpu_wall_pad_x2")
    save(cyl(7.4, [0, 0, 0], [0, 0, 2.5]), "tpu_swivel_face")
    band = diff(cyl(28, [0, 0, 0], [0, 0, 2.5]), [cyl(24, [0, 0, -0.1], [0, 0, 2.6])])
    save(band, "tpu_band_x2")


if __name__ == "__main__":
    make_saddle()
    make_mast()
    for name, ang in WEDGE_ANGLES.items():
        make_wedge(name, ang)
    make_cradle()
    make_clamp_screw()
    make_pad_cap()
    make_pins()
    make_tpu()
    print("\nAll STLs written to", OUT, "- posed for printing, no rotation needed")
