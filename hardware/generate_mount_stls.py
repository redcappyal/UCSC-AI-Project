"""Parametric STL generator for the squash fin mount — fully-printed revision.

Source of truth: docs/mount-spec.md. All dimensions mm. Iteration-1 design
goals: NO purchased hardware (printed threads, dovetails, tapered pins), TPU
68A for all soft parts, relaxed tolerances (usable, not precise).

Edit the PARAMS block (site-measured values marked TODO) and re-run:

    python3 hardware/generate_mount_stls.py

Outputs to hardware/stl/. Requires: trimesh, manifold3d, shapely, numpy.
Draft solids: no fillets (spec 7.1 wants >=2 near glass — acceptable for the
first test print).
"""
import numpy as np
import trimesh
from trimesh.creation import box as _box, cylinder as _cyl, extrude_polygon
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


def save(mesh, name):
    assert mesh.is_watertight, f"{name} not watertight"
    assert mesh.volume > 0, f"{name} zero volume"
    mesh.export(OUT / f"{name}.stl")
    ext = np.round(mesh.extents, 1)
    print(f"{name:22s} extents {ext} mm  volume {mesh.volume/1000:.1f} cm3")
    return mesh


# shared dovetail profile: root 20 wide, crest 26, height 6
def male_rail_xz(cx, z0):
    return [(cx - 10, z0), (cx + 10, z0), (cx + 13, z0 + 6), (cx - 13, z0 + 6)]


def female_slot_xz(cx, z0):
    c, h = DT_CLEAR, 6 + DT_CLEAR
    return [(cx - 10 - c, z0 - 0.1), (cx + 10 + c, z0 - 0.1),
            (cx + 13 + c, z0 + h), (cx - 13 - c, z0 + h)]


# ------------------------------------------------------- 1. saddle + mast
# One part. Frame: X along fin (0 = wall end), Y across fin, Z from fin top
# edge. Cap z 0.5..15.5; mast column rises to a dovetail rail for the wedge.
def make_saddle_mast():
    cap = box(0, SADDLE_LEN, -CAP_HALF_W, CAP_HALF_W, SEAT_PROUD, CAP_TOP)
    jaw_n = box(0, SADDLE_LEN, -HALF_CH - JAW_WALL, -HALF_CH, -JAW_DEPTH, SEAT_PROUD)
    jaw_p = box(0, SADDLE_LEN, HALF_CH, HALF_CH + JAW_WALL, -JAW_DEPTH, SEAT_PROUD)
    boss_hi = box(-(G - 1), 0, -10, 10, -30, -10)
    boss_lo = box(-(G - 1), 0, -10, 10, -80, -60)
    col = box(5, 50, -15, 15, CAP_TOP, CAP_TOP + 40)
    rail = prism_xz(male_rail_xz(27.5, CAP_TOP + 40), -15, 15)
    stop = prism_xz([(12, CAP_TOP + 38), (43, CAP_TOP + 38),
                     (43, CAP_TOP + 50), (12, CAP_TOP + 50)], 15, 17)
    screw_boss = cyl(13, [SCREW_X, HALF_CH + JAW_WALL - 0.1, SCREW_Z],
                     [SCREW_X, HALF_CH + JAW_WALL + 14, SCREW_Z])
    body = union([cap, jaw_n, jaw_p, boss_hi, boss_lo, col, rail, stop, screw_boss])

    nut = threaded_rod(THR_MAJ + THR_CLEAR, THR_MIN + THR_CLEAR, THR_PITCH, 24)
    nut.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
    nut.apply_translation([SCREW_X, HALF_CH + JAW_WALL + 14.1 - 24, SCREW_Z])
    cuts = [
        box(9.75, 120.25, -6.2, 6.2, SEAT_PROUD, SEAT_PROUD + SEAT_RECESS),
        box(10, 120, -HALF_CH - PAD_RECESS, -HALF_CH + 0.1, -70, -10),
        box(10, 120, HALF_CH - 0.1, HALF_CH + PAD_RECESS, -70, -10),
        box(-(G - 1) - 0.1, -(G - 1) + WALL_RECESS, -8, 8, -28, -12),
        box(-(G - 1) - 0.1, -(G - 1) + WALL_RECESS, -8, 8, -78, -62),
        nut,                                                    # printed nut thread
        cyl(10, [SCREW_X, HALF_CH - 0.1, SCREW_Z], [SCREW_X, HALF_CH + 6, SCREW_Z]),
        cyl(2.6, [44.5, 0, CAP_TOP + 30], [44.5, 0, CAP_TOP + 40.1]),  # wedge pin
    ]
    return save(diff(body, cuts), "saddle_mast")


# ------------------------------------------------------- 2. wedges
# Underside female slot rides the mast rail; inclined face carries its own
# male rail (along Y) + end stop for the cradle, plus a tapered-pin hole.
def make_wedge(name, pitch_deg):
    slope = np.radians(90.0 - pitch_deg)
    x0, x1, h_low = 0.0, 42.0, 10.0
    h_high = h_low + (x1 - x0) * np.tan(slope)
    body = prism_xz([(x0, 0), (x1, 0), (x1, h_low), (x0, h_high)], -15, 15)
    slot = prism_xz(female_slot_xz(22.5, 0), -15.2, 15.2)  # mast rail cx 27.5 -> local 22.5

    n = np.array([np.sin(np.radians(pitch_deg)), 0, np.cos(np.radians(pitch_deg))])
    fx = np.array([np.cos(np.radians(pitch_deg)), 0, -np.sin(np.radians(pitch_deg))])
    mid = np.array([(x0 + x1) / 2, 0, (h_low + h_high) / 2])
    # face rail: build along Y at origin, then pitch-rotate and move to face
    rail = prism_xz(male_rail_xz(0, 0), -15, 15)
    stop = prism_xz([(-13, -2), (13, -2), (13, 8), (-13, 8)], 15, 17)
    R = np.eye(4); R[:3, :3] = trimesh.transformations.rotation_matrix(
        np.radians(pitch_deg), [0, 1, 0])[:3, :3]
    for part in (rail, stop):
        part.apply_transform(R)
        part.apply_translation(mid - 0.05 * n)
    body = union([body, rail, stop])
    cuts = [slot,
            cyl(2.6, mid - 10 * fx + 7 * n, mid - 10 * fx - 12 * n),  # cradle pin
            cyl(2.6, [37.5, 0, -0.1], [37.5, 0, 12])]                  # mast pin
    return save(diff(body, cuts), f"wedge_{name}")


# ------------------------------------------------------- 3. cradle
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
    # dovetail boss on the back: slot along X (open toward -x, stop at +x end)
    cx, cy = L / 2, H / 2 + 3
    boss = box(cx - 20, cx + 20, cy - 17, cy + 17, -8, 0)
    body = union([plate, rail_b, rail_l, rail_r, prow, boss] + tabs)
    # female slot: profile in (y, z), mouth at z=-8 flaring toward z=-1.6
    c, h = DT_CLEAR, 6 + DT_CLEAR
    slot = extrude_polygon(Polygon([(cy - 10 - c, -8.1), (cy + 10 + c, -8.1),
                                    (cy + 13 + c, -8.1 + h), (cy - 13 - c, -8.1 + h)]),
                           40.1)
    slot.apply_transform(np.array([[0, 0, 1, cx - 20.2], [1, 0, 0, 0],
                                   [0, 1, 0, 0], [0, 0, 0, 1]], float))
    # slot stops 6 short of the +x end of the boss (end stop material)
    slot.apply_translation([-6, 0, 0])
    cuts = [
        box(24, L - 24, 14, H - 8, -0.1, 5.1),
        box(-0.1, 60, H - 26, H + 0.1, -0.1, rail_top + 6),
        box(cx - 8, cx + 8, -0.1, 6.1, 4.9, rail_top + 6),
        box(52, 61, -0.1, H + 0.1, -0.1, 2),
        box(L - 61, L - 52, -0.1, H + 0.1, -0.1, 2),
        slot,
        cyl(2.6, [cx - 10, cy, -8.1], [cx - 10, cy, 5.1]),   # tapered-pin hole
    ]
    return save(diff(body, cuts), "cradle")


# ------------------------------------------------------- 4. printed clamp
def make_clamp_screw():
    knob = cyl(16, [0, 0, 0], [0, 0, 10])
    notches = [cyl(3.5, [32 / 2 * np.cos(a), 32 / 2 * np.sin(a), -0.1],
                   [32 / 2 * np.cos(a), 32 / 2 * np.sin(a), 10.1])
               for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    collar = cyl(7, [0, 0, 10], [0, 0, 14])
    thread = threaded_rod(THR_MAJ, THR_MIN, THR_PITCH, 30)
    thread.apply_translation([0, 0, 14])
    tip = cyl(3.5, [0, 0, 44], [0, 0, 48])
    ball = trimesh.creation.icosphere(radius=4, subdivisions=3)
    ball.apply_translation([0, 0, 49])
    body = diff(union([knob, collar, thread, tip, ball]), notches)
    return save(body, "clamp_screw")


def make_pad_cap():
    disc = cyl(9, [0, 0, 0], [0, 0, 4])
    face_pocket = cyl(7.6, [0, 0, -0.1], [0, 0, 1.5])       # 68A face disc pocket
    socket = trimesh.creation.icosphere(radius=4.3, subdivisions=3)
    socket.apply_translation([0, 0, 4.6])                    # snap-over ball socket
    entry = cyl(3.4, [0, 0, 3.9], [0, 0, 8.1])
    return save(diff(disc, [face_pocket, socket, entry]), "pad_cap")


def make_pins():
    # tapered retaining pin: dia 5.2 -> 4.6 over 20 (friction fit), plus head
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
    return save(union([frustum, head]), "pin_x2")


# ------------------------------------------------------- 5. TPU 68A parts
def make_tpu():
    save(box(0, 110.5, 0, 12.2, 0, SEAT_T), "tpu_seat_strip")
    save(box(0, 110, 0, 60, 0, PAD_T), "tpu_jaw_pad_x2")
    save(box(0, 16, 0, 16, 0, WALL_T), "tpu_wall_pad_x2")
    save(cyl(7.4, [0, 0, 0], [0, 0, 2.5]), "tpu_swivel_face")
    band = diff(cyl(28, [0, 0, 0], [0, 0, 2.5]), [cyl(24, [0, 0, -0.1], [0, 0, 2.6])])
    save(band, "tpu_band_x2")


if __name__ == "__main__":
    make_saddle_mast()
    for name, ang in WEDGE_ANGLES.items():
        make_wedge(name, ang)
    make_cradle()
    make_clamp_screw()
    make_pad_cap()
    make_pins()
    make_tpu()
    print("\nAll STLs written to", OUT)
