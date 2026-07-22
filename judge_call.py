import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV_PATH = Path(__file__).with_name("ball_coordinates.csv")


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Line:
    left: Point
    right: Point

    @property
    def dx(self):
        return self.right.x - self.left.x

    @property
    def dy(self):
        return self.right.y - self.left.y

    @property
    def length(self):
        return (self.dx * self.dx + self.dy * self.dy) ** 0.5

    def y_at_x(self, x):
        if self.left.x == self.right.x:
            raise ValueError("Line endpoints must have different x coordinates.")

        slope = self.dy / self.dx
        return self.left.y + slope * (x - self.left.x)

    def contains_x(self, x):
        return min(self.left.x, self.right.x) <= x <= max(self.left.x, self.right.x)

    def point_at(self, u):
        return Point(self.left.x + self.dx * u, self.left.y + self.dy * u)

    def signed_distance_below(self, point):
        """Perpendicular signed distance: positive is below the left->right line."""
        length = self.length
        if length <= 0:
            raise ValueError("Line endpoints must be distinct.")
        cross = self.dx * (point.y - self.left.y) - self.dy * (point.x - self.left.x)
        return cross / length


@dataclass(frozen=True)
class WallCorners:
    """Front-wall reference corners from the wizard's corner taps.

    Top corners are the OUT LINE's junctions with the side walls (15 ft),
    bottom corners the wall/floor seam — all four have exact court
    coordinates, so they double as camera-pose correspondences
    (court_model._camera_correspondences). Judging uses only the lateral
    bounds, which are height-independent (the side-wall edges are vertical).
    """

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    def _edge_x_at_y(self, top, bottom, y):
        if abs(bottom.y - top.y) <= 1e-9:
            return (top.x + bottom.x) / 2
        t = (y - top.y) / (bottom.y - top.y)
        return top.x + t * (bottom.x - top.x)

    def x_bounds_at_y(self, y):
        left = self._edge_x_at_y(self.top_left, self.bottom_left, y)
        right = self._edge_x_at_y(self.top_right, self.bottom_right, y)
        return (min(left, right), max(left, right))

    def contains_point(self, point):
        left, right = self.x_bounds_at_y(point.y)
        top_y = min(self.top_left.y, self.top_right.y)
        bottom_y = max(self.bottom_left.y, self.bottom_right.y)
        return left <= point.x <= right and top_y <= point.y <= bottom_y

    def normalized_x(self, point):
        left, right = self.x_bounds_at_y(point.y)
        if right <= left:
            return 0.5
        return (point.x - left) / (right - left)

    def outside_distance_px(self, point):
        left, right = self.x_bounds_at_y(point.y)
        top_y = min(self.top_left.y, self.top_right.y)
        bottom_y = max(self.bottom_left.y, self.bottom_right.y)
        dx = max(left - point.x, 0.0, point.x - right)
        dy = max(top_y - point.y, 0.0, point.y - bottom_y)
        return (dx * dx + dy * dy) ** 0.5


def parse_point(value):
    try:
        x_text, y_text = value.split(",", 1)
        return Point(float(x_text.strip()), float(y_text.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected coordinate as x,y but got {value!r}."
        ) from error


def prompt_point(label):
    while True:
        raw_value = input(f"{label} (x,y): ").strip()
        try:
            return parse_point(raw_value)
        except argparse.ArgumentTypeError as error:
            print(error)


def prompt_int(label):
    while True:
        raw_value = input(f"{label}: ").strip()
        try:
            return int(raw_value)
        except ValueError:
            print("Enter an integer frame number.")


def line_from_calibration(line):
    endpoints = line["endpoints"]
    return Line(
        Point(float(endpoints[0][0]), float(endpoints[0][1])),
        Point(float(endpoints[1][0]), float(endpoints[1][1])),
    )


def load_calibration_lines(calibration):
    lines = {line.get("name"): line for line in calibration.get("lines", [])}
    top = lines.get("out_line_lower_edge")
    bottom = lines.get("tin_top_edge")

    if top is None or bottom is None:
        raise ValueError("Calibration must include out_line_lower_edge and tin_top_edge.")

    return line_from_calibration(top), line_from_calibration(bottom)


def load_wall_corners(calibration):
    wall = ((calibration or {}).get("planes") or {}).get("wall") or {}
    corners = wall.get("corners") or []
    by_id = {
        corner.get("id"): corner
        for corner in corners
        if isinstance(corner, dict) and corner.get("tap_px") is not None
    }
    required = ("top_left", "top_right", "bottom_right", "bottom_left")
    if not all(corner_id in by_id for corner_id in required):
        return None

    def point(corner_id):
        tap = by_id[corner_id]["tap_px"]
        return Point(float(tap[0]), float(tap[1]))

    try:
        wall_corners = WallCorners(
            top_left=point("top_left"),
            top_right=point("top_right"),
            bottom_right=point("bottom_right"),
            bottom_left=point("bottom_left"),
        )
    except (TypeError, ValueError, IndexError):
        return None

    left_mid = (wall_corners.top_left.x + wall_corners.bottom_left.x) / 2
    right_mid = (wall_corners.top_right.x + wall_corners.bottom_right.x) / 2
    if right_mid <= left_mid:
        return None
    return wall_corners


def line_x_bounds(line):
    return (
        min(line.left.x, line.right.x),
        max(line.left.x, line.right.x),
    )


def calibration_wall_x_bounds(top_line, bottom_line, frame_width):
    top_min, top_max = line_x_bounds(top_line)
    bottom_min, bottom_max = line_x_bounds(bottom_line)
    left = max(top_min, bottom_min)
    right = min(top_max, bottom_max)

    if right <= left:
        return 0.0, max(1.0, float(frame_width or 1))

    return left, right


def load_ball_positions(csv_path):
    positions = {}
    with csv_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            detected = row.get("detected", "").strip().lower()
            if detected in {"true", "1", "yes"} and row.get("x_center"):
                positions[int(row["source_frame"])] = Point(
                    float(row["x_center"]), float(row["y_center"])
                )
    return positions


def load_ball_position(csv_path, frame):
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if int(row["source_frame"]) != frame:
                continue

            detected = row.get("detected", "").strip().lower()
            if detected not in {"true", "1", "yes"}:
                raise ValueError(f"No ball detection recorded for frame {frame}.")

            return Point(float(row["x_center"]), float(row["y_center"]))

    raise ValueError(f"Frame {frame} was not found in {csv_path}.")


def judge_ball(ball, top_line, bottom_line, wall_corners=None):
    try:
        top_y = top_line.y_at_x(ball.x)
        bottom_y = bottom_line.y_at_x(ball.x)
    except ValueError:
        top_y = bottom_y = None

    top_margin = top_line.signed_distance_below(ball)
    bottom_margin = -bottom_line.signed_distance_below(ball)

    top_mid = top_line.point_at(0.5)
    bottom_mid = bottom_line.point_at(0.5)
    if top_line.signed_distance_below(bottom_mid) <= 0 or bottom_line.signed_distance_below(top_mid) >= 0:
        raise ValueError(
            "The calibrated top/bottom lines do not form a valid wall band. "
            "Check that the line coordinates were entered correctly."
        )

    if wall_corners is not None and not wall_corners.contains_point(ball):
        return "OUT", "outside_wall_bounds", top_y, bottom_y

    if top_margin <= 0:
        return "OUT", "above_or_on_top_line", top_y, bottom_y

    if bottom_margin <= 0:
        return "OUT", "below_or_on_bottom_line", top_y, bottom_y

    return "IN", "between_lines", top_y, bottom_y


def judge_margin_px(ball, top_line, bottom_line, wall_corners=None):
    """Positive when IN, negative when OUT, measured perpendicular to tilted lines."""
    if wall_corners is not None and not wall_corners.contains_point(ball):
        return -wall_corners.outside_distance_px(ball)
    top_margin = top_line.signed_distance_below(ball)
    bottom_margin = -bottom_line.signed_distance_below(ball)
    return min(top_margin, bottom_margin)


def wall_diagram_coordinates(ball, top_line, bottom_line, frame_width=None, wall_corners=None):
    """Map a point into the tilted quadrilateral between calibrated wall lines.

    x is progress along the calibrated wall lines. y is 0 on the out line and
    1 on the tin line, measured along the interpolated connector between them.
    """
    best = None
    # Closed-form bilinear inversion is overkill here and fragile for nearly
    # parallel lines. A dense 1-D search is stable and sub-pixel enough for UI.
    for step in range(1001):
        u = step / 1000
        top = top_line.point_at(u)
        bottom = bottom_line.point_at(u)
        vx = bottom.x - top.x
        vy = bottom.y - top.y
        denom = vx * vx + vy * vy
        if denom <= 1e-9:
            continue
        v = ((ball.x - top.x) * vx + (ball.y - top.y) * vy) / denom
        projected_x = top.x + vx * v
        projected_y = top.y + vy * v
        error = (ball.x - projected_x) ** 2 + (ball.y - projected_y) ** 2
        if best is None or error < best[0]:
            best = (error, u, v)

    if best is None:
        wall_left, wall_right = (
            wall_corners.x_bounds_at_y(ball.y)
            if wall_corners is not None
            else calibration_wall_x_bounds(top_line, bottom_line, frame_width)
        )
        top_y = top_line.y_at_x(ball.x)
        bottom_y = bottom_line.y_at_x(ball.x)
        return {
            "x": (ball.x - wall_left) / (wall_right - wall_left),
            "y": (ball.y - top_y) / (bottom_y - top_y),
            "x_span": [wall_left, wall_right],
            "x_reference": "wall_corners" if wall_corners is not None else "line_span",
        }

    _, u, v = best
    if wall_corners is not None:
        wall_left, wall_right = wall_corners.x_bounds_at_y(ball.y)
        return {
            "x": wall_corners.normalized_x(ball),
            "y": v,
            "x_span": [wall_left, wall_right],
            "x_reference": "wall_corners",
        }
    return {
        "x": u,
        "y": v,
        "x_span": [0.0, 1.0],
        "x_reference": "line_span",
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Call a squash ball IN or OUT from a frame and court-line coordinates."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Ball coordinate CSV path. Defaults to {DEFAULT_CSV_PATH.name}.",
    )
    parser.add_argument("--frame", type=int, help="Source frame to judge.")
    parser.add_argument("--top-left", type=parse_point, help="Top line left endpoint as x,y.")
    parser.add_argument("--top-right", type=parse_point, help="Top line right endpoint as x,y.")
    parser.add_argument(
        "--bottom-left", type=parse_point, help="Bottom line left endpoint as x,y."
    )
    parser.add_argument(
        "--bottom-right", type=parse_point, help="Bottom line right endpoint as x,y."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    csv_path = args.csv
    frame = args.frame if args.frame is not None else prompt_int("Frame to judge")

    # top_left = args.top_left or prompt_point("Top line left endpoint")
    # top_right = args.top_right or prompt_point("Top line right endpoint")
    # bottom_left = args.bottom_left or prompt_point("Bottom line left endpoint")
    # bottom_right = args.bottom_right or prompt_point("Bottom line right endpoint")

    top_left = Point(0.0, 18.77)
    top_right = Point(1919, 63.9)
    bottom_left = Point(0.0, 631.8)
    bottom_right = Point(1919, 618.74)

    top_line = Line(top_left, top_right)
    bottom_line = Line(bottom_left, bottom_right)
    ball = load_ball_position(csv_path, frame)
    call, reason, top_y, bottom_y = judge_ball(ball, top_line, bottom_line)

    print(f"Frame: {frame}")
    print(f"Ball center: ({ball.x:.3f}, {ball.y:.3f})")
    print(f"Top line y at ball x: {top_y:.3f}")
    print(f"Bottom line y at ball x: {bottom_y:.3f}")

    if not top_line.contains_x(ball.x) or not bottom_line.contains_x(ball.x):
        print("Warning: ball x-coordinate is outside at least one provided line segment.")

    print(f"Call: {call}")
    print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
