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

    def y_at_x(self, x):
        if self.left.x == self.right.x:
            raise ValueError("Line endpoints must have different x coordinates.")

        slope = (self.right.y - self.left.y) / (self.right.x - self.left.x)
        return self.left.y + slope * (x - self.left.x)

    def contains_x(self, x):
        return min(self.left.x, self.right.x) <= x <= max(self.left.x, self.right.x)


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


def judge_ball(ball, top_line, bottom_line):
    top_y = top_line.y_at_x(ball.x)
    bottom_y = bottom_line.y_at_x(ball.x)

    if top_y >= bottom_y:
        raise ValueError(
            "At the ball x-coordinate, the top line is not above the bottom line. "
            "Check that the line coordinates were entered correctly."
        )

    if ball.y <= top_y:
        return "OUT", "above_or_on_top_line", top_y, bottom_y

    if ball.y >= bottom_y:
        return "OUT", "below_or_on_bottom_line", top_y, bottom_y

    return "IN", "between_lines", top_y, bottom_y


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
