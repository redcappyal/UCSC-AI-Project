"""Run the local ball model and retain its raw returned predictions.

This entry point uses the same video decode and inference configuration as
``local_model_eval.py`` but forces raw-prediction annotation:

* every box returned by the model is drawn;
* every returned box is written to CSV (multiple rows per frame are allowed);
* no dust/stationary filtering;
* no motion linking or track-support requirement;
* no post-inference confidence filtering;
* no trajectory interpolation or edge filtering.

The inference request's ``--inference-confidence`` remains the model/server
threshold that determines which predictions are returned at all.
"""

from pathlib import Path

import local_model_eval as evaluator


ROOT = Path(__file__).resolve().parent
VIDEO_OUTPUT_PATH = ROOT / "annotated_output_raw.mp4"
CSV_OUTPUT_PATH = ROOT / "ball_coordinates_raw.csv"

_base_parse_args = evaluator.parse_args
_base_model_config_summary = evaluator.model_config_summary


def raw_parse_args():
    args = _base_parse_args()
    args.raw_predictions = True
    args.trajectory_fill_max_gap = 0
    args.trajectory_fill_edge_margin = 0.0
    return args


def raw_model_config_summary(args):
    return {
        **_base_model_config_summary(args),
        "comparison_mode": "raw_model_predictions",
        "stationary_filtering_enabled": False,
        "motion_linking_enabled": False,
        "track_support_required": False,
        "post_inference_confidence_filtering_enabled": False,
        "trajectory_filling_enabled": False,
        "edge_filtering_enabled": False,
    }


def configure_evaluator():
    evaluator.VIDEO_OUTPUT_PATH = VIDEO_OUTPUT_PATH
    evaluator.CSV_OUTPUT_PATH = CSV_OUTPUT_PATH
    evaluator.TRAJECTORY_FILL_MAX_GAP = 0
    evaluator.TRAJECTORY_FILL_EDGE_MARGIN = 0.0
    evaluator.parse_args = raw_parse_args
    evaluator.model_config_summary = raw_model_config_summary
    return evaluator


if __name__ == "__main__":
    print(
        "Comparison mode: raw model predictions; all tracking, dust, "
        "trajectory, and edge post-processing disabled.",
        flush=True,
    )
    configure_evaluator().main()
