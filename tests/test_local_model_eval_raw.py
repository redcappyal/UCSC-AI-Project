import importlib
from pathlib import Path


def load_raw_module():
    return importlib.import_module("local_model_eval_raw")


def test_raw_evaluator_uses_separate_outputs_and_forces_raw_mode(monkeypatch):
    module = load_raw_module()
    supplied = type("Args", (), {
        "raw_predictions": False,
        "trajectory_fill_max_gap": 12,
        "trajectory_fill_edge_margin": 24.0,
    })()
    monkeypatch.setattr(module, "_base_parse_args", lambda: supplied)

    args = module.raw_parse_args()

    assert module.VIDEO_OUTPUT_PATH.name == "annotated_output_raw.mp4"
    assert module.CSV_OUTPUT_PATH.name == "ball_coordinates_raw.csv"
    assert args.raw_predictions is True
    assert args.trajectory_fill_max_gap == 0
    assert args.trajectory_fill_edge_margin == 0.0


def test_raw_metadata_lists_every_disabled_postprocessing_stage(monkeypatch):
    module = load_raw_module()
    monkeypatch.setattr(
        module,
        "_base_model_config_summary",
        lambda _args: {"model_id": "model/1"},
    )

    summary = module.raw_model_config_summary(object())

    assert summary["comparison_mode"] == "raw_model_predictions"
    assert summary["stationary_filtering_enabled"] is False
    assert summary["motion_linking_enabled"] is False
    assert summary["track_support_required"] is False
    assert summary["post_inference_confidence_filtering_enabled"] is False
    assert summary["trajectory_filling_enabled"] is False
    assert summary["edge_filtering_enabled"] is False


def test_configure_evaluator_points_at_raw_outputs(monkeypatch):
    module = load_raw_module()

    with monkeypatch.context() as patch:
        patch.setattr(module.evaluator, "VIDEO_OUTPUT_PATH", Path("normal.mp4"))
        patch.setattr(module.evaluator, "CSV_OUTPUT_PATH", Path("normal.csv"))
        configured = module.configure_evaluator()

        assert configured.VIDEO_OUTPUT_PATH == module.VIDEO_OUTPUT_PATH
        assert configured.CSV_OUTPUT_PATH == module.CSV_OUTPUT_PATH
        assert configured.TRAJECTORY_FILL_MAX_GAP == 0
        assert configured.TRAJECTORY_FILL_EDGE_MARGIN == 0.0
