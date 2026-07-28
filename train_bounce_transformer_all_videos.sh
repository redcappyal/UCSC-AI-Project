#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-"${PROJECT_DIR}/transformer_training"}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-squashanalytics}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
else
  PYTHON_CMD=(python3)
fi

mkdir -p "${OUTPUT_DIR}"

INFERENCE_CONFIDENCE="${BALL_INFERENCE_CONFIDENCE:-0.10}"
CONFIDENCE="${BALL_CONFIDENCE:-0.40}"
INFERENCE_WIDTH="${BALL_INFERENCE_WIDTH:-960}"
TRAJECTORY_MAX_GAP="${BALL_TRAJECTORY_MAX_GAP:-6}"
TRAJECTORY_EDGE_MARGIN="${BALL_TRAJECTORY_EDGE_MARGIN:-24}"
TRANSFORMER_EPOCHS="${TRANSFORMER_EPOCHS:-40}"
TRANSFORMER_DEVICE="${TRANSFORMER_DEVICE:-auto}"
TRANSFORMER_HIT_THRESHOLD="${TRANSFORMER_HIT_THRESHOLD:-0.15}"
FORCE_TRACK="${FORCE_TRACK:-0}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"

run_feature_generation() {
  local source_name="$1"
  local video_path="$2"
  local labels_path="$3"
  local calibration_path="$4"
  local output_slug="$5"
  local ball_csv="${OUTPUT_DIR}/ball_coordinates_${output_slug}.csv"
  local features_csv="${OUTPUT_DIR}/features_${output_slug}.csv"

  echo
  if [[ -s "${features_csv}" && "${FORCE_FEATURES}" != "1" && "${FORCE_TRACK}" != "1" ]]; then
    echo "Using cached bounce features for ${source_name}: ${features_csv}"
    return
  fi

  echo "Generating confirmed ball tracks and bounce features for ${source_name}..."
  echo "Ball-coordinate cache: ${ball_csv}"
  echo "Feature output: ${features_csv}"

  local command=(
    "${PYTHON_CMD[@]}" "${PROJECT_DIR}/train_bounce_classifier.py"
    --video "${video_path}" \
    --labels "${labels_path}" \
    --calibration "${calibration_path}" \
    --ball-csv "${ball_csv}" \
    --features-output "${features_csv}" \
    --model-output "${OUTPUT_DIR}/unused_${output_slug}_gb.pkl" \
    --generate-ball-csv \
    --track-all-frames \
    --features-only \
    --include-geometry \
    --no-stationary-filter \
    --context 4 \
    --positive-window 1 \
    --negative-ratio 6 \
    --inference-width "${INFERENCE_WIDTH}" \
    --inference-confidence "${INFERENCE_CONFIDENCE}" \
    --confidence "${CONFIDENCE}" \
    --trajectory-fill-max-gap "${TRAJECTORY_MAX_GAP}" \
    --trajectory-fill-edge-margin "${TRAJECTORY_EDGE_MARGIN}"
  )

  if [[ "${FORCE_TRACK}" == "1" ]]; then
    command+=(--force-track)
  fi

  "${command[@]}"
}

run_feature_generation \
  "ModelTrainTest" \
  "${PROJECT_DIR}/ModelTrainTest.mp4" \
  "${PROJECT_DIR}/wall_hits.csv" \
  "${PROJECT_DIR}/calibration.json" \
  "modeltraintest"

run_feature_generation \
  "BayClub" \
  "${PROJECT_DIR}/Bay Club Squash 5min+audio.mov" \
  "${PROJECT_DIR}/bayclub_wall_hits.csv" \
  "${PROJECT_DIR}/bayclub_calibration.json" \
  "bayclub"

run_feature_generation \
  "MatchplayEp3Clip2" \
  "${PROJECT_DIR}/MatchplayEp3Clip2_h264.mp4" \
  "${PROJECT_DIR}/matchplay_ep3_wall_hits.csv" \
  "${PROJECT_DIR}/calibration.json" \
  "matchplay_ep3_clip2"

echo
echo "Training the temporal transformer on all three feature sets..."
"${PYTHON_CMD[@]}" "${PROJECT_DIR}/train_bounce_transformer.py" \
  --features "ModelTrainTest=${OUTPUT_DIR}/features_modeltraintest.csv" \
  --features "BayClub=${OUTPUT_DIR}/features_bayclub.csv" \
  --features "MatchplayEp3Clip2=${OUTPUT_DIR}/features_matchplay_ep3_clip2.csv" \
  --calibration "ModelTrainTest=${PROJECT_DIR}/calibration.json" \
  --calibration "BayClub=${PROJECT_DIR}/bayclub_calibration.json" \
  --calibration "MatchplayEp3Clip2=${PROJECT_DIR}/calibration.json" \
  --combined-features-output "${OUTPUT_DIR}/bounce_transformer_features.csv" \
  --model-output "${OUTPUT_DIR}/bounce_transformer_model.pkl" \
  --hit-threshold "${TRANSFORMER_HIT_THRESHOLD}" \
  --epochs "${TRANSFORMER_EPOCHS}" \
  --device "${TRANSFORMER_DEVICE}"

echo
echo "Transformer training complete."
echo "Model: ${OUTPUT_DIR}/bounce_transformer_model.pkl"
echo "To use this artifact in the app, set:"
echo "BOUNCE_MODEL_PATH=${OUTPUT_DIR}/bounce_transformer_model.pkl"
