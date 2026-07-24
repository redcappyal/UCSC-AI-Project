#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v caffeinate >/dev/null 2>&1 && [[ "${SQUASH_CAFFEINATED:-}" != "1" ]]; then
  exec env SQUASH_CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE_TRACK="${FORCE_TRACK:-0}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
HARD_MINING_ROUNDS="${HARD_MINING_ROUNDS:-0}"
RETRAIN_EXISTING_FEATURES_ONLY="${RETRAIN_EXISTING_FEATURES_ONLY:-1}"
TRAJECTORY_FILL_MAX_GAP="${TRAJECTORY_FILL_MAX_GAP:-4}"
TRAJECTORY_FILL_EDGE_MARGIN="${TRAJECTORY_FILL_EDGE_MARGIN:-24}"

if [[ "${FORCE_TRACK}" == "1" || "${FORCE_FEATURES}" == "1" ]]; then
  RETRAIN_EXISTING_FEATURES_ONLY=0
fi

MODELTRAIN_VIDEO="${MODELTRAIN_VIDEO:-ModelTrainTest.mp4}"
MODELTRAIN_LABELS="${MODELTRAIN_LABELS:-wall_hits.csv}"
MODELTRAIN_START_FRAME="${MODELTRAIN_START_FRAME:-2245}"
MODELTRAIN_END_FRAME="${MODELTRAIN_END_FRAME:-47308}"
MODELTRAIN_CALIBRATION="${MODELTRAIN_CALIBRATION:-calibration.json}"

BAYCLUB_VIDEO="${BAYCLUB_VIDEO:-Bay Club Squash 5min+audio.mov}"
BAYCLUB_LABELS="${BAYCLUB_LABELS:-bayclub_wall_hits.csv}"
BAYCLUB_START_FRAME="${BAYCLUB_START_FRAME:-0}"
BAYCLUB_END_FRAME="${BAYCLUB_END_FRAME:-18000}"
BAYCLUB_CALIBRATION="${BAYCLUB_CALIBRATION:-bayclub_calibration.json}"

MATCHPLAY_VIDEO="${MATCHPLAY_VIDEO:-MatchplayEp3Clip2_h264.mp4}"
MATCHPLAY_LABELS="${MATCHPLAY_LABELS:-matchplay_ep3_wall_hits.csv}"
MATCHPLAY_START_FRAME="${MATCHPLAY_START_FRAME:-0}"
MATCHPLAY_END_FRAME="${MATCHPLAY_END_FRAME:-7283}"
MATCHPLAY_CALIBRATION="${MATCHPLAY_CALIBRATION:-calibration.json}"

FEATURE_END_TOLERANCE_FRAMES="${FEATURE_END_TOLERANCE_FRAMES:-120}"

TRACK_FLAGS=(--generate-ball-csv)
if [[ "${FORCE_TRACK:-0}" == "1" ]]; then
  TRACK_FLAGS+=(--force-track)
fi
TRACK_FLAGS+=(
  --trajectory-fill-max-gap "${TRAJECTORY_FILL_MAX_GAP}"
  --trajectory-fill-edge-margin "${TRAJECTORY_FILL_EDGE_MARGIN}"
)

should_generate_features() {
  local feature_csv="$1"
  local expected_end_frame="$2"
  local force_one="${3:-0}"
  if [[ "${FORCE_FEATURES:-0}" == "1" || "${FORCE_TRACK:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${force_one}" == "1" ]]; then
    return 0
  fi
  if [[ ! -s "${feature_csv}" ]]; then
    return 0
  fi

  if "${PYTHON_BIN}" - "$feature_csv" "$expected_end_frame" "$FEATURE_END_TOLERANCE_FRAMES" <<'PY'
import sys
import pandas as pd

feature_csv = sys.argv[1]
expected_end = int(float(sys.argv[2]))
tolerance = int(float(sys.argv[3]))

try:
    features = pd.read_csv(feature_csv, usecols=["frame"])
except Exception as error:
    print(f"{feature_csv}: could not read feature coverage ({error}); rebuilding.")
    sys.exit(1)

if features.empty:
    print(f"{feature_csv}: empty feature file; rebuilding.")
    sys.exit(1)

max_frame = int(features["frame"].max())
required_min = expected_end - tolerance
if max_frame < required_min:
    print(
        f"{feature_csv}: max frame {max_frame} is before requested end {expected_end} "
        f"(tolerance {tolerance}); rebuilding."
    )
    sys.exit(1)

print(f"{feature_csv}: coverage ok through frame {max_frame}; reusing.")
sys.exit(0)
PY
  then
    return 1
  else
    return 0
  fi
}

require_existing_features() {
  local feature_csv="$1"
  if [[ ! -s "${feature_csv}" ]]; then
    echo "${feature_csv} does not exist or is empty."
    echo "Run with RETRAIN_EXISTING_FEATURES_ONLY=0 to build missing features."
    exit 1
  fi
  echo "Retrain-only mode: reusing ${feature_csv}."
}

echo "=== 1/5 ModelTrainTest feature rows ==="
if [[ "${RETRAIN_EXISTING_FEATURES_ONLY}" == "1" ]]; then
  require_existing_features features_modeltraintest.csv
elif should_generate_features features_modeltraintest.csv "${MODELTRAIN_END_FRAME}" "${FORCE_MODELTRAIN_FEATURES:-0}"; then
  "${PYTHON_BIN}" train_bounce_classifier.py \
    --video "${MODELTRAIN_VIDEO}" \
    --labels "${MODELTRAIN_LABELS}" \
    --ball-csv ball_coordinates_modeltraintest.csv \
    --calibration "${MODELTRAIN_CALIBRATION}" \
    --features-output features_modeltraintest.csv \
    --model-output /private/tmp/modeltrain_temp.pkl \
    --start-frame "${MODELTRAIN_START_FRAME}" \
    --end-frame "${MODELTRAIN_END_FRAME}" \
    --include-geometry \
    "${TRACK_FLAGS[@]}"
else
  echo "Reusing existing features_modeltraintest.csv. Set FORCE_FEATURES=1 to rebuild it."
fi

echo "=== 2/5 Bay Club feature rows ==="
if [[ "${RETRAIN_EXISTING_FEATURES_ONLY}" == "1" ]]; then
  require_existing_features features_bayclub.csv
elif should_generate_features features_bayclub.csv "${BAYCLUB_END_FRAME}" "${FORCE_BAYCLUB_FEATURES:-0}"; then
  "${PYTHON_BIN}" train_bounce_classifier.py \
    --video "${BAYCLUB_VIDEO}" \
    --labels "${BAYCLUB_LABELS}" \
    --ball-csv ball_coordinates_bayclub.csv \
    --calibration "${BAYCLUB_CALIBRATION}" \
    --features-output features_bayclub.csv \
    --model-output /private/tmp/bayclub_temp.pkl \
    --start-frame "${BAYCLUB_START_FRAME}" \
    --end-frame "${BAYCLUB_END_FRAME}" \
    --include-geometry \
    "${TRACK_FLAGS[@]}"
else
  echo "Reusing existing features_bayclub.csv. Set FORCE_FEATURES=1 to rebuild it."
fi

echo "=== 3/5 Matchplay feature rows ==="
if [[ "${RETRAIN_EXISTING_FEATURES_ONLY}" == "1" ]]; then
  require_existing_features features_matchplay_ep3_clip2.csv
elif should_generate_features features_matchplay_ep3_clip2.csv "${MATCHPLAY_END_FRAME}" "${FORCE_MATCHPLAY_FEATURES:-0}"; then
  "${PYTHON_BIN}" train_bounce_classifier.py \
    --video "${MATCHPLAY_VIDEO}" \
    --labels "${MATCHPLAY_LABELS}" \
    --ball-csv ball_coordinates_matchplay_ep3_clip2.csv \
    --calibration "${MATCHPLAY_CALIBRATION}" \
    --features-output features_matchplay_ep3_clip2.csv \
    --model-output /private/tmp/matchplay_temp.pkl \
    --start-frame "${MATCHPLAY_START_FRAME}" \
    --end-frame "${MATCHPLAY_END_FRAME}" \
    --include-geometry \
    "${TRACK_FLAGS[@]}"
else
  echo "Reusing existing features_matchplay_ep3_clip2.csv. Set FORCE_FEATURES=1 to rebuild it."
fi

echo "=== 4/5 Combine feature rows from all videos ==="
"${PYTHON_BIN}" -c "
import pandas as pd

modeltrain = pd.read_csv('features_modeltraintest.csv').assign(source_video='ModelTrainTest')
bayclub = pd.read_csv('features_bayclub.csv').assign(source_video='BayClub')
matchplay = pd.read_csv('features_matchplay_ep3_clip2.csv').assign(source_video='MatchplayEp3Clip2')

feature_columns = [set(frame.columns) - {'source_video'} for frame in (modeltrain, bayclub, matchplay)]
if feature_columns[1:] != feature_columns[:-1]:
    raise ValueError('Source feature CSV schemas do not match; rebuild all three with FORCE_FEATURES=1.')

combined = pd.concat([modeltrain, bayclub, matchplay], ignore_index=True)
combined.to_csv('bounce_training_features_combined.csv', index=False)

print(f'ModelTrainTest rows: {len(modeltrain)} ({int(modeltrain[\"is_wall_hit\"].sum())} positive)')
print(f'Bay Club rows: {len(bayclub)} ({int(bayclub[\"is_wall_hit\"].sum())} positive)')
print(f'Matchplay rows: {len(matchplay)} ({int(matchplay[\"is_wall_hit\"].sum())} positive)')
print(f'Combined training rows: {len(combined)} ({int(combined[\"is_wall_hit\"].sum())} positive)')
"

echo "=== 5/5 Train final app model on combined data ==="
"${PYTHON_BIN}" train_bounce_classifier.py \
  --features-input bounce_training_features_combined.csv \
  --features-output bounce_training_features.csv \
  --model-output bounce_gb_model.pkl \
  --calibration "${MODELTRAIN_CALIBRATION}" \
  --source-calibration-map "ModelTrainTest=${MODELTRAIN_CALIBRATION},BayClub=${BAYCLUB_CALIBRATION},MatchplayEp3Clip2=${MATCHPLAY_CALIBRATION}" \
  --hyperparameter-matrix \
  --gb-n-estimators-grid 100,200 \
  --gb-learning-rate-grid 0.05,0.1 \
  --gb-max-depth-grid 2,3 \
  --gb-min-samples-leaf-grid 1,3 \
  --gb-subsample-grid 0.8,1.0 \
  --selection-beta "${SELECTION_BETA:-3.0}" \
  --min-selection-precision "${MIN_SELECTION_PRECISION:-0.45}" \
  --hard-mining-rounds "${HARD_MINING_ROUNDS:-2}" \
  --hard-fp-weight-multiplier "${HARD_FP_WEIGHT_MULTIPLIER:-3.0}" \
  --hard-fn-weight-multiplier "${HARD_FN_WEIGHT_MULTIPLIER:-3.0}" \
  --hit-threshold 0.25

echo "=== Done ==="
echo "Final app model saved to bounce_gb_model.pkl"
echo "Restart python app.py so the app reloads the updated model."
