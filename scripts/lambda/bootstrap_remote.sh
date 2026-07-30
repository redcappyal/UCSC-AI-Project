#!/usr/bin/env bash
# Runs on the Lambda instance as ubuntu. Idempotent.
set -euo pipefail
REF="${1:-main}"

cd "$HOME"
# The clone URL is spelled out here, not sourced from common.sh: this script is
# scp'd to the box and runs there, where common.sh does not exist. Anonymous clone
# — the zero-secrets model needs this repo to stay public.
if [ ! -d UCSC-AI-Project ]; then
  git clone --branch "$REF" https://github.com/redcappyal/UCSC-AI-Project
fi
cd UCSC-AI-Project
git fetch origin "$REF" && git checkout "$REF" && git pull --ff-only origin "$REF"

# The app's default ball backend is the hosted RF-DETR, which needs a Roboflow
# key. lambda_up.sh stages it at ~/.crosscourt-env (over ssh stdin, so it never
# lands in any process's argv) BEFORE this script runs; install it as the repo
# .env, which app.py loads via python-dotenv.
if [ -f "$HOME/.crosscourt-env" ]; then
  install -m 600 "$HOME/.crosscourt-env" .env
fi
# On a GPU box the ONNX fallback path must be told it may use CUDA:
# inference_engine.py pins CPU by default (correct on a Mac, where the CoreML
# EP fragments the graph) and its own comment says a GPU server overrides this
# in .env. Without it the rented GPU sits idle for the rfdetr backend.
if [ ! -f .env ] || ! grep -q '^ONNXRUNTIME_EXECUTION_PROVIDERS=' .env; then
  echo 'ONNXRUNTIME_EXECUTION_PROVIDERS=[CUDAExecutionProvider,CPUExecutionProvider]' >> .env
  chmod 600 .env
fi

# A CLEAN venv — deliberately NOT --system-site-packages. Reusing Lambda Stack's
# CUDA torch looks free but mixes two package sets that were never resolved
# together, and it failed three different ways on a real box (2026-07-30):
#   1. Debian's flatbuffers carries a non-PEP440 version, and pip >=24.1 REFUSES
#      to process any environment containing it;
#   2. Lambda Stack's torch is compiled against NumPy 1.x, so the fresh numpy 2.x
#      pip installs here breaks its C-API ("Numpy is not available");
#   3. pip skips packages the system already has (sklearn) while installing fresh
#      ones it lacks (scipy), leaving an old sklearn against a new scipy
#      (ImportError: line_search_wolfe2).
# Each is a symptom of the same thing, so the fix is the environment, not three
# pins. torch's own PyPI wheel bundles CUDA, so a clean venv still gets a GPU
# build — it costs one large download that datacenter bandwidth absorbs, and in
# exchange every version is resolved by one resolver against one set of
# constraints. The CUDA gate below is what proves the GPU build actually landed.
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
# torch FIRST, pinned, from the CUDA index matching Lambda Stack's DRIVER.
# The driver on this image reports CUDA 12.8, and a wheel built for a newer
# CUDA silently yields torch.cuda.is_available() == False ("driver is too
# old") — the exact failure the gate below exists to catch. Two traps here:
# PyPI's default torch targets a newer driver, and even the cu128 index serves
# a +cu130 build unless the version is pinned. 2.7.0 is what Lambda Stack
# itself ships, so it is known-good against this driver; torchvision must move
# with it or pip resolves 0.27.x and demands torch 2.12. requirements.txt
# leaves torch unpinned, so the install below sees it satisfied and keeps this
# build. Bump these together, and only with a live CUDA-gate run as proof.
.venv/bin/pip install -q "torch==2.7.0" "torchvision==0.22.0" \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -q -r requirements.txt

echo "--- CUDA gate"
.venv/bin/python - <<'PY'
import time
import numpy as np
import torch
assert torch.cuda.is_available(), (
    "CUDA unavailable — torch was clobbered by a dependency. Fix: "
    ".venv/bin/pip install --force-reinstall torch")
import ball_model
runner = ball_model.load_detector()
device = getattr(runner, "device", "cpu")
assert device == "cuda", f"ball model landed on {device!r}, expected cuda"
tile = (np.random.rand(416, 416, 9) * 255).astype("uint8")
runner.run_batch([tile])  # warmup pays graph optimization
start = time.perf_counter()
runner.run_batch([tile])
print(f"CUDA gate OK: {(time.perf_counter() - start) * 1000:.1f} ms/tile on {torch.cuda.get_device_name(0)}")
PY

echo "--- person detector pre-warm (non-fatal)"
.venv/bin/python -c "import person_model; d = person_model.load_person_detector(); print('person detector:', 'ready' if d else 'unavailable')" || true

echo "--- flask"
pkill -f "python app.py" 2>/dev/null || true
sleep 1
# BALL_DETECTOR is deliberately NOT set: app.py's own default (rfdetr) is the
# production backend, and following it here keeps the box's numbers comparable
# to real runs — and keeps tracking that default if the app changes it again.
# Running rfdetr needs a Roboflow key, which lambda_up.sh stages (see below);
# set BALL_DETECTOR=local by hand for a WASB/eval session instead.
nohup .venv/bin/python app.py > "$HOME/flask.log" 2>&1 &
# 60 s, not 20: a cold first boot imports torch + cv2 + rfdetr before /api/health
# answers, and a premature "Bootstrap FAILED" buys a paid debug session on a healthy box.
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:5188/api/health > /dev/null; then
    echo "Flask up on 127.0.0.1:5188 (log: ~/flask.log)"
    exit 0
  fi
  sleep 1
done
echo "Flask did not come up; tail ~/flask.log:" >&2
tail -20 "$HOME/flask.log" >&2
exit 1
