#!/usr/bin/env bash
# Runs on the Lambda instance as ubuntu. Idempotent.
set -euo pipefail
REF="${1:-main}"

cd "$HOME"
if [ ! -d UCSC-AI-Project ]; then
  git clone --branch "$REF" https://github.com/redcappyal/UCSC-AI-Project
fi
cd UCSC-AI-Project
git fetch origin "$REF" && git checkout "$REF" && git pull --ff-only origin "$REF"

# Reuse Lambda Stack's CUDA torch; requirements.txt leaves torch unpinned so
# pip sees it satisfied. rfdetr==1.8.3 could still resolve its own torch —
# the CUDA gate below catches that.
if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/pip install -q --upgrade pip
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
nohup .venv/bin/python app.py > "$HOME/flask.log" 2>&1 &
for _ in $(seq 1 20); do
  if curl -sf http://127.0.0.1:5188/api/health > /dev/null; then
    echo "Flask up on 127.0.0.1:5188 (log: ~/flask.log)"
    exit 0
  fi
  sleep 1
done
echo "Flask did not come up; tail ~/flask.log:" >&2
tail -20 "$HOME/flask.log" >&2
exit 1
