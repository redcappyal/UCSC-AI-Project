# Cloud deploy (demo-grade)

One VM serves everything: /api/upload, the tracking pipeline (RF-DETR on
CPU), and index.html for the app's webview tabs.

## VM

- 8 vCPU / 16 GB (CPU inference ~275 ms/frame at 960 px; a 30 s rally at
  stride 4 ≈ 225 frames ≈ ~1 min. If rehearsal feels slow, move to a GPU
  box and set TRACKING_BACKEND=torch).
- Ubuntu 22.04+, a DNS A record for the chosen domain.

## Steps

1. `sudo useradd -r -m squash && sudo git clone <repo> /opt/squash-line-calling`
2. `cd /opt/squash-line-calling && python3 -m venv venv && venv/bin/pip install -r requirements.txt`
3. `.env`: ROBOFLOW_API_KEY=... (plus OPENAI_API_KEY for coach text, optional)
4. `sudo cp deploy/squash-line-calling.service /etc/systemd/system/ && sudo systemctl enable --now squash-line-calling`
5. Install Caddy (apt, official repo), edit deploy/Caddyfile with the real
   domain, `sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy`.
6. Check: `curl https://<domain>/api/health` → `{"ok": true, ...}`
7. Point the app at it: `ios/Sources/Config.swift` defaultBase → `https://<domain>`,
   rebuild, upload to TestFlight.

## Access control

Demo-grade: unguessable subdomain, no auth (spec decision). Do not reuse
this setup beyond the demo without adding auth.

## Demo-day flow

1. Mount phone, open the app's Matches tab (webview) → calibrate one run
   from the web flow on-site.
2. Native record → every rally reuses that calibration via
   /api/calibration/latest.
