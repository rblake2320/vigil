#!/usr/bin/env python3
"""
perception/wifi_ui.py — Real-time WiFi sensing dashboard
----------------------------------------------------------
FastAPI server with WebSocket push. Reads from wifi_source.py via stdin
or runs its own sim internally.

Usage:
    # Standalone sim (no hardware)
    python -u perception/wifi_ui.py --mode sim

    # Pipe from real wifi_source
    python -u perception/wifi_source.py --mode csi | python -u perception/wifi_ui.py --stdin

    # With port
    python -u perception/wifi_ui.py --mode sim --port 8897

UI: http://192.168.1.72:8897
"""

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Set

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:
    print("Installing fastapi + uvicorn...")
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"], check=True)
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

app = FastAPI()
_clients: Set[WebSocket] = set()
_latest: dict = {
    "wifi_presence": False,
    "wifi_motion": "none",
    "wifi_breathing_bpm": 0.0,
    "wifi_rssi": 0,
    "wifi_source": "---",
    "ts": 0,
}
_bcast_queue: asyncio.Queue = asyncio.Queue()

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WiFi Sensing — Vigil</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 32px 16px;
  }
  h1 {
    font-size: 1.1rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 40px;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    width: 100%;
    max-width: 700px;
  }
  .card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 28px 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    transition: border-color 0.3s;
  }
  .card.active { border-color: #4ade80; }
  .card.warning { border-color: #facc15; }
  .label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555;
  }
  .value {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    transition: color 0.3s;
  }
  .sub { font-size: 0.85rem; color: #555; }

  /* Presence card */
  #presence-card { grid-column: 1 / -1; }
  #presence-icon {
    font-size: 5rem;
    transition: all 0.4s;
    filter: grayscale(1) opacity(0.3);
  }
  #presence-icon.on { filter: none; }
  #presence-text { font-size: 1.6rem; font-weight: 600; color: #555; transition: color 0.3s; }
  #presence-text.on { color: #4ade80; }

  /* Motion bar */
  .motion-bar {
    width: 100%;
    height: 8px;
    background: #1e1e2e;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 4px;
  }
  .motion-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease, background 0.3s;
    width: 0%;
    background: #4ade80;
  }

  /* Breathing waveform */
  #breath-canvas {
    width: 100%;
    height: 60px;
    border-radius: 8px;
    background: #0d0d14;
  }

  /* RSSI meter */
  .rssi-bars {
    display: flex;
    gap: 4px;
    align-items: flex-end;
    height: 40px;
  }
  .rssi-bar {
    width: 12px;
    border-radius: 3px 3px 0 0;
    background: #1e1e2e;
    transition: background 0.3s;
  }
  .rssi-bar.lit { background: #4ade80; }

  /* Source badge */
  #source-badge {
    grid-column: 1 / -1;
    text-align: center;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #333;
    padding-top: 8px;
  }
  #conn-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #ef4444;
    margin-right: 6px;
    vertical-align: middle;
    transition: background 0.3s;
  }
  #conn-dot.live { background: #4ade80; }
</style>
</head>
<body>
<h1>WiFi Sensing — Vigil</h1>

<div class="grid">

  <!-- Presence -->
  <div class="card" id="presence-card">
    <div class="label">Presence</div>
    <div id="presence-icon">🧍</div>
    <div id="presence-text">ROOM EMPTY</div>
  </div>

  <!-- Motion -->
  <div class="card" id="motion-card">
    <div class="label">Motion</div>
    <div class="value" id="motion-value" style="color:#555">—</div>
    <div class="motion-bar"><div class="motion-fill" id="motion-fill"></div></div>
  </div>

  <!-- Breathing -->
  <div class="card" id="breath-card">
    <div class="label">Breathing</div>
    <div class="value" id="bpm-value" style="color:#555">—</div>
    <div class="sub">BPM</div>
    <canvas id="breath-canvas"></canvas>
  </div>

  <!-- RSSI -->
  <div class="card" id="rssi-card">
    <div class="label">Signal Strength</div>
    <div class="value" id="rssi-value" style="color:#555">—</div>
    <div class="sub">dBm</div>
    <div class="rssi-bars" id="rssi-bars">
      <div class="rssi-bar" style="height:20%"></div>
      <div class="rssi-bar" style="height:35%"></div>
      <div class="rssi-bar" style="height:55%"></div>
      <div class="rssi-bar" style="height:75%"></div>
      <div class="rssi-bar" style="height:100%"></div>
    </div>
  </div>

  <!-- Heart rate + fall -->
  <div class="card" id="hr-card">
    <div class="label">Heart Rate</div>
    <div class="value" id="hr-value" style="color:#555">—</div>
    <div class="sub">BPM</div>
  </div>
  <div class="card" id="fall-card">
    <div class="label">Fall Detected</div>
    <div class="value" id="fall-value" style="color:#555;font-size:2rem">—</div>
    <div class="sub" id="persons-sub"></div>
  </div>

  <!-- Camera ground truth -->
  <div class="card" id="camera-card" style="grid-column:1/-1;flex-direction:row;justify-content:space-between;padding:16px 24px;">
    <div style="display:flex;flex-direction:column;gap:4px">
      <div class="label">Camera (YOLO)</div>
      <div id="camera-text" style="font-size:1rem;color:#555">—</div>
    </div>
    <div id="contradiction-badge" style="display:none;background:#ef444422;border:1px solid #ef4444;border-radius:8px;padding:8px 14px;font-size:0.75rem;color:#ef4444;max-width:55%;text-align:right;line-height:1.4"></div>
  </div>

  <div id="source-badge">
    <span id="conn-dot"></span>
    <span id="source-text">connecting...</span>
  </div>

</div>

<script>
const bpmHistory = Array(80).fill(0);
let canvas, ctx;

function initCanvas() {
  canvas = document.getElementById('breath-canvas');
  ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * window.devicePixelRatio;
  canvas.height = canvas.offsetHeight * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

function drawBreath(bpm) {
  const w = canvas.offsetWidth, h = canvas.offsetHeight;
  ctx.clearRect(0, 0, w, h);
  if (!bpm) return;

  // Push synthetic waveform based on BPM
  const t = Date.now() / 1000;
  const val = Math.sin(2 * Math.PI * (bpm / 60) * t);
  bpmHistory.push((val + 1) / 2);
  bpmHistory.shift();

  ctx.beginPath();
  ctx.strokeStyle = '#4ade80';
  ctx.lineWidth = 1.5;
  bpmHistory.forEach((v, i) => {
    const x = (i / (bpmHistory.length - 1)) * w;
    const y = h - v * (h * 0.8) - h * 0.1;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(74,222,128,0.15)');
  grad.addColorStop(1, 'rgba(74,222,128,0)');
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();
}

function rssiStrength(dbm) {
  if (dbm >= -50) return 5;
  if (dbm >= -60) return 4;
  if (dbm >= -70) return 3;
  if (dbm >= -80) return 2;
  return 1;
}

function update(d) {
  // Presence
  const presCard = document.getElementById('presence-card');
  const icon = document.getElementById('presence-icon');
  const presText = document.getElementById('presence-text');
  // Camera is ground truth — if camera sees person, show present regardless of RSSI
  const effectivePresence = (d.camera_ok && d.camera_person) || d.wifi_presence;
  if (effectivePresence) {
    presCard.classList.add('active'); presCard.classList.remove('warning');
    icon.classList.add('on');
    presText.classList.add('on');
    presText.textContent = 'PERSON PRESENT';
  } else {
    presCard.classList.remove('active', 'warning');
    icon.classList.remove('on');
    presText.classList.remove('on');
    presText.textContent = 'ROOM EMPTY';
  }

  // Motion
  const motionMap = { none: 0, low: 40, high: 100 };
  const motionColor = { none: '#555', low: '#facc15', high: '#ef4444' };
  const motionVal = document.getElementById('motion-value');
  const motionFill = document.getElementById('motion-fill');
  const motionCard = document.getElementById('motion-card');
  motionVal.textContent = d.wifi_motion.toUpperCase();
  motionVal.style.color = motionColor[d.wifi_motion] || '#555';
  motionFill.style.width = (motionMap[d.wifi_motion] || 0) + '%';
  motionFill.style.background = motionColor[d.wifi_motion] || '#555';
  if (d.wifi_motion === 'high') motionCard.classList.add('warning');
  else motionCard.classList.remove('warning');

  // Breathing
  const bpm = d.wifi_breathing_bpm;
  const bpmEl = document.getElementById('bpm-value');
  const bCard = document.getElementById('breath-card');
  if (bpm) {
    bpmEl.textContent = bpm.toFixed(1);
    bpmEl.style.color = '#4ade80';
    bCard.classList.add('active');
  } else {
    bpmEl.textContent = '—';
    bpmEl.style.color = '#555';
    bCard.classList.remove('active');
  }
  drawBreath(bpm);

  // RSSI
  const rssi = d.wifi_rssi;
  document.getElementById('rssi-value').textContent = rssi ? rssi : '—';
  document.getElementById('rssi-value').style.color = rssi ? '#4ade80' : '#555';
  const bars = document.querySelectorAll('.rssi-bar');
  const strength = rssiStrength(rssi);
  bars.forEach((b, i) => {
    b.classList.toggle('lit', i < strength);
  });

  // Heart rate
  const hr = d.wifi_heartrate_bpm;
  const hrEl = document.getElementById('hr-value');
  if (hr) { hrEl.textContent = hr.toFixed(1); hrEl.style.color = '#4ade80'; }
  else { hrEl.textContent = '—'; hrEl.style.color = '#555'; }

  // Fall + persons
  const fallEl = document.getElementById('fall-value');
  const fallCard = document.getElementById('fall-card');
  if (d.wifi_fall) {
    fallEl.textContent = '⚠ FALL'; fallEl.style.color = '#ef4444';
    fallCard.classList.add('warning');
  } else {
    fallEl.textContent = 'CLEAR'; fallEl.style.color = '#4ade80';
    fallCard.classList.remove('warning');
  }
  const n = d.wifi_n_persons || 0;
  document.getElementById('persons-sub').textContent = `${n} person${n===1?'':'s'} detected`;

  // Camera ground truth
  const camEl = document.getElementById('camera-text');
  const contrEl = document.getElementById('contradiction-badge');
  if (d.camera_ok === false) {
    camEl.textContent = 'offline'; camEl.style.color = '#555';
  } else {
    const objs = (d.camera_objects || []).join(', ') || 'nothing detected';
    camEl.textContent = d.camera_person ? `Person detected · ${objs}` : objs;
    camEl.style.color = d.camera_person ? '#4ade80' : '#888';
  }
  if (d.contradiction) {
    contrEl.style.display = 'block';
    contrEl.textContent = '⚠ ' + d.contradiction;
  } else {
    contrEl.style.display = 'none';
  }

  // Confidence
  const confColor = {high:'#4ade80', medium:'#facc15', low:'#ef4444', sim:'#888'};

  // Source
  const src = d.wifi_source;
  const conf = d.wifi_confidence || '';
  document.getElementById('source-text').textContent =
    `source: ${src} · confidence: ${conf} · ${new Date(d.ts * 1000).toLocaleTimeString()}`;
  document.getElementById('conn-dot').classList.add('live');
}

// WebSocket
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = e => update(JSON.parse(e.data));
  ws.onclose = () => {
    document.getElementById('conn-dot').classList.remove('live');
    setTimeout(connect, 2000);
  };
}

window.onload = () => { initCanvas(); connect(); setInterval(() => drawBreath(0), 50); };
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        await ws.send_json(_latest)
        # Heartbeat — resend current state every 3s so clients never stay stale
        while True:
            await asyncio.sleep(3)
            await ws.send_json(_latest)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.discard(ws)

async def _broadcast(data: dict):
    dead = set()
    for ws in _clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)

_loop: asyncio.AbstractEventLoop | None = None

def _push(data: dict):
    _latest.update(data)
    if _loop and not _loop.is_closed():
        # call_soon_threadsafe + put_nowait is the correct cross-thread pattern
        _loop.call_soon_threadsafe(_bcast_queue.put_nowait, dict(data))

@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_running_loop()
    asyncio.create_task(_drain_queue())

async def _drain_queue():
    while True:
        data = await _bcast_queue.get()
        await _broadcast(data)

# ─── Signal ingestion ─────────────────────────────────────────────────────────

def _read_stdin():
    buf = ""
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.startswith("─"):
            if buf.strip():
                try:
                    data = json.loads(buf)
                    if "wifi_presence" in data:
                        _push(data)
                except Exception:
                    pass
            buf = ""
        else:
            buf += line + "\n"

def _run_sim():
    sys.path.insert(0, str(Path(__file__).parent))
    from wifi_source import SimSource
    src = SimSource().run()
    while True:
        sig = src.signals()
        sig["ts"] = time.time()
        _push(sig)
        time.sleep(1.0)


def _run_ruview(rest_url: str = "http://localhost:3000"):
    sys.path.insert(0, str(Path(__file__).parent))
    from ruview_source import RuViewRESTSource
    src = RuViewRESTSource(rest_url).start_threads()
    import time as _time
    last: dict = {}
    print(f"[wifi-ui] RuView source: {rest_url}", flush=True)
    while True:
        sig = src.signals()
        sig["ts"] = _time.time()
        changed = any(sig.get(k) != last.get(k)
                      for k in ("wifi_presence", "wifi_motion", "wifi_breathing_bpm",
                                "wifi_fall", "wifi_n_persons"))
        if changed:
            _push(sig)
            last = sig.copy()
        _time.sleep(1.0)


def _run_fused(iface: str = "wlP9s9", vigil_url: str = "http://localhost:8896"):
    sys.path.insert(0, str(Path(__file__).parent))
    from wifi_source import FusedSource
    src = FusedSource(iface, vigil_url).start_threads()
    import time as _time
    last: dict = {}
    print("[wifi-ui] fused source running", flush=True)
    while True:
        sig = src.signals()
        sig["ts"] = _time.time()
        changed = any(sig.get(k) != last.get(k)
                      for k in ("wifi_presence", "wifi_motion", "camera_person",
                                "camera_severity", "contradiction"))
        if changed:
            _push(sig)
            last = sig.copy()
        _time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiFi sensing UI server")
    parser.add_argument("--mode",  choices=["sim", "stdin", "fused", "ruview"], default="ruview")
    parser.add_argument("--port",  type=int, default=8897)
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--iface", default="wlP9s9")
    parser.add_argument("--vigil", default="http://localhost:8896")
    parser.add_argument("--stdin", action="store_true", dest="use_stdin")
    args = parser.parse_args()

    use_stdin = args.use_stdin or args.mode == "stdin"

    if use_stdin:
        t = threading.Thread(target=_read_stdin, daemon=True)
        mode_str = "stdin"
    elif args.mode == "sim":
        t = threading.Thread(target=_run_sim, daemon=True)
        mode_str = "sim"
    elif args.mode == "fused":
        t = threading.Thread(target=_run_fused, args=(args.iface, args.vigil), daemon=True)
        mode_str = f"fused ({args.iface} + {args.vigil})"
    else:
        t = threading.Thread(target=_run_ruview, daemon=True)
        mode_str = "ruview (Docker sensing-server)"
    t.start()

    print(f"[wifi-ui] http://{args.host}:{args.port}")
    print(f"[wifi-ui] mode: {mode_str}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
