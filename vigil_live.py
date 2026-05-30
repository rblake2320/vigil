"""
vigil_live.py — Real-Time Vigil Monitor
=========================================
Runs the two-stage pipeline and streams results to a browser via SSE.

Start:
    python vigil_live.py [--source 0] [--port 8896] [--conf 0.4]

Then open: http://localhost:8896
"""
import cv2
import sys
import time
import json
import base64
import asyncio
import argparse
import threading
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))
from core.realtime_monitor import RealtimeMonitor, Alert, Detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
_monitor: RealtimeMonitor | None = None
_clients: list[asyncio.Queue] = []
_event_loop: asyncio.AbstractEventLoop | None = None
_latest_frame_b64: str = ""
_latest_stats: dict = {}
_latest_raw_frame = None  # raw numpy frame for capture mode
_capture_state: dict = {"active": False, "label": "", "until": 0.0, "count": 0}

# Focus ROI: normalized (x1,y1,x2,y2) or None for full frame
_focus_roi: tuple | None = None
# Hover mode: mouse position drives ROI
_hover_mode: bool = False
_hover_pos: tuple = (0.5, 0.5)  # normalized (cx, cy)


def _push(event: str, data: dict) -> None:
    """Push SSE event to all connected browsers."""
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    if _event_loop and not _event_loop.is_closed():
        for q in list(_clients):
            try:
                asyncio.run_coroutine_threadsafe(q.put(msg), _event_loop)
            except Exception:
                pass


def on_frame(frame, detections: list[Detection], ms: float) -> None:
    global _latest_frame_b64, _latest_stats, _latest_raw_frame
    _latest_raw_frame = frame
    # Sync ROI into monitor so detect loop can use it
    if _monitor:
        _monitor.focus_roi = _focus_roi
    # Encode frame
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    b64 = base64.b64encode(jpg.tobytes()).decode()
    _latest_frame_b64 = b64

    # Draw boxes
    vis = frame.copy()
    h, w = vis.shape[:2]
    for d in detections:
        x1,y1,x2,y2 = int(d.box[0]*w), int(d.box[1]*h), int(d.box[2]*w), int(d.box[3]*h)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(vis, f"{d.label} {d.confidence:.0%}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)

    _, vis_jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
    vis_b64 = base64.b64encode(vis_jpg.tobytes()).decode()

    if _monitor:
        _latest_stats = _monitor.stats.copy()

    # Only push frame every ~100ms to avoid flooding SSE
    now = time.time()
    if not hasattr(on_frame, "_last_push") or now - on_frame._last_push > 0.10:
        on_frame._last_push = now
        _push("frame", {
            "b64": vis_b64,
            "ms": round(ms, 1),
            "count": len(detections),
            "ts": now,
        })


def on_alert(alert: Alert) -> None:
    det_str = [{"label": d.label, "conf": round(d.confidence, 2)} for d in alert.detections]
    log.info(f"[ALERT] {alert.severity} — {det_str}")
    _push("alert", {
        "severity": alert.severity,
        "detections": det_str,
        "ts": alert.ts,
        "cosmos_pending": True,
    })


_tts_enabled = True
_tts_lock = threading.Lock()
_piper_voice = None

def _load_piper():
    global _piper_voice
    try:
        from piper.voice import PiperVoice
        model  = str(Path.home() / "piper-voices/en_US-ryan-high.onnx")
        config = str(Path.home() / "piper-voices/en_US-ryan-high.onnx.json")
        _piper_voice = PiperVoice.load(model, config_path=config)
        log.info("[TTS] Piper voice loaded (en_US-ryan-high)")
    except Exception as e:
        log.warning(f"[TTS] Piper load failed ({e}) — using espeak fallback")

threading.Thread(target=_load_piper, daemon=True).start()


def _speak(text: str) -> None:
    if not _tts_enabled:
        return
    clean = text.replace("THREAT:", "").replace("SAFE:", "").replace("MONITOR:", "").strip()
    if not clean:
        return
    def _run():
        with _tts_lock:
            try:
                if _piper_voice is not None:
                    import wave, tempfile, os, subprocess
                    import numpy as np
                    # piper-tts 1.4.2+ returns iterator of AudioChunk
                    chunks = []
                    rate = 22050
                    for chunk in _piper_voice.synthesize(clean):
                        rate = chunk.sample_rate
                        chunks.append((chunk.audio_float_array * 32767).astype(np.int16))
                    if not chunks:
                        raise RuntimeError("Piper produced no audio")
                    audio = np.concatenate(chunks)
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.close()
                    with wave.open(tmp.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(rate)
                        wf.writeframes(audio.tobytes())
                    subprocess.run(
                        ["paplay", tmp.name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20
                    )
                    os.unlink(tmp.name)
                else:
                    import subprocess
                    subprocess.run(
                        ["espeak-ng", "-s", "150", "-v", "en-us", clean],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
                    )
            except Exception as e:
                log.warning(f"[TTS] {e}")
    threading.Thread(target=_run, daemon=True).start()


def on_reasoning(alert: Alert, reasoning: str) -> None:
    log.info(f"[VLM] {reasoning[:120]}")
    _push("reasoning", {
        "text": reasoning,
        "severity": alert.severity,
        "ts": time.time(),
        "is_threat": reasoning.upper().startswith("THREAT"),
    })
    _speak(reasoning)


# ── HTML UI ────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Vigil — Real-Time Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#06060e;color:#ddd;font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{background:#0b0b18;border-bottom:1px solid #1a1a2a;padding:8px 14px;display:flex;align-items:center;gap:10px}
.logo{background:linear-gradient(135deg,#76b900,#00c080);width:26px;height:26px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;color:#000}
header h1{font-size:14px;font-weight:600;color:#fff}
#statBar{margin-left:auto;display:flex;gap:14px;font-size:11px;color:#555}
.stat{display:flex;flex-direction:column;align-items:center}
.stat span{font-size:14px;font-weight:700;color:#76b900}
#alertBanner{display:none;background:#c00;color:#fff;padding:6px 14px;font-size:13px;font-weight:700;text-align:center;animation:flash 0.4s infinite alternate}
@keyframes flash{from{background:#c00}to{background:#ff4444}}
.main{flex:1;display:grid;grid-template-columns:1fr 340px;overflow:hidden}
.feed-panel{position:relative;background:#000;overflow:hidden}
#feedImg{width:100%;height:100%;object-fit:contain;display:block}
.feed-overlay{position:absolute;top:6px;left:8px;display:flex;gap:8px}
.badge{font-size:10px;padding:2px 7px;border-radius:8px;font-family:monospace}
.badge.live{background:#c00;color:#fff}
.badge.fps{background:#111;color:#76b900;border:1px solid #2a2a1a}
.badge.ms{background:#111;color:#aaa;border:1px solid #222}
.right-panel{display:flex;flex-direction:column;border-left:1px solid #1a1a2a;overflow:hidden}
.panel-hdr{font-size:10px;color:#444;padding:6px 10px;border-bottom:1px solid #1a1a2a;text-transform:uppercase;letter-spacing:1px;display:flex;align-items:center;justify-content:space-between}
#eventLog{flex:1;overflow-y:auto;padding:8px;display:flex;flex-direction:column;gap:6px}
.ev{padding:8px 10px;border-radius:8px;font-size:12px;line-height:1.5}
.ev.alert{background:#1a0808;border-left:3px solid #c00;color:#ffaaaa}
.ev.reasoning{background:#0e0e1a;border-left:3px solid #76b900;color:#cce}
.ev.threat{background:#200808;border-left:3px solid #ff0000;color:#ff9999}
.ev.safe{background:#08120a;border-left:3px solid #00c060;color:#99ffbb}
.ev.status{background:#0a0a10;border-left:3px solid #333;color:#555;font-size:10px;font-style:italic}
.ev.chat-q{background:#0d0d20;border-left:3px solid #4488ff;color:#aaccff}
.ev.chat-a{background:#0a1a0a;border-left:3px solid #00cc88;color:#aaffcc}
.ev-ts{font-size:9px;color:#333;margin-bottom:3px;font-family:monospace}
.ev-label{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;opacity:0.7}
#chatBar{padding:8px;border-top:1px solid #1a1a2a;display:flex;gap:6px;background:#080810}
#chatInput{flex:1;background:#111;border:1px solid #2a2a3a;color:#eee;padding:6px 10px;border-radius:6px;font-size:12px;font-family:inherit}
#chatInput:focus{outline:none;border-color:#76b900}
#chatSend{background:#76b900;color:#000;border:none;padding:6px 12px;border-radius:6px;font-size:12px;font-weight:700;cursor:pointer}
#chatSend:disabled{opacity:0.4;cursor:default}
#focusBar{padding:6px 10px;border-top:1px solid #1a1a2a;background:#06060e;display:flex;gap:8px;align-items:center;font-size:11px;color:#555}
.fbtn{background:#111;border:1px solid #2a2a2a;color:#aaa;padding:3px 10px;border-radius:4px;font-size:10px;cursor:pointer}
.fbtn.active{border-color:#76b900;color:#76b900}
#focusIndicator{font-size:10px;color:#444;margin-left:auto}
#roiOverlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
#hoverBox{position:absolute;border:2px solid #76b900;border-radius:4px;pointer-events:none;display:none;box-shadow:0 0 8px #76b90066}
</style>
</head>
<body>
<header>
  <div class="logo">V</div>
  <h1>Vigil — Real-Time Monitor</h1>
  <div id="statBar">
    <div class="stat"><span id="sFPS">—</span>FPS</div>
    <div class="stat"><span id="sMS">—</span>ms/frame</div>
    <div class="stat"><span id="sAlerts">0</span>alerts</div>
    <div class="stat"><span id="sFrames">0</span>frames</div>
  </div>
</header>
<div id="alertBanner"></div>
<div class="main">
  <div class="feed-panel" id="feedPanel">
    <img id="feedImg" src="" alt="Connecting...">
    <div id="hoverBox"></div>
    <div class="feed-overlay">
      <div class="badge live" id="liveBadge">● LIVE</div>
      <div class="badge fps" id="fpsBadge">—fps</div>
      <div class="badge ms" id="msBadge">—ms</div>
    </div>
  </div>
  <div class="right-panel">
    <div class="panel-hdr">
      <span>Events — YOLO + Cosmos</span>
    </div>
    <div id="eventLog">
      <div class="ev status"><div class="ev-ts">--:--:--</div>Connecting to real-time pipeline...</div>
    </div>
    <div id="focusBar">
      <button class="fbtn" id="btnFull" onclick="setFocus('full')">Full Frame</button>
      <button class="fbtn" id="btnDraw" onclick="setFocus('draw')">Draw ROI</button>
      <button class="fbtn" id="btnHover" onclick="setFocus('hover')">Hover Mode</button>
      <span id="focusIndicator">focus: full</span>
    </div>
    <div id="chatBar">
      <input id="chatInput" type="text" placeholder="Ask about the scene..." onkeydown="if(event.key==='Enter')sendChat()">
      <button id="chatSend" onclick="sendChat()">Ask</button>
    </div>
  </div>
</div>

<script>
const es = new EventSource('/events');
let frameCount = 0, lastFrameTs = Date.now(), alertCount = 0;

function ts(epoch) {
  return epoch ? new Date(epoch*1000).toLocaleTimeString() : new Date().toLocaleTimeString();
}

function addEvent(cls, label, text, epoch) {
  const log = document.getElementById('eventLog');
  const div = document.createElement('div');
  div.className = 'ev ' + cls;
  div.innerHTML = `<div class="ev-ts">${ts(epoch)}</div><div class="ev-label">${label}</div>${esc(text)}`;
  log.prepend(div);
  // Keep last 50
  while (log.children.length > 50) log.removeChild(log.lastChild);
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

es.addEventListener('frame', e => {
  const d = JSON.parse(e.data);
  document.getElementById('feedImg').src = 'data:image/jpeg;base64,' + d.b64;
  document.getElementById('msBadge').textContent = d.ms + 'ms';
  frameCount++;
  const now = Date.now();
  if (now - lastFrameTs >= 1000) {
    const fps = Math.round(frameCount * 1000 / (now - lastFrameTs));
    document.getElementById('sFPS').textContent = fps;
    document.getElementById('fpsBadge').textContent = fps + 'fps';
    document.getElementById('sMS').textContent = d.ms;
    frameCount = 0; lastFrameTs = now;
  }
  document.getElementById('sFrames').textContent = d.count > 0
    ? d.count + ' detected' : 'watching';
});

es.addEventListener('alert', e => {
  const d = JSON.parse(e.data);
  alertCount++;
  document.getElementById('sAlerts').textContent = alertCount;
  const dets = d.detections.map(x => `${x.label} ${Math.round(x.conf*100)}%`).join(', ');
  addEvent('alert', '⚡ YOLO TRIGGER — ' + d.severity, dets + ' • Cosmos reasoning...', d.ts);
  // Flash banner
  const banner = document.getElementById('alertBanner');
  banner.style.display = 'block';
  banner.textContent = '⚡ ' + d.severity + ': ' + dets;
  setTimeout(() => banner.style.display = 'none', 4000);
});

es.addEventListener('reasoning', e => {
  const d = JSON.parse(e.data);
  const cls = d.is_threat ? 'threat' : 'reasoning';
  addEvent(cls, '🟢 VIGIL', d.text, d.ts);
});

es.addEventListener('status', e => {
  const d = JSON.parse(e.data);
  addEvent('status', 'status', d.msg, d.ts);
});

es.addEventListener('chat', e => {
  const d = JSON.parse(e.data);
  addEvent('chat-a', '💬 ANSWER', d.text, null);
  document.getElementById('chatSend').disabled = false;
  document.getElementById('chatInput').disabled = false;
});

es.onerror = () => addEvent('status', 'status', 'SSE disconnected — retrying...', null);

// ── Chat ────────────────────────────────────────────────────────────────────
async function sendChat() {
  const inp = document.getElementById('chatInput');
  const q = inp.value.trim();
  if (!q) return;
  inp.value = '';
  document.getElementById('chatSend').disabled = true;
  inp.disabled = true;
  addEvent('chat-q', '❓ YOU', q, null);
  await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question: q})});
}

// ── Focus / ROI / Hover ──────────────────────────────────────────────────────
let focusMode = 'full';
let drawing = false, drawStart = null;

function setFocus(mode) {
  focusMode = mode;
  ['btnFull','btnDraw','btnHover'].forEach(id => document.getElementById(id).classList.remove('active'));
  document.getElementById('btn' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');

  const panel = document.getElementById('feedPanel');
  const box   = document.getElementById('hoverBox');

  if (mode === 'full') {
    fetch('/roi/clear', {method:'POST'});
    box.style.display = 'none';
    document.getElementById('focusIndicator').textContent = 'focus: full frame';
  } else if (mode === 'draw') {
    document.getElementById('focusIndicator').textContent = 'focus: draw box on feed';
    box.style.display = 'none';
  } else if (mode === 'hover') {
    fetch('/roi/clear', {method:'POST'});
    document.getElementById('focusIndicator').textContent = 'focus: hover (move mouse over feed)';
    box.style.display = 'block';
  }
}

// Draw ROI on feed
const panel = document.getElementById('feedPanel');
let roiRect = null;

panel.addEventListener('mousedown', e => {
  if (focusMode !== 'draw') return;
  drawing = true;
  const r = panel.getBoundingClientRect();
  drawStart = {x: (e.clientX - r.left)/r.width, y: (e.clientY - r.top)/r.height};
});

panel.addEventListener('mousemove', e => {
  const r = panel.getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width;
  const ny = (e.clientY - r.top)  / r.height;

  if (focusMode === 'hover') {
    const hw = 0.25, hh = 0.25;
    const x1 = Math.max(0, nx - hw), y1 = Math.max(0, ny - hh);
    const x2 = Math.min(1, nx + hw), y2 = Math.min(1, ny + hh);
    const box = document.getElementById('hoverBox');
    box.style.left   = (x1 * r.width)  + 'px';
    box.style.top    = (y1 * r.height) + 'px';
    box.style.width  = ((x2-x1) * r.width)  + 'px';
    box.style.height = ((y2-y1) * r.height) + 'px';
    // Throttle server update to 10/s
    if (!panel._hoverThrottle || Date.now() - panel._hoverThrottle > 100) {
      panel._hoverThrottle = Date.now();
      fetch('/roi/hover', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({cx: nx, cy: ny, hw: hw, hh: hh})});
    }
  }

  if (focusMode === 'draw' && drawing && drawStart) {
    const box = document.getElementById('hoverBox');
    const x1 = Math.min(drawStart.x, nx), y1 = Math.min(drawStart.y, ny);
    const x2 = Math.max(drawStart.x, nx), y2 = Math.max(drawStart.y, ny);
    box.style.display = 'block';
    box.style.left   = (x1 * r.width)  + 'px';
    box.style.top    = (y1 * r.height) + 'px';
    box.style.width  = ((x2-x1) * r.width)  + 'px';
    box.style.height = ((y2-y1) * r.height) + 'px';
  }
});

panel.addEventListener('mouseup', e => {
  if (focusMode !== 'draw' || !drawing) return;
  drawing = false;
  const r = panel.getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width;
  const ny = (e.clientY - r.top)  / r.height;
  const x1 = Math.min(drawStart.x, nx), y1 = Math.min(drawStart.y, ny);
  const x2 = Math.max(drawStart.x, nx), y2 = Math.max(drawStart.y, ny);
  if (x2-x1 > 0.05 && y2-y1 > 0.05) {
    fetch('/roi/set', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({x1,y1,x2,y2})});
    document.getElementById('focusIndicator').textContent =
      `focus: ROI (${Math.round(x1*100)}%,${Math.round(y1*100)}%) → (${Math.round(x2*100)}%,${Math.round(y2*100)}%)`;
  }
});

// Init
setFocus('full');
</script>
</body>
</html>
"""



@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML


@app.get("/events")
async def events(request: Request):
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    _clients.append(q)

    async def stream():
        try:
            yield "event: status\ndata: {\"msg\": \"Connected to Vigil real-time pipeline\"}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=10)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": ka\n\n"
        finally:
            if q in _clients:
                _clients.remove(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/stats")
async def stats():
    return _latest_stats

@app.post("/tts/toggle")
async def tts_toggle():
    global _tts_enabled
    _tts_enabled = not _tts_enabled
    return {"tts": "on" if _tts_enabled else "off"}

@app.get("/tts/status")
async def tts_status():
    return {"tts": "on" if _tts_enabled else "off"}


@app.get("/training")
async def training_stats():
    from core.realtime_monitor import DataCollector
    import json as _json
    base = DataCollector.BASE
    images = list((base / "images").glob("*.jpg")) if (base / "images").exists() else []
    labels = list((base / "labels").glob("*.txt")) if (base / "labels").exists() else []
    yaml_path = base / "data.yaml"
    classes = {}
    if (base / "meta").exists():
        for f in (base / "meta").glob("*.meta.json"):
            try:
                m = _json.loads(f.read_text())
                for d in m.get("detections", []):
                    lbl = d["label"]
                    classes[lbl] = classes.get(lbl, 0) + 1
            except Exception:
                pass
    return {
        "total_samples": len(images),
        "labeled": len(labels),
        "classes_seen": classes,
        "data_yaml": str(yaml_path) if yaml_path.exists() else None,
        "output_dir": str(base),
    }


from pydantic import BaseModel

class CaptureRequest(BaseModel):
    label: str
    seconds: int = 10
    fps: int = 5  # frames per second to save

@app.post("/capture/start")
async def capture_start(req: CaptureRequest):
    """Start capturing labeled frames. Hold object in front of camera for `seconds`."""
    global _capture_state
    _capture_state = {
        "active": True,
        "label": req.label.strip().lower().replace(" ", "_"),
        "until": time.time() + req.seconds,
        "count": 0,
        "fps": req.fps,
        "last_saved": 0.0,
    }
    threading.Thread(target=_capture_loop, daemon=True).start()
    log.info(f"[Capture] Started — label='{req.label}' for {req.seconds}s")
    return {"status": "capturing", "label": req.label, "seconds": req.seconds}

@app.post("/capture/stop")
async def capture_stop():
    global _capture_state
    saved = _capture_state.get("count", 0)
    _capture_state["active"] = False
    return {"status": "stopped", "saved": saved}

@app.get("/capture/status")
async def capture_status():
    s = _capture_state
    remaining = max(0.0, s.get("until", 0) - time.time()) if s.get("active") else 0
    return {"active": s.get("active", False), "label": s.get("label", ""), "saved": s.get("count", 0), "remaining_s": round(remaining, 1)}

def _capture_loop():
    from core.realtime_monitor import DataCollector, Detection, Alert
    import numpy as np
    collector = DataCollector(enabled=True)
    interval = 1.0 / _capture_state.get("fps", 5)
    while _capture_state.get("active") and time.time() < _capture_state.get("until", 0):
        now = time.time()
        if now - _capture_state.get("last_saved", 0) >= interval:
            frame = _latest_raw_frame
            if frame is not None:
                label = _capture_state["label"]
                # Create a synthetic full-frame detection for this label
                det = Detection(label=label, confidence=1.0, box=(0.05, 0.05, 0.95, 0.95), ts=now)
                alert = Alert(detections=[det], frame=frame.copy(), ts=now)
                collector.collect(alert, reasoning=f"manual capture: {label}")
                _capture_state["count"] = _capture_state.get("count", 0) + 1
                _capture_state["last_saved"] = now
        time.sleep(0.05)
    _capture_state["active"] = False
    log.info(f"[Capture] Done — saved {_capture_state.get('count', 0)} frames for '{_capture_state.get('label')}'")


class AskRequest(BaseModel):
    question: str

class RoiRequest(BaseModel):
    x1: float = 0.0; y1: float = 0.0; x2: float = 1.0; y2: float = 1.0

class HoverRequest(BaseModel):
    cx: float; cy: float; hw: float = 0.25; hh: float = 0.25

@app.post("/ask")
async def ask(req: AskRequest):
    """Ask a question about the current frame — answered by VLM and spoken aloud."""
    global _latest_raw_frame
    frame = _latest_raw_frame
    if frame is None:
        return {"error": "no frame available"}
    import base64, cv2, json as _json, urllib.request
    from core.realtime_monitor import COSMOS_URL, STEP_URL, STEP_PROMPT
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(jpg.tobytes()).decode()
    prompt = f"Looking at this image: {req.question.strip()} Answer in one clear sentence."
    def _do():
        try:
            payload = {"model": "step-3.7-flash", "messages": [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
                {"type":"text","text":prompt}]}], "max_tokens":80, "temperature":0.2}
            r = urllib.request.urlopen(urllib.request.Request(
                STEP_URL, data=_json.dumps(payload).encode(),
                headers={"Content-Type":"application/json","Authorization":"Bearer local"}), timeout=12)
            msg = _json.loads(r.read())["choices"][0]["message"]
            answer = (msg.get("content") or msg.get("reasoning_content","")).strip()
        except Exception:
            try:
                payload = {"model":"nvidia/cosmos-reason2-8b","messages":[{"role":"user","content":[
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
                    {"type":"text","text":prompt}]}],"max_tokens":80,"temperature":0.2}
                r = urllib.request.urlopen(urllib.request.Request(
                    COSMOS_URL, data=_json.dumps(payload).encode(),
                    headers={"Content-Type":"application/json","Authorization":"Bearer local"}), timeout=20)
                answer = _json.loads(r.read())["choices"][0]["message"]["content"].strip()
            except Exception as e:
                answer = f"Could not answer: {e}"
        log.info(f"[Chat] Q: {req.question[:60]} → A: {answer[:80]}")
        _push("chat", {"text": answer})
        _speak(answer)
    threading.Thread(target=_do, daemon=True).start()
    return {"status": "processing"}

@app.post("/roi/set")
async def roi_set(req: RoiRequest):
    global _focus_roi
    _focus_roi = (req.x1, req.y1, req.x2, req.y2)
    log.info(f"[ROI] Set to {_focus_roi}")
    return {"roi": _focus_roi}

@app.post("/roi/clear")
async def roi_clear():
    global _focus_roi
    _focus_roi = None
    log.info("[ROI] Cleared — full frame")
    return {"roi": None}

@app.post("/roi/hover")
async def roi_hover(req: HoverRequest):
    global _focus_roi
    x1 = max(0.0, req.cx - req.hw); y1 = max(0.0, req.cy - req.hh)
    x2 = min(1.0, req.cx + req.hw); y2 = min(1.0, req.cy + req.hh)
    _focus_roi = (x1, y1, x2, y2)
    return {"roi": _focus_roi}


def _start_monitor(source, confidence):
    global _monitor
    _monitor = RealtimeMonitor(source=source, confidence=confidence, cooldown_s=8.0, fps_cap=30)
    _monitor.on_frame     = on_frame
    _monitor.on_alert     = on_alert
    _monitor.on_reasoning = on_reasoning
    _monitor.start()
    log.info("[Vigil] Monitor running")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  default="http://localhost:8891/stream",
                    help="Camera: 0=webcam, rtsp://..., elgato, or MJPEG URL")
    ap.add_argument("--port",    type=int, default=8896)
    ap.add_argument("--conf",    type=float, default=0.45, help="Detection confidence threshold")
    args = ap.parse_args()

    # Start monitor in background thread
    t = threading.Thread(target=_start_monitor, args=(args.source, args.conf), daemon=True)
    t.start()

    log.info(f"[Vigil] UI → http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
