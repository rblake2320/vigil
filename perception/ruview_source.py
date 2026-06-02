#!/usr/bin/env python3
"""
perception/ruview_source.py — RuView sensing-server source
-----------------------------------------------------------
Connects to the RuView Rust sensing-server WebSocket and translates
EdgeVitalsMessage into our standard signals dict.

Requires the sensing-server running (Docker or native):
  cd ~/ai-business/RuView
  CSI_SOURCE=simulated docker compose -f docker/docker-compose.yml up sensing-server

WebSocket: ws://localhost:3001/ws/sensing
REST:      http://localhost:3000/api/v1/vitals/<node>/latest

Output signals:
  wifi_presence, wifi_motion, wifi_breathing_bpm, wifi_heartrate_bpm,
  wifi_fall, wifi_rssi, wifi_n_persons, wifi_confidence, wifi_source

Usage:
  python -u perception/ruview_source.py
  python -u perception/ruview_source.py | python -u perception/coach.py --describe --stdin
  python -u perception/ruview_source.py --url ws://localhost:3001/ws/sensing
"""

import argparse
import asyncio
import json
import sys
import time
import threading
from collections import deque
from pathlib import Path

# Add RuView python client to path
RUVIEW_PATH = Path(__file__).parent.parent.parent / "RuView" / "python"
if RUVIEW_PATH.exists():
    sys.path.insert(0, str(RUVIEW_PATH))

try:
    from wifi_densepose.client.ws import SensingClient, EdgeVitalsMessage, ConnectionEstablishedMessage
    _CLIENT_OK = True
except ImportError:
    _CLIENT_OK = False


EMIT_INTERVAL = 1.0


class RuViewSource:
    """
    Connects to RuView sensing-server WebSocket, translates EdgeVitalsMessage
    into our standard signals dict. Thread-safe — call signals() from any thread.
    """

    def __init__(self, url: str = "ws://localhost:3001/ws/sensing"):
        self._url      = url
        self._latest   = {
            "wifi_presence":       False,
            "wifi_motion":         "none",
            "wifi_breathing_bpm":  0.0,
            "wifi_heartrate_bpm":  0.0,
            "wifi_fall":           False,
            "wifi_rssi":           0,
            "wifi_n_persons":      0,
            "wifi_confidence":     0.0,
            "wifi_source":         "ruview",
        }
        self._lock     = threading.Lock()
        self._connected = False
        self._node_id  = ""

    def _motion_from_energy(self, energy: float) -> str:
        if energy > 0.6:  return "high"
        if energy > 0.2:  return "low"
        return "none"

    def _translate(self, msg: "EdgeVitalsMessage") -> dict:
        return {
            "wifi_presence":      msg.presence,
            "wifi_motion":        self._motion_from_energy(msg.motion_energy or msg.motion),
            "wifi_breathing_bpm": round(msg.breathing_rate_bpm, 1) if msg.breathing_rate_bpm else 0.0,
            "wifi_heartrate_bpm": round(msg.heartrate_bpm, 1) if msg.heartrate_bpm else 0.0,
            "wifi_fall":          msg.fall_detected,
            "wifi_rssi":          int(msg.rssi) if msg.rssi is not None else 0,
            "wifi_n_persons":     msg.n_persons,
            "wifi_confidence":    round(msg.presence_score, 2),
            "wifi_source":        f"ruview:{self._node_id or 'sim'}",
        }

    async def _run(self):
        while True:
            try:
                print(f"[ruview] connecting to {self._url}", file=sys.stderr, flush=True)
                async with SensingClient(self._url) as client:
                    self._connected = True
                    print("[ruview] connected", file=sys.stderr, flush=True)
                    async for msg in client.stream():
                        if isinstance(msg, ConnectionEstablishedMessage):
                            self._node_id = msg.node_id
                            print(f"[ruview] node={msg.node_id} v={msg.version} caps={msg.capabilities}",
                                  file=sys.stderr, flush=True)
                        elif isinstance(msg, EdgeVitalsMessage):
                            translated = self._translate(msg)
                            with self._lock:
                                self._latest.update(translated)
            except Exception as e:
                self._connected = False
                print(f"[ruview] disconnected: {e} — retry in 3s", file=sys.stderr, flush=True)
                await asyncio.sleep(3)

    def start_threads(self):
        def _loop():
            asyncio.run(self._run())
        threading.Thread(target=_loop, daemon=True).start()
        return self

    def signals(self) -> dict:
        with self._lock:
            return dict(self._latest)


# ─── Fallback: poll REST API instead of WebSocket ────────────────────────────

class RuViewRESTSource:
    """Poll /api/v1/sensing/latest when WS isn't available."""

    def __init__(self, base: str = "http://localhost:3000"):
        self._base   = base
        self._latest = {}
        self._lock   = threading.Lock()

    def start_threads(self):
        import urllib.request
        def _poll():
            while True:
                try:
                    url = f"{self._base}/api/v1/sensing/latest"
                    with urllib.request.urlopen(url, timeout=3) as r:
                        data = json.loads(r.read())
                        # Response has nested classification + features
                        clf  = data.get("classification", data)
                        feat = data.get("features", {})
                        motion_level = clf.get("motion_level", "")
                        if "moving" in motion_level: motion = "high"
                        elif "still" in motion_level or clf.get("presence"): motion = "low"
                        else: motion = "none"
                        # Breathing: dominant_freq_hz in breathing band (0.1–0.5 Hz = 6–30 BPM)
                        dom_hz  = feat.get("dominant_freq_hz", 0)
                        br_bpm  = round(dom_hz * 60, 1) if 0.1 <= dom_hz <= 0.5 else 0.0
                        rssi    = int(feat.get("mean_rssi", 0))
                        with self._lock:
                            self._latest = {
                                "wifi_presence":      bool(clf.get("presence", False)),
                                "wifi_motion":        motion,
                                "wifi_breathing_bpm": br_bpm,
                                "wifi_heartrate_bpm": 0.0,
                                "wifi_fall":          False,
                                "wifi_rssi":          rssi,
                                "wifi_n_persons":     int(data.get("estimated_persons", 0)),
                                "wifi_confidence":    round(float(clf.get("confidence", 0)), 2),
                                "wifi_source":        "ruview:rest",
                            }
                except Exception as e:
                    print(f"[ruview-rest] {e}", file=sys.stderr, flush=True)
                time.sleep(1.0)
        threading.Thread(target=_poll, daemon=True).start()
        return self

    def signals(self) -> dict:
        with self._lock:
            return dict(self._latest)


# ─── Emit loop (same format as spark2_source / wifi_source) ──────────────────

def run(source, interval: float = EMIT_INTERVAL):
    print("[ruview] emitting signals — Ctrl+C to stop", file=sys.stderr, flush=True)
    last: dict = {}
    while True:
        sig = source.signals()
        sig["ts"] = time.time()
        changed = any(
            sig.get(k) != last.get(k)
            for k in ("wifi_presence", "wifi_motion", "wifi_breathing_bpm",
                      "wifi_heartrate_bpm", "wifi_fall")
        )
        if changed:
            print(json.dumps(sig, indent=2), flush=True)
            print("─" * 60, flush=True)
            last = sig.copy()
        time.sleep(interval)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RuView sensing-server source")
    parser.add_argument("--url",  default="ws://localhost:3001/ws/sensing",
                        help="WebSocket URL of RuView sensing-server")
    parser.add_argument("--rest", default="http://localhost:3000",
                        help="REST base URL (fallback if --use-rest)")
    parser.add_argument("--use-rest", action="store_true",
                        help="Use REST polling instead of WebSocket")
    parser.add_argument("--interval", type=float, default=EMIT_INTERVAL)
    args = parser.parse_args()

    if not _CLIENT_OK and not args.use_rest:
        print("[ruview] wifi_densepose client not found — falling back to REST", file=sys.stderr)
        args.use_rest = True

    if args.use_rest:
        src = RuViewRESTSource(args.rest).start_threads()
    else:
        src = RuViewSource(args.url).start_threads()

    try:
        run(src, args.interval)
    except KeyboardInterrupt:
        print("\n[ruview] stopped", file=sys.stderr)
