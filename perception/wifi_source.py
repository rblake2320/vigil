#!/usr/bin/env python3
"""
perception/wifi_source.py — WiFi + Camera Fused Presence Source
----------------------------------------------------------------
REAL implementation. No sim. No ESP32 required.

Two data streams fused:
  1. RSSI variance from the connected AP (managed mode, no disconnect)
     — polls `iw dev wlP9s9 station dump` every 0.5s
     — variance spike = motion/presence (body absorbs 6GHz RF)
  2. YOLO camera detections from Vigil live (:8896/events SSE)
     — if camera sees person → presence confirmed from ground truth
     — cross-validates WiFi reading

Contradiction detector:
  Camera: person seen | WiFi: room empty → flags conflict, trusts camera
  Camera: empty       | WiFi: presence   → flags conflict, trusts WiFi with low confidence
  Both agree                             → high confidence

Modes:
  fused   — RSSI + Vigil camera (default, recommended)
  rssi    — RSSI only (no camera)
  monitor — beacon RSSI via scapy monitor mode (disconnects WiFi, use 2nd NIC)
  sim     — simulation (for testing without hardware)

Usage:
    python -u perception/wifi_source.py               # fused mode
    python -u perception/wifi_source.py --mode rssi   # RSSI only
    python -u perception/wifi_source.py --mode sim    # sim
    python -u perception/wifi_source.py | python -u perception/coach.py --describe --stdin
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque

IFACE        = os.environ.get("WIFI_IFACE", "wlP9s9")
VIGIL_URL    = os.environ.get("VIGIL_URL",  "http://localhost:8896")
RSSI_HZ      = 2.0           # polls per second
HISTORY      = 60            # samples (~30s)
VAR_THRESH   = 8.0           # RSSI variance threshold for presence (natural jitter ~2-5 on 6GHz)
VAR_MOTION_LOW  = 5.0        # variance for low motion
VAR_MOTION_HIGH = 12.0       # variance for high motion
EMIT_INTERVAL = 1.0


# ─── RSSI poller (managed mode — no disconnect) ────────────────────────────────

class RSSIPoller:
    def __init__(self, iface: str = IFACE):
        self._iface   = iface
        self._history = deque(maxlen=HISTORY)
        self._rssi    = 0
        self._lock    = threading.Lock()

    def _read_rssi(self) -> int | None:
        """Read RSSI from connected AP via iw station dump."""
        try:
            out = subprocess.check_output(
                ["iw", "dev", self._iface, "station", "dump"],
                stderr=subprocess.DEVNULL, timeout=2, text=True,
            )
            for line in out.splitlines():
                if "signal:" in line and "avg" not in line:
                    val = line.split("signal:")[1].split()[0]
                    return int(val.strip())
        except Exception:
            pass
        # fallback: iwconfig
        try:
            out = subprocess.check_output(
                ["iwconfig", self._iface], stderr=subprocess.DEVNULL, timeout=2, text=True,
            )
            for line in out.splitlines():
                if "Signal level" in line:
                    return int(line.split("Signal level=")[1].split()[0])
        except Exception:
            pass
        return None

    def start(self):
        while True:
            rssi = self._read_rssi()
            if rssi is not None:
                with self._lock:
                    self._history.append(rssi)
                    self._rssi = rssi
            time.sleep(1.0 / RSSI_HZ)

    def stats(self) -> dict:
        with self._lock:
            h = list(self._history)
            rssi = self._rssi
        if len(h) < 5:
            return {"rssi": rssi, "variance": 0.0, "presence": False, "motion": "none"}
        mean = sum(h) / len(h)
        var  = sum((x - mean) ** 2 for x in h) / len(h)
        recent_var = 0.0
        if len(h) >= 10:
            r = h[-10:]
            rm = sum(r) / len(r)
            recent_var = sum((x - rm) ** 2 for x in r) / len(r)
        presence = var > VAR_THRESH
        if recent_var > VAR_MOTION_HIGH:
            motion = "high"
        elif recent_var > VAR_MOTION_LOW:
            motion = "low"
        else:
            motion = "none"
        return {"rssi": rssi, "variance": round(var, 2), "presence": presence, "motion": motion}


# ─── Vigil camera reader (SSE event stream) ───────────────────────────────────

class VigilCameraReader:
    """
    Reads the /events SSE stream from Vigil live.
    Extracts person detections and updates presence state.
    """
    def __init__(self, base_url: str = VIGIL_URL):
        self._base_url   = base_url
        self._person_seen = False
        self._last_seen   = 0.0
        self._objects     = set()
        self._severity    = ""       # last alert severity: HIGH / MEDIUM / LOW
        self._last_alert  = 0.0
        self._lock        = threading.Lock()
        self._ok          = False

    def start(self):
        import urllib.request
        url = f"{self._base_url}/events"
        person_decay = 8.0  # seconds before presence clears without new detection
        while True:
            try:
                req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    self._ok = True
                    buf = ""
                    for raw in resp:
                        line = raw.decode("utf-8", errors="replace").rstrip("\n")
                        if line.startswith("data:"):
                            buf = line[5:].strip()
                        elif line.startswith("event:") or line.startswith(":"):
                            pass  # event type / keepalive — ignore
                        elif line == "" and buf:
                            try:
                                ev = json.loads(buf)
                                self._process_event(ev)
                            except Exception:
                                pass
                            buf = ""
                        # Decay: clear presence if no detection recently
                        with self._lock:
                            if self._person_seen and time.time() - self._last_seen > person_decay:
                                self._person_seen = False
            except Exception as e:
                self._ok = False
                time.sleep(3)

    def _process_event(self, ev: dict):
        # Alert events have detections with labels
        # Frame events only have count (no labels)
        detections = ev.get("detections", [])
        severity = ev.get("severity", "")
        if severity:
            with self._lock:
                self._severity   = severity
                self._last_alert = time.time()
        if detections:
            objects = {d.get("label", d.get("class", "")) for d in detections}
            person = "person" in objects
            with self._lock:
                self._objects = objects
                if person:
                    self._person_seen = True
                    self._last_seen   = time.time()

    def state(self) -> dict:
        with self._lock:
            # Severity decays after 5s — if no new alert, motion = none
            sev = self._severity if time.time() - self._last_alert < 5.0 else ""
            return {
                "camera_person":   self._person_seen,
                "camera_objects":  list(self._objects),
                "camera_ok":       self._ok,
                "camera_severity": sev,
            }


# ─── Fused source ─────────────────────────────────────────────────────────────

class FusedSource:
    """
    Merges WiFi RSSI and camera YOLO.
    Contradiction detection: flags when camera and WiFi disagree.
    Camera detections are ground truth — WiFi is corroborating signal.
    """
    def __init__(self, iface: str = IFACE, vigil_url: str = VIGIL_URL):
        self._rssi   = RSSIPoller(iface)
        self._camera = VigilCameraReader(vigil_url)

    def start_threads(self):
        threading.Thread(target=self._rssi.start,   daemon=True).start()
        threading.Thread(target=self._camera.start, daemon=True).start()
        time.sleep(1.5)  # let first samples arrive
        return self

    def signals(self) -> dict:
        rs  = self._rssi.stats()
        cam = self._camera.state()

        wifi_presence  = rs["presence"]
        cam_presence   = cam["camera_person"]
        cam_ok         = cam["camera_ok"]
        cam_severity   = cam.get("camera_severity", "")

        # Motion: camera severity is ground truth, RSSI fills in when camera offline
        sev_map = {"HIGH": "high", "MEDIUM": "low", "LOW": "low"}
        if cam_ok and cam_severity:
            fused_motion = sev_map.get(cam_severity, "none")
        else:
            fused_motion = rs["motion"]

        # Fusion logic
        if cam_ok:
            fused_presence = cam_presence
            confidence = "high" if cam_presence == wifi_presence else "low"
            contradiction = cam_presence != wifi_presence
        else:
            fused_presence = wifi_presence
            confidence = "medium"
            contradiction = False

        sig = {
            "wifi_presence":       fused_presence,
            "wifi_motion":         fused_motion,
            "wifi_breathing_bpm":  0.0,           # needs CSI hardware
            "wifi_rssi":           rs["rssi"],
            "wifi_variance":       rs["variance"],
            "wifi_source":         "fused",
            "wifi_confidence":     confidence,
            "camera_person":       cam_presence,
            "camera_objects":      cam["camera_objects"],
            "camera_ok":           cam_ok,
        }
        if contradiction:
            if cam_presence and not wifi_presence:
                sig["contradiction"] = "camera:person wifi:empty — RSSI variance too low, person may be stationary"
            else:
                sig["contradiction"] = "wifi:motion camera:empty — motion detected but no person in frame"
            print(f"[wifi] CONTRADICTION: {sig['contradiction']}", file=sys.stderr)

        return sig


# ─── RSSI-only source ─────────────────────────────────────────────────────────

class RSSIOnlySource:
    def __init__(self, iface: str = IFACE):
        self._poller = RSSIPoller(iface)

    def start_threads(self):
        threading.Thread(target=self._poller.start, daemon=True).start()
        time.sleep(1.5)
        return self

    def signals(self) -> dict:
        s = self._poller.stats()
        return {
            "wifi_presence":      s["presence"],
            "wifi_motion":        s["motion"],
            "wifi_breathing_bpm": 0.0,
            "wifi_rssi":          s["rssi"],
            "wifi_variance":      s["variance"],
            "wifi_source":        "rssi",
            "wifi_confidence":    "medium",
        }


# ─── Monitor mode source (needs 2nd NIC or disconnect) ────────────────────────

class MonitorSource:
    """
    Puts NIC in monitor mode and sniffs beacon RSSI via scapy.
    WARNING: disconnects the interface from WiFi network.
    Use only with a dedicated second NIC.
    """
    def __init__(self, iface: str = IFACE):
        self._iface   = iface
        self._history = deque(maxlen=HISTORY)
        self._rssi    = 0
        self._lock    = threading.Lock()

    def _enable_monitor(self):
        cmds = [
            ["sudo", "ip", "link", "set", self._iface, "down"],
            ["sudo", "iw", "dev", self._iface, "set", "type", "monitor"],
            ["sudo", "ip", "link", "set", self._iface, "up"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True)
        print(f"[wifi] {self._iface} → monitor mode", file=sys.stderr)

    def start(self):
        try:
            from scapy.all import sniff, Dot11, RadioTap
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "scapy", "-q"])
            from scapy.all import sniff, Dot11, RadioTap

        self._enable_monitor()

        def _packet(pkt):
            if pkt.haslayer(RadioTap) and pkt.haslayer(Dot11):
                rssi = getattr(pkt[RadioTap], "dBm_AntSignal", None)
                if rssi is not None:
                    with self._lock:
                        self._history.append(int(rssi))
                        self._rssi = int(rssi)

        sniff(iface=self._iface, prn=_packet, store=False)

    def start_threads(self):
        threading.Thread(target=self.start, daemon=True).start()
        time.sleep(2)
        return self

    def signals(self) -> dict:
        with self._lock:
            h = list(self._history)
            rssi = self._rssi
        var = 0.0
        if len(h) >= 5:
            mean = sum(h) / len(h)
            var = sum((x - mean) ** 2 for x in h) / len(h)
        return {
            "wifi_presence":      var > VAR_THRESH,
            "wifi_motion":        "high" if var > VAR_THRESH * 3 else ("low" if var > VAR_THRESH else "none"),
            "wifi_breathing_bpm": 0.0,
            "wifi_rssi":          rssi,
            "wifi_variance":      round(var, 2),
            "wifi_source":        "monitor",
            "wifi_confidence":    "medium",
        }


# ─── Sim source ───────────────────────────────────────────────────────────────

class SimSource:
    def __init__(self):
        self._t = 0.0
        self._proc_hist = deque(maxlen=HISTORY)

    def start_threads(self):
        def _loop():
            while True:
                self._t += 0.5
                if 5.0 < self._t < 60.0:
                    breath = math.sin(2 * math.pi * 0.25 * self._t)
                    rssi = -55 + breath * 3 + random.gauss(0, 1.5)
                else:
                    rssi = -75 + random.gauss(0, 0.3)
                self._proc_hist.append(rssi)
                time.sleep(0.5)
        threading.Thread(target=_loop, daemon=True).start()
        time.sleep(0.5)
        return self

    def signals(self) -> dict:
        h = list(self._proc_hist)
        var = 0.0
        if len(h) >= 5:
            mean = sum(h) / len(h)
            var = sum((x - mean) ** 2 for x in h) / len(h)
        presence = self._t > 5.0 and self._t < 60.0
        return {
            "wifi_presence":      presence,
            "wifi_motion":        "low" if presence else "none",
            "wifi_breathing_bpm": 15.0 if presence else 0.0,
            "wifi_rssi":          int(h[-1]) if h else -80,
            "wifi_variance":      round(var, 2),
            "wifi_source":        "sim",
            "wifi_confidence":    "sim",
        }


# ─── Emit loop ────────────────────────────────────────────────────────────────

def run(source, interval: float = EMIT_INTERVAL):
    print("[wifi] running — Ctrl+C to stop", file=sys.stderr)
    last: dict = {}
    while True:
        sig = source.signals()
        sig["ts"] = time.time()
        changed = any(sig.get(k) != last.get(k)
                      for k in ("wifi_presence", "wifi_motion", "camera_person", "contradiction"))
        if changed:
            print(json.dumps(sig, indent=2, default=str), flush=True)
            print("─" * 60, flush=True)
            last = sig.copy()
        time.sleep(interval)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiFi + Camera fused presence source")
    parser.add_argument("--mode",  choices=["fused", "rssi", "monitor", "sim"],
                        default="fused", help="fused=WiFi+camera (default), rssi=WiFi only, monitor=scapy, sim=test")
    parser.add_argument("--iface", default=IFACE)
    parser.add_argument("--vigil", default=VIGIL_URL)
    parser.add_argument("--interval", type=float, default=EMIT_INTERVAL)
    args = parser.parse_args()

    if args.mode == "fused":
        src = FusedSource(args.iface, args.vigil).start_threads()
        print(f"[wifi] fused mode — RSSI({args.iface}) + Vigil camera({args.vigil})", file=sys.stderr)
    elif args.mode == "rssi":
        src = RSSIOnlySource(args.iface).start_threads()
        print(f"[wifi] RSSI-only mode — {args.iface}", file=sys.stderr)
    elif args.mode == "monitor":
        src = MonitorSource(args.iface).start_threads()
        print(f"[wifi] monitor mode — {args.iface} (WiFi will disconnect)", file=sys.stderr)
    else:
        src = SimSource().start_threads()
        print("[wifi] simulation mode", file=sys.stderr)

    try:
        run(src, args.interval)
    except KeyboardInterrupt:
        print("\n[wifi] stopped", file=sys.stderr)
