#!/usr/bin/env python3
"""
perception/wifi_source.py — WiFi CSI Sensing Source
-----------------------------------------------------
Detects human presence, motion, and breathing rate from WiFi signals.
No camera needed. Sees through walls. Range 3-5m.

Modes:
  csi   — Live ESP32-S3 CSI data over UDP (real hardware, port 5500)
  rssi  — Passive RSSI monitoring via NIC in monitor mode (no ESP32)
  sim   — Simulation for testing without any hardware

Hardware for CSI mode:
  ESP32-S3 dev board (~$8), flashed with espressif/esp-csi firmware
  Set WIFI_TARGET_IP to this machine's IP in firmware config

Signal output (JSON to stdout, same format as spark2_source.py):
  {
    "wifi_presence": true,
    "wifi_motion": "high",        # none / low / high
    "wifi_breathing_bpm": 16.4,   # 0 if not detected
    "wifi_rssi": -62,
    "wifi_source": "csi",
    "ts": 1780384000.0
  }

Usage:
    python -u perception/wifi_source.py --mode csi
    python -u perception/wifi_source.py --mode rssi --iface wlan0
    python -u perception/wifi_source.py --mode sim        # no hardware needed
    python -u perception/wifi_source.py --mode sim | python perception/coach.py --describe --stdin
"""

import argparse
import cmath
import json
import math
import random
import socket
import subprocess
import sys
import threading
import time
from collections import deque

# ─── Config ───────────────────────────────────────────────────────────────────

UDP_HOST        = "0.0.0.0"
UDP_PORT        = 5500          # ESP32 sends here
CSI_HISTORY     = 100           # samples in rolling window (~5s at 20Hz)
PRESENCE_THRESH = 0.08          # CSI amplitude variance threshold for presence
MOTION_LOW      = 0.15          # variance delta threshold for low motion
MOTION_HIGH     = 0.40          # variance delta threshold for high motion
BREATH_MIN_HZ   = 0.2           # 12 BPM
BREATH_MAX_HZ   = 0.5           # 30 BPM
EMIT_INTERVAL   = 1.0           # seconds between signal emissions


# ─── CSI Parsing (espressif/esp-csi format) ───────────────────────────────────

def parse_csi_line(line: str) -> list[float] | None:
    """
    Parse one CSV line from ESP32 esp-csi firmware.
    Format: type,seq,mac,rssi,rate,...,len,first_word,[I0,Q0,I1,Q1,...]
    Returns list of amplitudes (one per subcarrier) or None on parse error.
    """
    try:
        parts = line.strip().split(",")
        if len(parts) < 25:
            return None
        # CSI data starts at index 24, format: [I0 Q0 I1 Q1 ...]
        raw = parts[24].strip().strip("[]").split()
        if len(raw) < 2:
            return None
        iq = [int(x) for x in raw]
        # Compute amplitude for each subcarrier (I,Q pairs)
        amps = []
        for i in range(0, len(iq) - 1, 2):
            amp = math.sqrt(iq[i] ** 2 + iq[i + 1] ** 2)
            amps.append(amp)
        return amps if amps else None
    except Exception:
        return None


def extract_rssi(line: str) -> int:
    """Extract RSSI field from esp-csi CSV line."""
    try:
        return int(line.split(",")[3])
    except Exception:
        return 0


# ─── Signal Processing ────────────────────────────────────────────────────────

class SignalProcessor:
    """
    Rolling window processor: CSI amplitudes → presence / motion / breathing.
    Works the same whether input is real CSI or RSSI-only.
    """

    def __init__(self, history: int = CSI_HISTORY, sample_rate: float = 20.0):
        self._history   = deque(maxlen=history)
        self._variances = deque(maxlen=history)
        self._sample_rate = sample_rate
        self._last_var  = 0.0

    def push(self, amplitudes: list[float]):
        if not amplitudes:
            return
        mean = sum(amplitudes) / len(amplitudes)
        var  = sum((a - mean) ** 2 for a in amplitudes) / len(amplitudes)
        # Normalize by mean to handle different hardware gain levels
        norm_var = var / (mean + 1e-6)
        self._history.append(norm_var)
        self._variances.append(norm_var)

    def presence(self) -> bool:
        if len(self._history) < 5:
            return False
        recent = list(self._history)[-20:]
        return (sum(recent) / len(recent)) > PRESENCE_THRESH

    def motion(self) -> str:
        if len(self._variances) < 10:
            return "none"
        recent = list(self._variances)[-10:]
        delta = max(recent) - min(recent)
        if delta >= MOTION_HIGH:
            return "high"
        if delta >= MOTION_LOW:
            return "low"
        return "none"

    def breathing_bpm(self) -> float:
        """FFT on variance time series to detect breathing frequency."""
        if len(self._variances) < 40:
            return 0.0
        data = list(self._variances)
        n = len(data)
        mean = sum(data) / n
        centered = [x - mean for x in data]

        # DFT over breathing frequency range only (avoid scipy dependency)
        best_mag  = 0.0
        best_freq = 0.0
        for k in range(1, n // 2):
            freq = k * self._sample_rate / n
            if not (BREATH_MIN_HZ <= freq <= BREATH_MAX_HZ):
                continue
            re = sum(centered[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
            im = sum(centered[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
            mag = math.sqrt(re ** 2 + im ** 2)
            if mag > best_mag:
                best_mag  = mag
                best_freq = freq

        if best_mag < 0.5:
            return 0.0
        return round(best_freq * 60, 1)


# ─── CSI Mode (ESP32-S3 via UDP) ─────────────────────────────────────────────

class CSISource:
    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self._host = host
        self._port = port
        self._proc = SignalProcessor(sample_rate=20.0)
        self._rssi = 0
        self._lock = threading.Lock()

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self._host, self._port))
        sock.settimeout(1.0)
        print(f"[wifi] CSI listening on UDP {self._host}:{self._port}", file=sys.stderr)
        while True:
            try:
                data, _ = sock.recvfrom(4096)
                line = data.decode("utf-8", errors="replace")
                amps = parse_csi_line(line)
                rssi = extract_rssi(line)
                if amps:
                    with self._lock:
                        self._proc.push(amps)
                        self._rssi = rssi
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[wifi] CSI recv error: {e}", file=sys.stderr)

    def signals(self) -> dict:
        with self._lock:
            return {
                "wifi_presence":      self._proc.presence(),
                "wifi_motion":        self._proc.motion(),
                "wifi_breathing_bpm": self._proc.breathing_bpm(),
                "wifi_rssi":          self._rssi,
                "wifi_source":        "csi",
            }


# ─── RSSI Mode (passive NIC monitor) ─────────────────────────────────────────

class RSSISource:
    """
    Passive RSSI monitoring. Puts NIC into monitor mode, reads beacon frames,
    tracks RSSI variance as a presence/motion proxy.
    No ESP32 needed — works with any WiFi NIC.
    """

    def __init__(self, iface: str = "wlan0"):
        self._iface = iface
        self._proc  = SignalProcessor(sample_rate=2.0)
        self._rssi  = 0
        self._lock  = threading.Lock()

    def _enable_monitor(self):
        try:
            subprocess.run(["sudo", "ip", "link", "set", self._iface, "down"],   check=True, capture_output=True)
            subprocess.run(["sudo", "iw", self._iface, "set", "monitor", "none"], check=True, capture_output=True)
            subprocess.run(["sudo", "ip", "link", "set", self._iface, "up"],      check=True, capture_output=True)
            print(f"[wifi] {self._iface} in monitor mode", file=sys.stderr)
            return True
        except Exception as e:
            print(f"[wifi] monitor mode failed: {e} — falling back to managed mode RSSI", file=sys.stderr)
            return False

    def start(self):
        monitor_ok = self._enable_monitor()
        iw_cmd = ["sudo", "iw", "dev", self._iface, "scan"]
        iwconfig_cmd = ["iwconfig", self._iface]

        while True:
            try:
                if monitor_ok:
                    # Read RSSI from scan results
                    out = subprocess.check_output(iw_cmd, stderr=subprocess.DEVNULL, timeout=3)
                    rssis = []
                    for line in out.decode().splitlines():
                        if "signal:" in line:
                            try:
                                rssis.append(float(line.split("signal:")[1].split()[0]))
                            except Exception:
                                pass
                    if rssis:
                        avg = sum(rssis) / len(rssis)
                        with self._lock:
                            self._rssi = int(avg)
                            # RSSI as a 1-element amplitude list — variance tracks signal change
                            self._proc.push([abs(avg)])
                else:
                    # Managed mode fallback: read link quality
                    out = subprocess.check_output(iwconfig_cmd, stderr=subprocess.DEVNULL, timeout=2)
                    for line in out.decode().splitlines():
                        if "Signal level" in line:
                            try:
                                rssi = int(line.split("Signal level=")[1].split()[0])
                                with self._lock:
                                    self._rssi = rssi
                                    self._proc.push([abs(rssi)])
                            except Exception:
                                pass
            except Exception as e:
                print(f"[wifi] RSSI read error: {e}", file=sys.stderr)

            time.sleep(0.5)

    def signals(self) -> dict:
        with self._lock:
            return {
                "wifi_presence":      self._proc.presence(),
                "wifi_motion":        self._proc.motion(),
                "wifi_breathing_bpm": self._proc.breathing_bpm(),
                "wifi_rssi":          self._rssi,
                "wifi_source":        "rssi",
            }


# ─── Simulation Mode (no hardware) ───────────────────────────────────────────

class SimSource:
    """
    Generates realistic synthetic CSI data for testing the full pipeline
    without any hardware. Simulates presence, motion events, and breathing.
    """

    def __init__(self):
        self._proc    = SignalProcessor(sample_rate=20.0)
        self._present = False
        self._phase   = 0.0
        self._t       = 0.0

    def _tick(self):
        """Generate one synthetic CSI sample."""
        self._t += 1.0 / 20.0

        # Simulate person entering after 5s, leaving after 60s
        if 5.0 < self._t < 60.0:
            self._present = True
        elif self._t > 60.0:
            self._present = False

        n_subcarriers = 52
        if self._present:
            # Base reflection + breathing (0.25 Hz = 15 BPM) + noise
            breath = math.sin(2 * math.pi * 0.25 * self._t)
            amps = [
                20.0 + 8.0 * breath + random.gauss(0, 1.5)
                for _ in range(n_subcarriers)
            ]
            # Occasional motion burst
            if random.random() < 0.05:
                amps = [a + random.gauss(0, 8) for a in amps]
        else:
            # Empty room: low stable signal
            amps = [5.0 + random.gauss(0, 0.3) for _ in range(n_subcarriers)]

        self._proc.push(amps)

    def run(self) -> "SimSource":
        """Background thread generates samples at 20 Hz."""
        def _loop():
            while True:
                self._tick()
                time.sleep(1.0 / 20.0)
        threading.Thread(target=_loop, daemon=True).start()
        return self

    def signals(self) -> dict:
        return {
            "wifi_presence":      self._proc.presence(),
            "wifi_motion":        self._proc.motion(),
            "wifi_breathing_bpm": self._proc.breathing_bpm(),
            "wifi_rssi":          -55 if self._present else -80,
            "wifi_source":        "sim",
        }


# ─── Main emit loop ───────────────────────────────────────────────────────────

def run(source, emit_interval: float = EMIT_INTERVAL):
    print("[wifi] starting — Ctrl+C to stop", file=sys.stderr)
    last_sig: dict = {}
    while True:
        sig = source.signals()
        sig["ts"] = time.time()

        # Only emit when something changed
        changed = (
            sig.get("wifi_presence")      != last_sig.get("wifi_presence") or
            sig.get("wifi_motion")        != last_sig.get("wifi_motion") or
            abs(sig.get("wifi_breathing_bpm", 0) - last_sig.get("wifi_breathing_bpm", 0)) > 1.0
        )
        if changed:
            print(json.dumps(sig, indent=2), flush=True)
            print("─" * 60, flush=True)
            last_sig = sig.copy()

        time.sleep(emit_interval)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiFi CSI perception source")
    parser.add_argument("--mode",  choices=["csi", "rssi", "sim"], default="sim",
                        help="csi=ESP32 UDP, rssi=NIC monitor mode, sim=simulation (default)")
    parser.add_argument("--host",  default=UDP_HOST, help=f"UDP host for CSI mode (default {UDP_HOST})")
    parser.add_argument("--port",  type=int, default=UDP_PORT, help=f"UDP port for CSI mode (default {UDP_PORT})")
    parser.add_argument("--iface", default="wlan0", help="NIC interface for RSSI mode (default wlan0)")
    parser.add_argument("--interval", type=float, default=EMIT_INTERVAL, help="Emit interval seconds")
    args = parser.parse_args()

    if args.mode == "csi":
        src = CSISource(args.host, args.port)
        threading.Thread(target=src.start, daemon=True).start()
        print(f"[wifi] CSI mode — waiting for ESP32 on UDP :{args.port}", file=sys.stderr)
        print("[wifi] Firmware: github.com/espressif/esp-csi", file=sys.stderr)
        time.sleep(1)
    elif args.mode == "rssi":
        src = RSSISource(args.iface)
        threading.Thread(target=src.start, daemon=True).start()
        print(f"[wifi] RSSI mode — monitoring {args.iface}", file=sys.stderr)
        time.sleep(1)
    else:
        src = SimSource().run()
        print("[wifi] Simulation mode — person enters at t=5s, breathing at 15 BPM", file=sys.stderr)
        time.sleep(1)

    try:
        run(src, args.interval)
    except KeyboardInterrupt:
        print("\n[wifi] stopped", file=sys.stderr)
