"""
vigil/core/realtime_monitor.py
================================
Two-stage real-time vision monitor.

Stage 1 — YOLO fast trigger (~1ms TRT / ~3ms PyTorch):
  Runs every frame. Fires alert immediately on detection.

Stage 2 — Cosmos async reasoner (3-5s, non-blocking):
  Enriches the alert with context AFTER it already fired.
  "Unknown person, adult male, entering from rear left"

Usage:
    monitor = RealtimeMonitor(source=0)          # webcam
    monitor = RealtimeMonitor(source="rtsp://...") # IP cam
    monitor = RealtimeMonitor(source="elgato")    # Elgato HDMI
    monitor.on_alert = my_alert_handler
    monitor.start()
"""

import cv2
import time
import base64
import json
import threading
import logging
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from ultralytics import YOLO

log = logging.getLogger(__name__)

COSMOS_URL  = "http://10.0.0.1:8000/v1/chat/completions"
ELGATO_DEV  = "/dev/video0"
STREAM_URL  = "http://localhost:8891/stream"  # CheatVision MJPEG stream

# COCO classes we care about for Vigil scenarios
VIGIL_CLASSES = {
    0:  "person",
    56: "chair",    # person sitting/fallen
    57: "couch",
    67: "cell phone",
    73: "laptop",
}

# Alert severity by class
SEVERITY = {
    "person": "HIGH",
    "default": "MEDIUM",
}

COSMOS_PROMPT = (
    "You are a real-time security and safety AI. A fast detector just triggered on this frame.\n"
    "Detections: {detections}\n\n"
    "In 2 sentences max:\n"
    "1. Describe exactly what you see (position, action, any threat indicators)\n"
    "2. Recommended action (monitor / alert owner / call 911)\n\n"
    "Start with THREAT: or SAFE: or MONITOR: based on your assessment."
)


class Detection:
    __slots__ = ("label", "confidence", "box", "ts")

    def __init__(self, label: str, confidence: float, box: tuple, ts: float):
        self.label      = label
        self.confidence = confidence
        self.box        = box       # (x1, y1, x2, y2) normalized 0-1
        self.ts         = ts


class Alert:
    __slots__ = ("detections", "frame", "ts", "severity", "cosmos_reasoning")

    def __init__(self, detections: list[Detection], frame, ts: float):
        self.detections       = detections
        self.frame            = frame
        self.ts               = ts
        self.severity         = max((SEVERITY.get(d.label, "MEDIUM") for d in detections),
                                    key=lambda s: ["LOW","MEDIUM","HIGH"].index(s))
        self.cosmos_reasoning: Optional[str] = None  # filled in async


def _grab_mjpeg_frame() -> Optional[bytes]:
    """Pull one JPEG from the CheatVision MJPEG stream."""
    try:
        with urllib.request.urlopen(STREAM_URL, timeout=3) as r:
            buf = b""
            while len(buf) < 400_000:
                buf += r.read(4096)
                s = buf.find(b"\xff\xd8\xff")
                if s >= 0:
                    e = buf.find(b"\xff\xd9", s + 2)
                    if e >= 0:
                        return buf[s:e + 2]
    except Exception:
        pass
    return None


class FastDetector:
    """
    YOLO detector. Auto-selects TensorRT engine if available, falls back to PyTorch.
    Runs in ~1ms (TRT) or ~3ms (PyTorch) on GB10.
    """

    TRT_PATH = Path("/tmp/yolo11n.engine")
    PT_PATH  = Path("/tmp/yolo11n.pt")

    def __init__(self, confidence: float = 0.45, classes: list[int] | None = None):
        self.confidence = confidence
        self.classes    = classes or list(VIGIL_CLASSES.keys())
        self._model     = None
        self._warmup_done = False

    def load(self) -> str:
        if self.TRT_PATH.exists():
            self._model = YOLO(str(self.TRT_PATH))
            mode = "TensorRT FP16"
        elif self.PT_PATH.exists():
            self._model = YOLO(str(self.PT_PATH))
            mode = "PyTorch"
        else:
            self._model = YOLO("yolo11n.pt")
            mode = "PyTorch (downloaded)"
        log.info(f"[FastDetector] Loaded YOLO11n — {mode}")
        return mode

    def warmup(self, n: int = 5) -> float:
        """Warmup GPU, return average inference ms."""
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self._model(dummy, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times[2:]) / len(times[2:])
        log.info(f"[FastDetector] Warmup done — avg {avg:.1f}ms/frame")
        self._warmup_done = True
        return avg

    def detect(self, frame) -> tuple[list[Detection], float]:
        """Run YOLO on frame. Returns (detections, ms_elapsed)."""
        t0   = time.perf_counter()
        results = self._model(
            frame,
            conf=self.confidence,
            classes=self.classes,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections = []
        h, w = frame.shape[:2]
        for r in results:
            for box in r.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                label = VIGIL_CLASSES.get(cls, r.names.get(cls, str(cls)))
                detections.append(Detection(
                    label      = label,
                    confidence = conf,
                    box        = (x1/w, y1/h, x2/w, y2/h),
                    ts         = time.time(),
                ))
        return detections, elapsed_ms


class CosmosReasoner:
    """Async VLM reasoning. Never blocks the detection loop."""

    def __init__(self):
        self._queue: list = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, alert: Alert, callback: Callable[[Alert, str], None]) -> None:
        with self._lock:
            self._queue.append((alert, callback))

    def _worker(self):
        while True:
            job = None
            with self._lock:
                if self._queue:
                    job = self._queue.pop(0)
            if job is None:
                time.sleep(0.05)
                continue
            alert, callback = job
            try:
                reasoning = self._call_cosmos(alert)
                alert.cosmos_reasoning = reasoning
                callback(alert, reasoning)
            except Exception as e:
                log.warning(f"[Cosmos] Reasoning failed: {e}")

    def _call_cosmos(self, alert: Alert) -> str:
        _, jpg = cv2.imencode(".jpg", alert.frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(jpg.tobytes()).decode()

        det_str = ", ".join(
            f"{d.label} ({d.confidence:.0%})" for d in alert.detections
        )
        prompt = COSMOS_PROMPT.format(detections=det_str)

        payload = {
            "model":    "nvidia/cosmos-reason2-8b",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",      "text":      prompt},
            ]}],
            "max_tokens":  120,
            "temperature": 0.1,
        }
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            COSMOS_URL, data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]


class RealtimeMonitor:
    """
    Orchestrates the two-stage real-time pipeline.

    Callbacks:
      on_alert(alert: Alert)              — fires immediately on YOLO detection
      on_reasoning(alert: Alert, text)    — fires when Cosmos finishes (async)
      on_frame(frame, detections, ms)     — fires every processed frame
    """

    def __init__(
        self,
        source=STREAM_URL,
        confidence: float = 0.45,
        cooldown_s: float = 3.0,
        fps_cap: int       = 30,
    ):
        self.source     = source
        self.cooldown_s = cooldown_s
        self.fps_cap    = fps_cap

        self.on_alert:     Optional[Callable] = None
        self.on_reasoning: Optional[Callable] = None
        self.on_frame:     Optional[Callable] = None

        self._detector = FastDetector(confidence=confidence)
        self._reasoner = CosmosReasoner()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._last_alert_ts: float = 0.0

        # Rolling stats
        self.stats = {
            "frames":     0,
            "alerts":     0,
            "avg_ms":     0.0,
            "start_ts":   0.0,
        }

    def start(self) -> None:
        mode = self._detector.load()
        log.info(f"[Monitor] Starting — source={self.source} model={mode}")
        self._detector.warmup()
        self._running  = True
        self.stats["start_ts"] = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        frame_interval = 1.0 / self.fps_cap
        cap = None

        # Decide input
        use_mjpeg = isinstance(self.source, str) and self.source.startswith("http")

        if not use_mjpeg:
            src = ELGATO_DEV if self.source == "elgato" else self.source
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self._running:
            t_start = time.perf_counter()

            # --- Grab frame ---
            frame = None
            if use_mjpeg:
                jpg = _grab_mjpeg_frame()
                if jpg:
                    arr = __import__("numpy").frombuffer(jpg, dtype=__import__("numpy").uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            elif cap:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

            if frame is None:
                time.sleep(0.1)
                continue

            # --- Stage 1: YOLO fast detection ---
            detections, ms = self._detector.detect(frame)

            # Update stats
            self.stats["frames"] += 1
            n = self.stats["frames"]
            self.stats["avg_ms"] = (self.stats["avg_ms"] * (n-1) + ms) / n

            # Fire frame callback
            if self.on_frame:
                try:
                    self.on_frame(frame, detections, ms)
                except Exception:
                    pass

            # --- Check for alert-worthy detections ---
            if detections:
                now = time.time()
                if now - self._last_alert_ts >= self.cooldown_s:
                    self._last_alert_ts = now
                    self.stats["alerts"] += 1
                    alert = Alert(detections, frame.copy(), now)

                    # Fire immediate alert
                    if self.on_alert:
                        try:
                            self.on_alert(alert)
                        except Exception as e:
                            log.warning(f"[Monitor] on_alert error: {e}")

                    # Queue async Cosmos reasoning
                    self._reasoner.submit(alert, self._on_reasoning_done)

            # --- Cap FPS ---
            elapsed = time.perf_counter() - t_start
            sleep = frame_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        if cap:
            cap.release()

    def _on_reasoning_done(self, alert: Alert, reasoning: str) -> None:
        if self.on_reasoning:
            try:
                self.on_reasoning(alert, reasoning)
            except Exception as e:
                log.warning(f"[Monitor] on_reasoning error: {e}")
