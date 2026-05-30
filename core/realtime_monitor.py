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
import os

from ultralytics import YOLO

log = logging.getLogger(__name__)

COSMOS_URL  = "http://10.0.0.1:8000/v1/chat/completions"   # Cosmos fallback
STEP_URL    = "http://localhost:8898/v1/chat/completions"  # Step-3.7-Flash (primary, local)
ELGATO_DEV  = 0  # /dev/video0 — pass as int, not string (OpenCV V4L2 backend)
STREAM_URL  = "http://localhost:8891/stream"  # CheatVision MJPEG stream

# COCO classes — full coverage: people, devices, vehicles, weapons, bags, animals
VIGIL_CLASSES = {
    # People
    0:  "person",
    # Vehicles
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    # Weapons / dangerous objects
    34: "baseball bat",
    43: "knife",
    76: "scissors",
    # Bags / carry items
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    # Electronics / devices
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    72: "refrigerator",
    # Furniture / environment
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    61: "toilet",
    # Animals
    15: "cat",
    16: "dog",
    17: "horse",
    # Other security-relevant
    39: "bottle",
    73: "book",
    74: "clock",
    75: "vase",
    77: "teddy bear",
}

# Alert severity by class
SEVERITY = {
    "person":       "HIGH",
    "knife":        "CRITICAL",
    "scissors":     "HIGH",
    "baseball bat": "HIGH",
    "car":          "MEDIUM",
    "truck":        "MEDIUM",
    "motorcycle":   "MEDIUM",
    "bus":          "MEDIUM",
    "bicycle":      "LOW",
    "backpack":     "MEDIUM",
    "suitcase":     "MEDIUM",
    "handbag":      "LOW",
    "laptop":       "MEDIUM",
    "cell phone":   "LOW",
    "tv":           "LOW",
    "dog":          "MEDIUM",
    "cat":          "LOW",
    "default":      "LOW",
}

COSMOS_PROMPT = (
    "Describe this image in one sentence for a blind person. "
    "Detected objects: {detections}. "
    "Name exactly what you see — people (clothing, position, action), objects, setting. "
    "Be specific and concrete. No threat assessment. Max 20 words."
)

STEP_PROMPT = (
    "Describe this scene in one sentence for a blind person. "
    "Name specific objects, people (appearance, clothing, action), and what is happening. "
    "Be concrete and specific — no vague words like 'various' or 'several'. Max 20 words."
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
            self._model(dummy, verbose=False, device="cpu")
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
            device="cpu",
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
                reasoning = self._call_vlm(alert)
                alert.cosmos_reasoning = reasoning
                callback(alert, reasoning)
            except Exception as e:
                log.warning(f"[Cosmos] Reasoning failed: {e}")

    def _call_vlm(self, alert: Alert) -> str:
        _, jpg = cv2.imencode(".jpg", alert.frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(jpg.tobytes()).decode()

        # Try Step-3.7-Flash first (richer descriptions, faster)
        try:
            payload = {
                "model":    "step-3.7-flash",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",      "text":      STEP_PROMPT},
                ]}],
                "max_tokens":  120,
                "temperature": 0.2,
            }
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                STEP_URL, data=data,
                headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                resp = json.loads(r.read())
                msg  = resp["choices"][0]["message"]
                # Step returns answer in content; reasoning is in reasoning_content
                text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                log.info(f"[Step] {text[:120]}")
                return text
        except Exception as e:
            log.warning(f"[Step] Failed ({e}), falling back to Cosmos")

        # Fallback: Cosmos
        det_str = ", ".join(f"{d.label} ({d.confidence:.0%})" for d in alert.detections)
        prompt  = COSMOS_PROMPT.format(detections=det_str)
        payload = {
            "model":    "nvidia/cosmos-reason2-8b",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",      "text":      prompt},
            ]}],
            "max_tokens":  40,
            "temperature": 0.1,
        }
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            COSMOS_URL, data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]


class DataCollector:
    """
    Auto-labeling pipeline. On every alert:
      - saves frame as JPEG
      - saves YOLO-format .txt annotation (class_id cx cy w h, normalized)
      - saves .meta.json with Cosmos reasoning + confidence
    Directory layout:
      vigil_training/
        images/  *.jpg
        labels/  *.txt          (YOLO format)
        meta/    *.meta.json    (Cosmos text + detections)
        data.yaml               (ready for yolo train)
    """

    BASE = Path(__file__).parent.parent / "vigil_training"

    # Map label name → sequential class id for the training dataset
    _class_map: dict[str, int] = {}
    _lock = threading.Lock()

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            (self.BASE / "images").mkdir(parents=True, exist_ok=True)
            (self.BASE / "labels").mkdir(parents=True, exist_ok=True)
            (self.BASE / "meta").mkdir(parents=True, exist_ok=True)
            self._load_class_map()
            log.info(f"[DataCollector] Saving training data → {self.BASE}")

    def _load_class_map(self):
        yaml_path = self.BASE / "data.yaml"
        if yaml_path.exists():
            import yaml
            with open(yaml_path) as f:
                d = yaml.safe_load(f)
            names = d.get("names", [])
            DataCollector._class_map = {n: i for i, n in enumerate(names)}

    def _save_yaml(self):
        names = sorted(DataCollector._class_map, key=lambda k: DataCollector._class_map[k])
        yaml_path = self.BASE / "data.yaml"
        lines = [
            "path: " + str(self.BASE),
            "train: images",
            "val: images",
            f"nc: {len(names)}",
            "names:",
        ] + [f"  - {n}" for n in names]
        yaml_path.write_text("\n".join(lines) + "\n")

    def collect(self, alert: Alert, reasoning: str = "") -> None:
        if not self.enabled:
            return
        try:
            ts = int(alert.ts * 1000)
            stem = f"vigil_{ts}"

            # Save image
            img_path = self.BASE / "images" / f"{stem}.jpg"
            cv2.imwrite(str(img_path), alert.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Build YOLO label lines
            h, w = alert.frame.shape[:2]
            lines = []
            with self._lock:
                for d in alert.detections:
                    if d.label not in DataCollector._class_map:
                        DataCollector._class_map[d.label] = len(DataCollector._class_map)
                        self._save_yaml()
                    cid = DataCollector._class_map[d.label]
                    x1, y1, x2, y2 = (d.box[0]*w, d.box[1]*h, d.box[2]*w, d.box[3]*h)
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (self.BASE / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")

            # Save metadata
            meta = {
                "ts": alert.ts,
                "reasoning": reasoning,
                "detections": [{"label": d.label, "conf": round(d.confidence, 3)} for d in alert.detections],
            }
            (self.BASE / "meta" / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))

        except Exception as e:
            log.warning(f"[DataCollector] Save failed: {e}")


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

        self._detector   = FastDetector(confidence=confidence)
        self._reasoner   = CosmosReasoner()
        self._collector  = DataCollector(enabled=True)
        self._running    = False
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
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
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

            # Apply ROI crop if set (from UI focus/hover mode)
            roi = getattr(self, 'focus_roi', None)
            detect_frame = frame
            roi_offset = (0, 0)
            if roi:
                h, w = frame.shape[:2]
                x1 = int(roi[0]*w); y1 = int(roi[1]*h)
                x2 = int(roi[2]*w); y2 = int(roi[3]*h)
                detect_frame = frame[y1:y2, x1:x2]
                roi_offset = (x1/w, y1/h)

            # --- Stage 1: YOLO fast detection ---
            detections, ms = self._detector.detect(detect_frame)

            # Remap detection boxes back to full-frame coordinates if ROI was used
            if roi and roi_offset != (0,0):
                rw = roi[2]-roi[0]; rh = roi[3]-roi[1]
                for d in detections:
                    bx1,by1,bx2,by2 = d.box
                    d.box = (roi[0]+bx1*rw, roi[1]+by1*rh, roi[0]+bx2*rw, roi[1]+by2*rh)

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

                    # Save frame + labels immediately (no reasoning yet)
                    self._collector.collect(alert, reasoning="")

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
        # Update meta file with Cosmos reasoning
        self._collector.collect(alert, reasoning=reasoning)
        if self.on_reasoning:
            try:
                self.on_reasoning(alert, reasoning)
            except Exception as e:
                log.warning(f"[Monitor] on_reasoning error: {e}")
