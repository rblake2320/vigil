"""
cameras/rtsp.py — RTSP / webcam / file input with reconnect
Any OpenCV-compatible source: webcam index, RTSP URL, file path, Elgato.
"""
import cv2
import time
import threading
import logging
from typing import Generator

log = logging.getLogger(__name__)


class CameraInput:
    """
    Thread-safe camera reader. Buffers the latest frame, reconnects on drop.

    Sources:
        0, 1, ...         — webcam index
        "rtsp://..."      — IP camera RTSP
        "elgato"          — /dev/video0 (Elgato 4K X on this machine)
        "/path/to/file"   — video file
    """

    ELGATO_DEVICE = "/dev/video0"

    def __init__(self, source, reconnect_delay: float = 3.0, fps_cap: int = 10):
        if source == "elgato":
            source = self.ELGATO_DEVICE
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.fps_cap = fps_cap
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_interval = 1.0 / fps_cap

    def start(self) -> "CameraInput":
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        # Wait for first frame
        for _ in range(50):
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.1)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def read(self):
        """Return latest frame or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def frames(self) -> Generator:
        """Yield frames at fps_cap rate."""
        while self._running:
            frame = self.read()
            if frame is not None:
                yield frame
            time.sleep(self._frame_interval)

    def _reader(self) -> None:
        while self._running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                log.warning(f"[Camera] Cannot open {self.source}, retrying in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)
                continue
            log.info(f"[Camera] Opened {self.source}")
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    log.warning(f"[Camera] Frame read failed — reconnecting")
                    break
                with self._lock:
                    self._frame = frame
                time.sleep(self._frame_interval)
            cap.release()
            if self._running:
                time.sleep(self.reconnect_delay)
