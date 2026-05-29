"""
detectors/fire.py — Fire and smoke visual detection
Pixel-level heuristic: bright orange/red/yellow saturation spike.
VLM confirmation layer rejects false positives (lamps, sunsets, etc).
"""
import cv2
import numpy as np


class FireDetector:
    """
    Stage 1: fast pixel heuristic (runs every frame, <1ms)
    Stage 2: VLM confirmation (runs only when heuristic triggers)
    """

    def __init__(self, pixel_threshold: float = 0.08, baseline_mult: float = 2.5):
        self.pixel_threshold = pixel_threshold  # fraction of frame that must be "fire-colored"
        self.baseline_mult = baseline_mult
        self._baseline: float | None = None
        self._frame_count = 0

    def update_baseline(self, frame: np.ndarray) -> None:
        pct = self._fire_pixel_pct(frame)
        if self._baseline is None:
            self._baseline = pct
        else:
            self._baseline = 0.95 * self._baseline + 0.05 * pct  # EMA

    def check(self, frame: np.ndarray) -> tuple[bool, float]:
        """Returns (triggered, fire_pixel_pct)."""
        pct = self._fire_pixel_pct(frame)
        self._frame_count += 1

        # Warm up baseline for first 30 frames
        if self._frame_count < 30:
            self.update_baseline(frame)
            return False, pct

        threshold = max(
            self.pixel_threshold,
            (self._baseline or 0) * self.baseline_mult,
        )
        return pct >= threshold, pct

    @staticmethod
    def _fire_pixel_pct(frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Orange-red hue range (0-25 and 160-180), high saturation, high value
        fire = (
            ((h <= 25) | (h >= 160)) &
            (s > 80) &
            (v > 150)
        )
        return float(fire.sum()) / fire.size


class SmokeDetector:
    """
    Smoke: gray pixels increasing, contrast decreasing over time.
    Less precise than fire — use VLM confirmation.
    """

    def __init__(self, gray_threshold: float = 0.12):
        self.gray_threshold = gray_threshold
        self._prev_gray_pct: float = 0.0

    def check(self, frame: np.ndarray) -> tuple[bool, float]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        # Low saturation, mid-to-high value = gray/white
        gray = (s < 40) & (v > 80) & (v < 220)
        pct = float(gray.sum()) / gray.size
        delta = pct - self._prev_gray_pct
        self._prev_gray_pct = pct
        return (pct > self.gray_threshold and delta > 0.005), pct
