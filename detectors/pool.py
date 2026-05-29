"""
detectors/pool.py — Pool / water body drowning detection
Combines zone definition (pool boundary), person detection,
and motionless-in-water analysis.

Setup: define the pool zone as a polygon on first run (or config file).
"""
import cv2
import numpy as np
import time


class PoolZone:
    """Polygon representing the water surface area."""

    def __init__(self, polygon_pts: list[tuple[int, int]] | None = None):
        # Default: full frame — override with actual pool boundary
        self._pts = polygon_pts
        self._mask: np.ndarray | None = None

    def set_frame_size(self, h: int, w: int) -> None:
        if self._pts is None:
            self._mask = np.ones((h, w), dtype=np.uint8) * 255
        else:
            self._mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(self._pts, dtype=np.int32)
            cv2.fillPoly(self._mask, [pts], 255)

    def person_in_zone(self, box: tuple[int, int, int, int]) -> bool:
        if self._mask is None:
            return True
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= cy < self._mask.shape[0] and 0 <= cx < self._mask.shape[1]:
            return self._mask[cy, cx] > 0
        return False


class PoolEmergencyDetector:
    """
    Triggers when a person is:
    1. Inside the defined pool zone
    2. Horizontal (fallen / floating face-down)
    3. Motionless for > threshold seconds

    Any one of these alone = warning. All three = EMERGENCY.
    """

    def __init__(
        self,
        zone: PoolZone | None = None,
        motionless_threshold: float = 6.0,
    ):
        self.zone = zone or PoolZone()
        self.motionless_threshold = motionless_threshold
        self._in_zone_since: dict[int, float] = {}
        self._last_positions: dict[int, tuple] = {}

    def update(
        self,
        tracked_persons: list[tuple[int, tuple[int, int, int, int]]],  # (id, box)
    ) -> tuple[str, float]:
        """
        Returns (severity, elapsed_seconds).
        severity: "clear", "warning", "emergency"
        """
        now = time.time()
        max_elapsed = 0.0
        severity = "clear"

        for track_id, box in tracked_persons:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1
            horizontal = h > 0 and w / h > 1.3

            in_zone = self.zone.person_in_zone(box)
            prev = self._last_positions.get(track_id)
            motionless = False
            if prev is not None:
                dist = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
                motionless = dist < 10.0

            self._last_positions[track_id] = (cx, cy)

            if in_zone:
                if track_id not in self._in_zone_since:
                    self._in_zone_since[track_id] = now
                elapsed = now - self._in_zone_since[track_id]
                max_elapsed = max(max_elapsed, elapsed)

                if horizontal and motionless and elapsed >= self.motionless_threshold:
                    severity = "emergency"
                elif horizontal or (motionless and elapsed > 3.0):
                    if severity != "emergency":
                        severity = "warning"
            else:
                self._in_zone_since.pop(track_id, None)

        return severity, max_elapsed
