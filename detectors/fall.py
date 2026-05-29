"""
detectors/fall.py — Person fall detection via pose keypoints
Uses YOLO11 pose model. Triggers when a detected person's aspect ratio
shifts from vertical (standing) to horizontal (fallen) and stays there.
"""
import time
import numpy as np
from collections import deque


class FallDetector:
    """
    Tracks person bounding-box aspect ratio over time.
    Standing person: height > width (ratio > 1)
    Fallen person:   width >= height (ratio <= 1) for > min_duration seconds
    """

    def __init__(self, min_duration: float = 3.0, history_len: int = 30):
        self.min_duration = min_duration
        self._history: deque = deque(maxlen=history_len)
        self._fallen_since: float | None = None

    def update(self, person_boxes: list[tuple]) -> tuple[bool, float]:
        """
        person_boxes: list of (x1, y1, x2, y2) for each detected person.
        Returns (fall_detected, confidence 0-1).
        """
        if not person_boxes:
            self._fallen_since = None
            return False, 0.0

        fallen_count = 0
        for x1, y1, x2, y2 in person_boxes:
            w = x2 - x1
            h = y2 - y1
            if h > 0 and w / h > 1.2:  # wider than tall
                fallen_count += 1

        if fallen_count > 0:
            if self._fallen_since is None:
                self._fallen_since = time.time()
            elapsed = time.time() - self._fallen_since
            if elapsed >= self.min_duration:
                confidence = min(1.0, elapsed / (self.min_duration * 2))
                return True, confidence
        else:
            self._fallen_since = None

        return False, 0.0


class MotionlessPersonDetector:
    """
    Detects a person who has been stationary for too long.
    Used for: unconscious person, medical event, pool emergency.
    """

    def __init__(self, max_stationary: float = 10.0, movement_threshold: float = 15.0):
        self.max_stationary = max_stationary
        self.movement_threshold = movement_threshold
        self._last_positions: dict[int, tuple] = {}
        self._stationary_since: dict[int, float] = {}

    def update(self, tracked_persons: list[tuple[int, float, float]]) -> tuple[bool, float]:
        """
        tracked_persons: list of (track_id, center_x, center_y)
        Returns (motionless_detected, longest_stationary_seconds)
        """
        now = time.time()
        max_elapsed = 0.0

        for track_id, cx, cy in tracked_persons:
            prev = self._last_positions.get(track_id)
            if prev is not None:
                dist = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
                if dist < self.movement_threshold:
                    if track_id not in self._stationary_since:
                        self._stationary_since[track_id] = now
                    elapsed = now - self._stationary_since[track_id]
                    max_elapsed = max(max_elapsed, elapsed)
                else:
                    self._stationary_since.pop(track_id, None)
            self._last_positions[track_id] = (cx, cy)

        if max_elapsed >= self.max_stationary:
            return True, max_elapsed
        return False, max_elapsed
