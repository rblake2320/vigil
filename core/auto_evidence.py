"""
auto_evidence.py — Zero-dependency wrapper around EvidenceRecorder for use in viewer.py.

Provides:
  - RingBuffer: fixed-size deque of raw frames for pre-trigger capture
  - AutoEvidence: push frames + verdicts, auto-saves MP4 clips to ledger on triggers

No imports from analyzer.py or capture.py. Safe to use directly in viewer.py.
"""

import cv2
import json
import time
import uuid
import hashlib
import threading
from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Verdict constants (mirrors analyzer.Verdict without importing it)
# ---------------------------------------------------------------------------
VERDICT_CONFIRMED = "CONFIRMED_CHEAT"
VERDICT_LIKELY    = "LIKELY_CHEAT"

# Cooldown multiplier for LIKELY_CHEAT relative to confirmed
_LIKELY_COOLDOWN_MULT = 2.0


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------

class RingBuffer:
    """Fixed-length ring buffer of raw numpy frames."""

    def __init__(self, maxlen: int = 300):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        """Add a frame, auto-evicting oldest if full."""
        with self._lock:
            self._buf.append(frame)

    def snapshot(self, pre: int = 150, post: int = 0) -> List[np.ndarray]:
        """Return the last *pre* frames currently in the buffer.

        *post* is accepted for API symmetry but ignored here — the caller
        must push additional frames after the trigger and collect them
        separately (AutoEvidence handles this).
        """
        with self._lock:
            frames = list(self._buf)
        return frames[-pre:] if len(frames) >= pre else frames[:]


# ---------------------------------------------------------------------------
# Simple JSONL ledger helpers (no external deps)
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _chain_hash(entry_without_chain: dict) -> str:
    """SHA-256 of the JSON-serialised entry (no chain_hash key present)."""
    payload = json.dumps(entry_without_chain, sort_keys=True).encode()
    return _sha256_bytes(payload)


def _ledger_head(ledger_path: Path) -> str:
    """Return chain_hash of the last ledger entry, or 64 zeros if empty."""
    if not ledger_path.exists():
        return "0" * 64
    with open(ledger_path, "rb") as f:
        lines = f.read().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line:
            try:
                return json.loads(line).get("chain_hash", "0" * 64)
            except json.JSONDecodeError:
                continue
    return "0" * 64


def _append_ledger(ledger_path: Path, entry: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def _write_mp4(path: Path, frames: List[np.ndarray], fps: float = 30.0) -> bool:
    """Write a list of BGR numpy frames to an MP4 file. Returns True on success."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True


def _stamp_hash_overlay(frame: np.ndarray, sha256: str) -> np.ndarray:
    """Return a copy of *frame* with the SHA-256 hash stamped in the top-left."""
    out = frame.copy()
    short = sha256[:24] + "…"
    cv2.putText(
        out,
        f"SHA256:{short}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# AutoEvidence
# ---------------------------------------------------------------------------

class AutoEvidence:
    """
    Frame-by-frame evidence accumulator for viewer.py.

    Usage inside capture loop
    -------------------------
    ae = AutoEvidence(output_dir="evidence", ledger_file="evidence/ledger.jsonl")

    while True:
        raw   = grab_raw_frame()
        ann   = annotate(raw)
        stats = compute_stats()

        ae.push_frame(raw, ann)
        result = ae.check_verdict(stats["verdict"], stats["score"])
        if result:
            print("Evidence saved:", result["path"])
    """

    def __init__(
        self,
        output_dir: str,
        ledger_file: str,
        pre_frames: int = 150,
        post_frames: int = 90,
        cooldown_s: float = 15.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(ledger_file)

        self.pre_frames  = pre_frames
        self.post_frames = post_frames
        self.cooldown_s  = cooldown_s

        # Ring buffers — raw for pre-trigger, annotated for final clip
        self._raw_buf = RingBuffer(maxlen=max(pre_frames, 300))
        self._ann_buf = RingBuffer(maxlen=max(pre_frames, 300))

        # Post-trigger collection state
        self._triggered       = False
        self._trigger_verdict = ""
        self._trigger_score   = 0.0
        self._post_raw: List[np.ndarray]  = []
        self._post_ann: List[np.ndarray]  = []
        self._pre_snapshot_raw: List[np.ndarray] = []
        self._pre_snapshot_ann: List[np.ndarray] = []

        # Cooldown tracking per verdict level
        self._last_confirmed_ts = 0.0
        self._last_likely_ts    = 0.0

        # In-memory index of recent entries
        self._recent: deque = deque(maxlen=200)

        # Chain state (loaded from ledger so we survive restarts)
        self._prev_hash = _ledger_head(self.ledger_path)
        self._lock = threading.Lock()

        # Reload recent entries from ledger if it exists
        self._load_recent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_frame(self, raw_frame: np.ndarray, annotated_frame: np.ndarray) -> None:
        """Call every iteration of the capture loop before check_verdict."""
        self._raw_buf.push(raw_frame)
        self._ann_buf.push(annotated_frame)

        if self._triggered:
            self._post_raw.append(raw_frame)
            self._post_ann.append(annotated_frame)

    def check_verdict(self, verdict: str, score: float) -> Optional[dict]:
        """
        Evaluate the current verdict.

        Returns a dict (path, sha256, chain_hash, …) once a clip is complete,
        or None while still collecting or when no trigger has fired.
        """
        with self._lock:
            now = time.time()

            # --- Check if post-trigger collection is done ---
            if self._triggered:
                if len(self._post_raw) >= self.post_frames:
                    return self._finalise_clip()
                return None

            # --- Decide whether to trigger ---
            if verdict == VERDICT_CONFIRMED:
                elapsed = now - self._last_confirmed_ts
                if elapsed < self.cooldown_s:
                    return None
                self._arm_trigger(verdict, score, now)
                return None

            if verdict == VERDICT_LIKELY:
                elapsed = now - self._last_likely_ts
                likely_cooldown = self.cooldown_s * _LIKELY_COOLDOWN_MULT
                if elapsed < likely_cooldown:
                    return None
                self._arm_trigger(verdict, score, now)
                return None

            return None

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last *n* evidence entries (most-recent last)."""
        entries = list(self._recent)
        return entries[-n:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _arm_trigger(self, verdict: str, score: float, now: float) -> None:
        """Snapshot pre-frames and begin post-frame collection."""
        self._triggered       = True
        self._trigger_verdict = verdict
        self._trigger_score   = score
        self._post_raw        = []
        self._post_ann        = []

        # Snapshot current ring buffer contents for pre-clip
        self._pre_snapshot_raw = self._raw_buf.snapshot(pre=self.pre_frames)
        self._pre_snapshot_ann = self._ann_buf.snapshot(pre=self.pre_frames)

        # Update cooldown timestamps
        if verdict == VERDICT_CONFIRMED:
            self._last_confirmed_ts = now
        else:
            self._last_likely_ts = now

    def _finalise_clip(self) -> Optional[dict]:
        """Write clip + ledger entry. Resets trigger state. Returns entry dict."""
        verdict = self._trigger_verdict
        score   = self._trigger_score

        # Build frame list: pre (annotated) + post (annotated)
        clip_frames = self._pre_snapshot_ann + self._post_ann[: self.post_frames]

        # Timestamp for filename
        ts_str  = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        clip_name = f"{ts_str}_{verdict}.mp4"
        clip_path = self.output_dir / clip_name

        # Write video
        success = _write_mp4(clip_path, clip_frames)
        if not success or not clip_path.exists():
            self._reset_trigger()
            return None

        # SHA-256 of the raw video bytes
        sha256 = _sha256_file(clip_path)

        # Stamp hash on first frame and rewrite clip with stamped first frame
        self._stamp_first_frame(clip_path, clip_frames, sha256)

        # Re-hash after stamp (the file changed)
        sha256 = _sha256_file(clip_path)

        # Build ledger entry (without chain_hash first)
        event_id = str(uuid.uuid4())
        ts_iso   = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        entry_body: dict = {
            "id":        event_id,
            "ts":        ts_iso,
            "verdict":   verdict,
            "score":     round(score, 4),
            "sha256":    sha256,
            "prev_hash": self._prev_hash,
            "path":      str(clip_path),
        }

        ch = _chain_hash(entry_body)
        entry_body["chain_hash"] = ch

        # Persist
        _append_ledger(self.ledger_path, entry_body)
        self._prev_hash = ch
        self._recent.append(entry_body)

        self._reset_trigger()
        return entry_body

    def _stamp_first_frame(
        self,
        clip_path: Path,
        original_frames: List[np.ndarray],
        sha256: str,
    ) -> None:
        """Rewrite the clip with the SHA-256 hash stamped on the first frame."""
        if not original_frames:
            return
        stamped = [_stamp_hash_overlay(original_frames[0], sha256)] + list(original_frames[1:])
        tmp_path = clip_path.with_suffix(".tmp.mp4")
        ok = _write_mp4(tmp_path, stamped)
        if ok and tmp_path.exists():
            tmp_path.replace(clip_path)

    def _reset_trigger(self) -> None:
        self._triggered           = False
        self._trigger_verdict     = ""
        self._trigger_score       = 0.0
        self._post_raw            = []
        self._post_ann            = []
        self._pre_snapshot_raw    = []
        self._pre_snapshot_ann    = []

    def _load_recent(self) -> None:
        """Populate in-memory recent list from existing ledger."""
        if not self.ledger_path.exists():
            return
        try:
            with open(self.ledger_path) as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        self._recent.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
