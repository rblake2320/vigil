"""
discord_alert.py — CheatVision Discord webhook alerter.

Sends rich embeds (with optional frame attachments) to a Discord webhook
when cheating is detected.  All network I/O runs in a background thread so
the capture loop is never blocked.

Dependencies: os, json, time, threading, urllib.request, urllib.parse,
              io, cv2, numpy  (no third-party HTTP library).
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Dict, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_frame_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR numpy array to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _build_multipart(fields: Dict[str, str], file_bytes: bytes,
                     filename: str, content_type: str, boundary: str) -> bytes:
    """
    Build a multipart/form-data body manually.

    fields      : plain text fields  {name: value}
    file_bytes  : raw bytes of the file part
    filename    : filename for the file part
    content_type: MIME type of the file part
    boundary    : multipart boundary string
    """
    parts: list[bytes] = []

    # Text fields
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )

    # File part
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode()
        + file_bytes
        + b"\r\n"
    )

    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)


def _post_json(url: str, payload: dict) -> None:
    """POST JSON payload to *url*; silently swallow errors."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass  # Never crash the capture loop


def _post_multipart(url: str, fields: Dict[str, str],
                    file_bytes: bytes, filename: str,
                    file_content_type: str = "image/jpeg") -> None:
    """POST multipart/form-data (embed JSON + file) to *url*."""
    boundary = "CheatVisionBoundary" + str(int(time.time()))
    body = _build_multipart(fields, file_bytes, filename,
                             file_content_type, boundary)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Colour / title mappings
# ---------------------------------------------------------------------------

_EMBED_COLOR: Dict[str, int] = {
    "SUSPICIOUS":     0xFFCC00,   # yellow
    "LIKELY_CHEAT":   0xFF8C00,   # orange
    "CONFIRMED_CHEAT": 0xFF0000,  # red
}

_EMBED_TITLE: Dict[str, str] = {
    "SUSPICIOUS":     "⚠️ CheatVision Alert",
    "LIKELY_CHEAT":   "🚨 CheatVision Alert",
    "CONFIRMED_CHEAT": "🚨 CHEAT CONFIRMED",
}

_DEFAULT_COLOR = 0x808080  # grey for unknown verdicts
_DEFAULT_TITLE = "ℹ️ CheatVision Alert"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DiscordAlerter:
    """
    Sends Discord webhook alerts when cheating is detected.

    If no webhook URL is configured, every public method silently no-ops so
    the rest of CheatVision never needs to guard calls.
    """

    def __init__(self, webhook_url: str = None, game: str = "Unknown"):
        """
        Parameters
        ----------
        webhook_url : Discord webhook URL.  Falls back to the environment
                      variable ``CHEATVISION_DISCORD_WEBHOOK`` if *None*.
        game        : Game / session name shown in embed fields.
        """
        self.webhook_url: str = (
            webhook_url
            or os.environ.get("CHEATVISION_DISCORD_WEBHOOK", "")
        )
        self.game: str = game

        # {verdict: last_sent_timestamp}
        self.last_sent: Dict[str, float] = {}

        # Per-verdict cooldown in seconds
        self.cooldown: Dict[str, float] = {
            "SUSPICIOUS":     60.0,
            "LIKELY_CHEAT":   30.0,
            "CONFIRMED_CHEAT": 10.0,
        }

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(
        self,
        verdict: str,
        score: float,
        frame: Optional[np.ndarray] = None,
        evidence_path: Optional[str] = None,
    ) -> None:
        """
        Send a cheat-detection alert to the configured Discord webhook.

        Parameters
        ----------
        verdict       : One of SUSPICIOUS / LIKELY_CHEAT / CONFIRMED_CHEAT
                        (or any custom string).
        score         : Confidence / cheat score in [0, 1].
        frame         : Optional BGR numpy frame.  Encoded as JPEG and
                        attached to the message.
        evidence_path : Optional filesystem path to an evidence clip.
                        Shown as a field in the embed.
        """
        if not self.webhook_url:
            return

        now = time.time()
        cooldown = self.cooldown.get(verdict, 0.0)

        with self._lock:
            last = self.last_sent.get(verdict, 0.0)
            if now - last < cooldown:
                return  # still in cooldown
            self.last_sent[verdict] = now

        # Build in a background thread — never block the caller
        t = threading.Thread(
            target=self._dispatch_alert,
            args=(verdict, score, frame, evidence_path),
            daemon=True,
        )
        t.start()

    def send_session_summary(self, report_path: str, stats: dict) -> None:
        """
        Send a session-end summary embed.

        Parameters
        ----------
        report_path : Filesystem path to the generated report file.
        stats       : Arbitrary dict of stats (keys are used as field names).
        """
        if not self.webhook_url:
            return

        t = threading.Thread(
            target=self._dispatch_summary,
            args=(report_path, stats),
            daemon=True,
        )
        t.start()

    def configure(self, webhook_url: str) -> None:
        """Update the webhook URL at runtime (e.g. from /config endpoint)."""
        with self._lock:
            self.webhook_url = webhook_url

    # ------------------------------------------------------------------
    # Internal dispatch (runs in background threads)
    # ------------------------------------------------------------------

    def _dispatch_alert(
        self,
        verdict: str,
        score: float,
        frame: Optional[np.ndarray],
        evidence_path: Optional[str],
    ) -> None:
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        color = _EMBED_COLOR.get(verdict, _DEFAULT_COLOR)
        title = _EMBED_TITLE.get(verdict, _DEFAULT_TITLE)

        fields = [
            {"name": "Game",    "value": self.game,               "inline": True},
            {"name": "Verdict", "value": f"`{verdict}`",          "inline": True},
            {"name": "Score",   "value": f"{score:.4f}",          "inline": True},
            {"name": "Time",    "value": timestamp_iso,           "inline": True},
        ]
        if evidence_path:
            fields.append(
                {"name": "Evidence clip",
                 "value": f"`{evidence_path}`",
                 "inline": False}
            )

        # If a frame is attached, include a hint in the embed
        if frame is not None:
            fields.append(
                {"name": "Frame snapshot",
                 "value": "Attached below ↓",
                 "inline": False}
            )
            attachment_url = "attachment://frame.jpg"
        else:
            attachment_url = None

        embed: dict = {
            "title": title,
            "color": color,
            "fields": fields,
            "footer": {"text": "CheatVision anti-cheat system"},
            "timestamp": timestamp_iso,
        }
        if attachment_url:
            embed["image"] = {"url": attachment_url}

        payload = {"embeds": [embed]}

        if frame is not None:
            try:
                jpeg_bytes = _encode_frame_jpeg(frame)
            except Exception:
                # Fall back to JSON-only if encoding fails
                _post_json(self.webhook_url, payload)
                return

            json_str = json.dumps(payload)
            _post_multipart(
                self.webhook_url,
                fields={"payload_json": json_str},
                file_bytes=jpeg_bytes,
                filename="frame.jpg",
                file_content_type="image/jpeg",
            )
        else:
            _post_json(self.webhook_url, payload)

    def _dispatch_summary(self, report_path: str, stats: dict) -> None:
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        fields = [
            {"name": "Game",        "value": self.game,    "inline": True},
            {"name": "Report",      "value": f"`{report_path}`", "inline": False},
        ]
        for key, val in stats.items():
            fields.append(
                {"name": str(key), "value": str(val), "inline": True}
            )

        embed = {
            "title": "📊 CheatVision Session Summary",
            "color": 0x5865F2,  # Discord blurple
            "fields": fields,
            "footer": {"text": "CheatVision anti-cheat system"},
            "timestamp": timestamp_iso,
        }
        payload = {"embeds": [embed]}
        _post_json(self.webhook_url, payload)


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python discord_alert.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    url = os.environ.get("CHEATVISION_DISCORD_WEBHOOK", "")
    if not url:
        print("Set CHEATVISION_DISCORD_WEBHOOK to test.")
        sys.exit(0)

    alerter = DiscordAlerter(game="TestGame")

    print("Sending SUSPICIOUS alert …")
    alerter.send_alert("SUSPICIOUS", 0.72)
    time.sleep(1)

    print("Sending LIKELY_CHEAT alert with dummy frame …")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "CheatVision test frame",
                (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    alerter.send_alert("LIKELY_CHEAT", 0.91, frame=dummy_frame,
                        evidence_path="/tmp/evidence_clip.mp4")
    time.sleep(1)

    print("Sending CONFIRMED_CHEAT alert …")
    alerter.send_alert("CONFIRMED_CHEAT", 0.99)
    time.sleep(1)

    print("Sending session summary …")
    alerter.send_session_summary(
        report_path="/tmp/session_report.json",
        stats={
            "duration_s": 1800,
            "frames_analysed": 54000,
            "suspicious_events": 12,
            "confirmed_cheats": 3,
        },
    )

    # Give background threads time to finish
    time.sleep(3)
    print("Done.")
