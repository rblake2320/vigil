#!/usr/bin/env python3
"""
perception/blink_source.py — Blink camera polling + YOLO detection
-------------------------------------------------------------------
Polls all 5 Blink cameras every 60s (API throttle limit), runs YOLO11n
on each snapshot, emits detections as signals dict.

Credentials: ~/.config/blink/credentials.json (written by blink_setup.py)

Output per camera (JSON to stdout, separator line after):
  {
    "source": "blink",
    "camera": "Living room",
    "detections": {"person": 2, "suitcase": 1},
    "person": true,
    "snapshot": "/path/to/snap.jpg",
    "ts": 1234567890.0
  }

Usage:
    python -u perception/blink_source.py
    python -u perception/blink_source.py --interval 60
    python -u perception/blink_source.py --camera "Living room"   # single cam
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

CREDS_FILE  = Path.home() / ".config/blink/credentials.json"
SNAP_DIR    = Path(__file__).parent.parent / "blink_snaps"
YOLO_MODEL  = Path(__file__).parent.parent / "yolo11n.pt"
MIN_INTERVAL = 60   # Blink API throttle
CONF         = 0.3


# ─── Blink auth ───────────────────────────────────────────────────────────────

async def load_blink():
    from aiohttp import ClientSession
    from blinkpy.blinkpy import Blink
    from blinkpy.auth import Auth

    if not CREDS_FILE.exists():
        raise FileNotFoundError(f"No credentials at {CREDS_FILE} — run blink_setup.py first")

    creds = json.loads(CREDS_FILE.read_text())
    session = ClientSession()
    blink = Blink(session=session)
    blink.auth = Auth(creds, no_prompt=True)
    await blink.start()
    return blink, session


# ─── YOLO detection ───────────────────────────────────────────────────────────

_yolo = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(str(YOLO_MODEL))
    return _yolo


def detect(image_path: str) -> dict:
    model = get_yolo()
    results = model(image_path, verbose=False, device="cpu", conf=CONF)
    counts = {}
    if results and results[0].boxes:
        for c in results[0].boxes.cls:
            label = results[0].names[int(c)]
            counts[label] = counts.get(label, 0) + 1
    return counts


# ─── Poll loop ────────────────────────────────────────────────────────────────

async def poll_once(blink, camera_filter: str | None = None):
    await blink.refresh()
    results = []
    for name, cam in blink.cameras.items():
        if camera_filter and camera_filter.lower() not in name.lower():
            continue

        snap_path = SNAP_DIR / f"{name.replace(' ', '_').replace('/', '_')}.jpg"
        try:
            await cam.snap_picture()
            await blink.refresh()
            await cam.image_to_file(str(snap_path))
            detections = detect(str(snap_path))
        except Exception as e:
            print(f"[blink] {name}: error — {e}", file=sys.stderr)
            detections = {}

        sig = {
            "source":     "blink",
            "camera":     name,
            "detections": detections,
            "person":     "person" in detections,
            "snapshot":   str(snap_path),
            "ts":         time.time(),
        }
        results.append(sig)
        print(json.dumps(sig, indent=2), flush=True)
        print("─" * 60, flush=True)

    return results


async def run(interval: int, camera_filter: str | None):
    print(f"[blink] loading credentials from {CREDS_FILE}", file=sys.stderr)
    blink, session = await load_blink()
    print(f"[blink] connected — {len(blink.cameras)} cameras: {list(blink.cameras)}", file=sys.stderr)

    try:
        while True:
            t0 = time.time()
            await poll_once(blink, camera_filter)
            elapsed = time.time() - t0
            wait = max(0, interval - elapsed)
            print(f"[blink] next poll in {int(wait)}s", file=sys.stderr, flush=True)
            await asyncio.sleep(wait)
    finally:
        await session.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blink camera polling source")
    parser.add_argument("--interval", type=int, default=MIN_INTERVAL,
                        help=f"Poll interval seconds (min {MIN_INTERVAL}, default {MIN_INTERVAL})")
    parser.add_argument("--camera", help="Filter to a single camera by name substring")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()

    interval = max(MIN_INTERVAL, args.interval)
    if args.once:
        async def _once():
            blink, session = await load_blink()
            await poll_once(blink, args.camera)
            await session.close()
        asyncio.run(_once())
    else:
        asyncio.run(run(interval, args.camera))
