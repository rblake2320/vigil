#!/usr/bin/env python3
"""
Spark2 Perception Source
------------------------
Linux-native perception layer that fuses structured signals + gated VLM.

Signal hierarchy (cheapest → most expensive):
  1. Window / process state    — xdotool + /proc (always-on, ~0ms)
  2. Filesystem / git events   — inotify via watchdog (always-on, ~0ms)
  3. Screen change gate        — perceptual hash diff (always-on, ~5ms)
  4. Cosmos VLM description    — fires ONLY when gate opens (~2-4s on Spark1)

Output: JSON signals dict printed to stdout each cycle (or feed into engine).

Usage:
    python spark2_source.py                        # watch current desktop
    python spark2_source.py --watch ~/myproject    # also watch a git repo
    python spark2_source.py --cosmos               # enable VLM (needs Spark1 :8000)
    python spark2_source.py --no-screen            # structured signals only
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue, Empty

import imagehash
import mss
import mss.tools
import requests
from PIL import Image
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

COSMOS_URL = "http://10.0.0.1:8000/v1/chat/completions"
COSMOS_MODEL = "cosmos-reason2-8b"
HASH_THRESHOLD = 8          # perceptual hash distance — lower = more sensitive
SCREEN_INTERVAL = 1.0       # seconds between screen checks
WINDOW_INTERVAL = 2.0       # seconds between window/process polls
VLM_COOLDOWN = 8.0          # minimum seconds between Cosmos calls
SCREEN_SCALE = 0.25         # downscale factor for hash (speed)


# ─── Window + Process Source ──────────────────────────────────────────────────

def get_window_state() -> dict:
    state = {}

    try:
        win_id = subprocess.check_output(
            ["xdotool", "getactivewindow"], text=True
        ).strip()
        state["window_id"] = win_id

        title = subprocess.check_output(
            ["xdotool", "getwindowname", win_id], text=True
        ).strip()
        state["window_title"] = title

        pid = subprocess.check_output(
            ["xdotool", "getwindowpid", win_id], text=True
        ).strip()
        state["window_pid"] = pid

        proc_name = Path(f"/proc/{pid}/comm").read_text().strip()
        state["process"] = proc_name

        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
        state["cmdline"] = cmdline[:120]

    except Exception:
        pass

    return state


def get_process_load(top_n: int = 5) -> list:
    try:
        out = subprocess.check_output(
            ["ps", "aux", "--sort=-%cpu"],
            text=True
        ).splitlines()[1 : top_n + 1]
        procs = []
        for line in out:
            parts = line.split(None, 10)
            if len(parts) >= 11:
                procs.append({
                    "pid": parts[1],
                    "cpu": parts[2],
                    "mem": parts[3],
                    "cmd": parts[10][:60],
                })
        return procs
    except Exception:
        return []


# ─── Git / Filesystem Source ──────────────────────────────────────────────────

class GitEventHandler(FileSystemEventHandler):
    def __init__(self, queue: Queue):
        self._q = queue
        self._last = 0.0

    def on_any_event(self, event):
        if event.is_directory:
            return
        now = time.time()
        if now - self._last < 1.0:
            return
        self._last = now
        self._q.put({"fs_event": event.event_type, "path": event.src_path})


def get_git_state(repo_path: str) -> dict:
    try:
        branch = subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", repo_path, "status", "--short"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        staged = len([l for l in status.splitlines() if l and l[0] in "MADRC"])
        unstaged = len([l for l in status.splitlines() if l and l[1] in "MD?"])
        return {"git_branch": branch, "git_staged": staged, "git_unstaged": unstaged}
    except Exception:
        return {}


# ─── Screen + Change Gate ─────────────────────────────────────────────────────

def capture_screen() -> Image.Image:
    with mss.mss() as sct:
        mon = sct.monitors[1]
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
    w = int(img.width * SCREEN_SCALE)
    h = int(img.height * SCREEN_SCALE)
    return img.resize((w, h), Image.LANCZOS)


def image_to_b64(img: Image.Image, scale: float = 0.5) -> str:
    w = int(img.width / SCREEN_SCALE * scale)
    h = int(img.height / SCREEN_SCALE * scale)
    resized = img.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Cosmos VLM ───────────────────────────────────────────────────────────────

def call_cosmos(img: Image.Image, context: str = "") -> str:
    b64 = image_to_b64(img)
    prompt = "Describe what is happening on this screen in 1-2 sentences. Focus on the active task and any visible errors or notable content."
    if context:
        prompt = f"Context: {context}\n\n{prompt}"
    try:
        resp = requests.post(
            COSMOS_URL,
            json={
                "model": COSMOS_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 120,
                "temperature": 0.3,
            },
            timeout=15,
        )
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[cosmos error: {e}]"


# ─── Main Perception Loop ─────────────────────────────────────────────────────

def run(watch_path: str | None, use_cosmos: bool, use_screen: bool):
    fs_queue: Queue = Queue()
    observer = None

    if watch_path:
        handler = GitEventHandler(fs_queue)
        observer = Observer()
        observer.schedule(handler, watch_path, recursive=True)
        observer.start()
        print(f"[perceive] watching {watch_path} for filesystem events")

    last_hash = None
    last_vlm_time = 0.0
    last_window_check = 0.0
    last_window_state: dict = {}
    last_git_state: dict = {}
    cycle = 0

    print("[perceive] starting — Ctrl+C to stop\n")

    try:
        while True:
            signals: dict = {"ts": time.time(), "cycle": cycle}
            cycle += 1

            # ── Window + process (every WINDOW_INTERVAL) ──────────────────
            now = time.time()
            if now - last_window_check >= WINDOW_INTERVAL:
                last_window_check = now
                ws = get_window_state()
                if ws != last_window_state:
                    last_window_state = ws
                    signals["window_changed"] = True
                signals.update(ws)
                signals["top_procs"] = get_process_load(3)
                if watch_path:
                    gs = get_git_state(watch_path)
                    if gs != last_git_state:
                        last_git_state = gs
                        signals["git_changed"] = True
                    signals.update(gs)

            # ── Filesystem events (non-blocking drain) ────────────────────
            fs_events = []
            while True:
                try:
                    fs_events.append(fs_queue.get_nowait())
                except Empty:
                    break
            if fs_events:
                signals["fs_events"] = fs_events

            # ── Screen change gate ────────────────────────────────────────
            gate_open = False
            if use_screen:
                try:
                    img = capture_screen()
                    h = imagehash.phash(img)
                    if last_hash is not None:
                        dist = h - last_hash
                        signals["screen_hash_dist"] = dist
                        if dist >= HASH_THRESHOLD:
                            gate_open = True
                            signals["screen_changed"] = True
                    else:
                        gate_open = True  # first frame always opens
                    last_hash = h
                except Exception as e:
                    signals["screen_error"] = str(e)

            # ── VLM (only when gate open + cooldown elapsed) ──────────────
            if use_cosmos and gate_open and use_screen:
                if time.time() - last_vlm_time >= VLM_COOLDOWN:
                    ctx = last_window_state.get("window_title", "")
                    signals["cosmos"] = call_cosmos(img, ctx)
                    last_vlm_time = time.time()

            # ── Emit ──────────────────────────────────────────────────────
            # Only print if something meaningful changed
            has_signal = (
                signals.get("window_changed")
                or signals.get("git_changed")
                or signals.get("fs_events")
                or signals.get("screen_changed")
                or signals.get("cosmos")
            )
            if has_signal:
                print(json.dumps(signals, indent=2, default=str))
                print("─" * 60)

            time.sleep(SCREEN_INTERVAL)

    except KeyboardInterrupt:
        print("\n[perceive] stopped")
    finally:
        if observer:
            observer.stop()
            observer.join()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark2 perception source")
    parser.add_argument("--watch", metavar="PATH", help="Git repo / directory to watch for filesystem events")
    parser.add_argument("--cosmos", action="store_true", help="Enable Cosmos VLM (requires Spark1 :8000)")
    parser.add_argument("--no-screen", action="store_true", help="Skip screen capture (structured signals only)")
    parser.add_argument("--threshold", type=int, default=HASH_THRESHOLD, help=f"Hash distance threshold (default {HASH_THRESHOLD})")
    args = parser.parse_args()

    HASH_THRESHOLD = args.threshold
    run(
        watch_path=args.watch,
        use_cosmos=args.cosmos,
        use_screen=not args.no_screen,
    )
