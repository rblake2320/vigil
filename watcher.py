"""
watcher.py — Continuous Screen-Watching System
================================================
Captures any screen region at configurable FPS, batches frames into temporal
clips, sends multi-frame clips to Cosmos-Reason2-8B for understanding, and
speaks observations via Piper TTS.

Modes:
  describe  — continuous narration of what's on screen (default)
  coach     — loads a JSON procedure, speaks only corrections / next-step prompts
  monitor   — silent watch, logs to file, speaks only on anomalies

Usage:
  python watcher.py --mode describe --fps 2
  python watcher.py --mode coach --procedure watcher_procedures/it_basic.json --fps 2
  python watcher.py --mode monitor --log /tmp/watcher.log --fps 1

Architecture:
  FrameCapture (mss, configurable region + FPS)
      ↓ rolling frame buffer
  ChangeDetector (mean-abs-diff grayscale, skip static screens)
      ↓ changed clips
  ClipSampler (pick N frames from last W seconds)
      ↓ base64-encoded JPEG list
  CosmosClient (POST multi-image to :8000/v1/chat/completions)
      ↓ description text + rolling context
  Coach / Monitor / Narrator
      ↓ text
  PiperTTS → paplay → HDMI audio
"""

import os
import sys
import cv2
import mss
import json
import time
import base64
import logging
import argparse
import threading
import tempfile
import subprocess
import urllib.request
import urllib.error
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("watcher")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
COSMOS_URL   = os.environ.get("COSMOS_API_URL",  "http://10.0.0.1:8000/v1")
COSMOS_KEY   = os.environ.get("COSMOS_API_KEY",  os.environ.get("NGC_API_KEY", "nim"))
COSMOS_MODEL = os.environ.get("COSMOS_MODEL",    "nvidia/cosmos-reason2-8b")
COSMOS_TIMEOUT = int(os.environ.get("COSMOS_TIMEOUT", "60"))

PIPER_VOICE  = os.environ.get(
    "PIPER_VOICE",
    "/home/rblake2320/piper-voices/en_US-ryan-high.onnx",
)

# How many past descriptions to include in each prompt as rolling context
CONTEXT_WINDOW = 5

# Change-detection: skip clip if mean absolute diff between consecutive
# grayscale frames is below this threshold (0–255 scale)
CHANGE_THRESHOLD = float(os.environ.get("CHANGE_THRESHOLD", "4.0"))


# ---------------------------------------------------------------------------
# Piper TTS
# ---------------------------------------------------------------------------
class PiperTTS:
    """Wraps Piper voice synthesis → paplay. Non-blocking via thread queue."""

    def __init__(self, voice_path: str = PIPER_VOICE):
        self.voice_path = voice_path
        self._voice = None
        self._lock = threading.Lock()
        self._queue: deque = deque(maxlen=4)   # drop oldest if backed up
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._busy = False

    def _load(self):
        if self._voice is None:
            try:
                from piper import PiperVoice
                self._voice = PiperVoice.load(self.voice_path)
                log.info("[tts] Voice loaded: %s", self.voice_path)
            except Exception as e:
                log.error("[tts] Failed to load voice: %s", e)

    def speak(self, text: str):
        """Queue text for TTS. Returns immediately."""
        if not text or not text.strip():
            return
        self._queue.append(text.strip())

    def _worker(self):
        while True:
            if not self._queue:
                time.sleep(0.05)
                continue
            text = self._queue.popleft()
            self._busy = True
            try:
                self._load()
                if self._voice is None:
                    log.warning("[tts] No voice loaded, skipping: %s", text[:60])
                    continue
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                try:
                    with open(wav_path, "wb") as wav_file:
                        self._voice.synthesize(text, wav_file)
                    subprocess.run(
                        ["paplay", wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                finally:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
            except Exception as e:
                log.error("[tts] Speak error: %s", e)
            finally:
                self._busy = False

    @property
    def is_speaking(self) -> bool:
        return self._busy or bool(self._queue)


# ---------------------------------------------------------------------------
# Frame capture (mss software screen capture)
# ---------------------------------------------------------------------------
class FrameCapture:
    """
    Captures screen frames using mss at a target FPS.
    Runs in a background thread, fills a rolling deque.
    """

    def __init__(
        self,
        monitor: Optional[dict] = None,
        fps: float = 2.0,
        buffer_seconds: float = 5.0,
    ):
        # monitor dict: {"top": 0, "left": 0, "width": 1920, "height": 1080}
        # If None, uses primary monitor (mss monitor 1)
        self.monitor = monitor
        self.fps = max(0.1, fps)
        self.interval = 1.0 / self.fps
        self.buffer_size = max(6, int(buffer_seconds * self.fps))
        self.frames: deque = deque(maxlen=self.buffer_size)
        self.frame_times: deque = deque(maxlen=self.buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("[capture] Started at %.1f FPS, buffer=%d frames", self.fps, self.buffer_size)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _capture_loop(self):
        with mss.mss() as sct:
            # Auto-detect primary monitor if none specified
            mon = self.monitor
            if mon is None:
                info = sct.monitors[1]  # monitor 1 = primary
                mon = {
                    "top":    info["top"],
                    "left":   info["left"],
                    "width":  info["width"],
                    "height": info["height"],
                }
                log.info("[capture] Monitor: %dx%d @ (%d,%d)",
                         mon["width"], mon["height"], mon["left"], mon["top"])

            while self._running:
                t_start = time.monotonic()
                try:
                    img = sct.grab(mon)
                    # mss returns BGRA — convert to BGR for cv2 compat
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    with self._lock:
                        self.frames.append(frame)
                        self.frame_times.append(time.time())
                except Exception as e:
                    log.error("[capture] Grab error: %s", e)

                elapsed = time.monotonic() - t_start
                sleep_t = self.interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    def get_clip(self, n_frames: int = 6) -> list:
        """Return up to n_frames evenly sampled from the buffer (newest first → oldest last)."""
        with self._lock:
            buf = list(self.frames)
        if not buf:
            return []
        if len(buf) <= n_frames:
            return list(buf)
        # Evenly sample
        indices = [int(i * (len(buf) - 1) / (n_frames - 1)) for i in range(n_frames)]
        return [buf[i] for i in indices]

    @property
    def frame_count(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------
class ChangeDetector:
    """
    Compares the most recent clip against the previous one.
    Returns True if enough visual change warrants a VLM call.
    """

    def __init__(self, threshold: float = CHANGE_THRESHOLD, downsample: int = 4):
        self.threshold = threshold
        self.downsample = downsample  # spatial downsampling factor for speed
        self._prev_gray: Optional[np.ndarray] = None

    def has_changed(self, frames: list) -> bool:
        """Return True if frames are meaningfully different from last check."""
        if not frames:
            return False
        # Use the middle frame as the representative frame
        frame = frames[len(frames) // 2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downsample for speed
        h, w = gray.shape
        small = cv2.resize(gray, (w // self.downsample, h // self.downsample))

        if self._prev_gray is None:
            self._prev_gray = small
            return True  # first frame always passes

        diff = float(np.mean(np.abs(small.astype(np.float32) - self._prev_gray.astype(np.float32))))
        self._prev_gray = small

        changed = diff >= self.threshold
        if not changed:
            log.debug("[change] Skipped (diff=%.2f < threshold=%.2f)", diff, self.threshold)
        else:
            log.debug("[change] Clip passed (diff=%.2f)", diff)
        return changed


# ---------------------------------------------------------------------------
# Cosmos VLM client
# ---------------------------------------------------------------------------
def encode_frame_b64(frame) -> str:
    """Encode a BGR cv2 frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def call_cosmos(
    frames: list,
    prompt: str,
    model: str = COSMOS_MODEL,
    api_url: str = COSMOS_URL,
    api_key: str = COSMOS_KEY,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: int = COSMOS_TIMEOUT,
) -> str:
    """
    Send a multi-frame clip to Cosmos-Reason2-8B.
    frames — list of BGR np.ndarray
    Returns the model's text response.
    """
    content = []
    for frame in frames:
        b64 = encode_frame_b64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{api_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Rolling description context
# ---------------------------------------------------------------------------
class DescriptionContext:
    """
    Maintains a rolling window of the last N descriptions so the VLM
    always knows what it already saw.
    """

    def __init__(self, max_entries: int = CONTEXT_WINDOW):
        self.max_entries = max_entries
        self._entries: deque = deque(maxlen=max_entries)

    def add(self, description: str):
        self._entries.append(description.strip())

    def as_text(self) -> str:
        if not self._entries:
            return "(No prior observations.)"
        lines = [f"  [{i+1}] {d}" for i, d in enumerate(self._entries)]
        return "\n".join(lines)

    def __len__(self):
        return len(self._entries)


# ---------------------------------------------------------------------------
# Procedure loader (for coach mode)
# ---------------------------------------------------------------------------
class Procedure:
    """Loads and tracks a JSON step-by-step procedure."""

    def __init__(self, path: str):
        self.path = path
        with open(path) as f:
            data = json.load(f)
        self.name: str = data.get("name", "Untitled Procedure")
        self.steps: list[dict] = data.get("steps", [])
        self.current_step_idx: int = 0
        log.info("[procedure] Loaded '%s' with %d steps", self.name, len(self.steps))

    @property
    def current_step(self) -> Optional[dict]:
        if self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_step_idx >= len(self.steps)

    def advance(self):
        self.current_step_idx += 1
        if self.is_complete:
            log.info("[procedure] All steps complete!")
        else:
            log.info("[procedure] Advancing to step %d: %s",
                     self.current_step_idx + 1,
                     self.current_step["description"])

    def check_step_complete(self, description: str) -> bool:
        """
        Returns True if the current step's detect keyword appears in
        the VLM description (case-insensitive).
        """
        step = self.current_step
        if step is None:
            return False
        detect = step.get("detect", "")
        return detect.lower() in description.lower()


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def _build_describe_prompt(context: DescriptionContext, n_frames: int) -> str:
    return f"""You are observing a live computer screen.
You are looking at a {n_frames}-frame sequence captured over the last few seconds.

Previous observations (most recent last):
{context.as_text()}

Describe what is currently happening on the screen in 1-3 concise sentences.
Focus on: what application is open, what the user is doing, any notable changes.
Be specific and factual. Do not repeat what was already observed unless it changed."""


def _build_coach_prompt(context: DescriptionContext, procedure: Procedure, n_frames: int) -> str:
    step = procedure.current_step
    if step is None:
        return _build_describe_prompt(context, n_frames)

    return f"""You are a coaching assistant watching someone follow a procedure on their computer.

Procedure: {procedure.name}
Current expected step ({step['id']} of {len(procedure.steps)}):
  {step['description']}
  (Look for: "{step.get('detect', 'anything relevant')}")

Previous screen observations (most recent last):
{context.as_text()}

You are seeing a {n_frames}-frame sequence from RIGHT NOW.

Answer these two questions:
1. SCREEN: What do you currently see on screen? (1-2 sentences)
2. MATCH: Does the screen show evidence of completing step {step['id']}? Answer YES or NO, then explain briefly.

Keep your total response under 80 words."""


def _build_monitor_prompt(context: DescriptionContext, n_frames: int) -> str:
    return f"""You are a silent security monitor watching a computer screen.

Previous observations (most recent last):
{context.as_text()}

You are seeing a {n_frames}-frame sequence captured just now.

Is there anything anomalous, unexpected, or worth reporting?
Examples: error dialogs, unexpected programs, security warnings, system alerts, unusual activity.

If nothing anomalous: respond with exactly "NORMAL"
If anomalous: respond with "ANOMALY: " followed by a brief description (max 2 sentences)."""


# ---------------------------------------------------------------------------
# Watcher (main orchestrator)
# ---------------------------------------------------------------------------
class Watcher:
    """
    Continuous screen watcher. Ties together:
      FrameCapture → ChangeDetector → CosmosClient → TTS
    """

    def __init__(
        self,
        mode: str = "describe",
        fps: float = 2.0,
        clip_frames: int = 6,
        monitor: Optional[dict] = None,
        procedure_path: Optional[str] = None,
        log_path: Optional[str] = None,
        tts_enabled: bool = True,
        change_threshold: float = CHANGE_THRESHOLD,
        vlm_interval: float = 3.0,  # min seconds between VLM calls
    ):
        self.mode = mode
        self.fps = fps
        self.clip_frames = clip_frames
        self.vlm_interval = vlm_interval
        self.log_path = log_path
        self.tts_enabled = tts_enabled

        self.capture = FrameCapture(monitor=monitor, fps=fps, buffer_seconds=max(5.0, clip_frames / fps))
        self.change_detector = ChangeDetector(threshold=change_threshold)
        self.context = DescriptionContext(max_entries=CONTEXT_WINDOW)
        self.tts = PiperTTS() if tts_enabled else None
        self.procedure: Optional[Procedure] = None

        if mode == "coach":
            if not procedure_path:
                raise ValueError("--procedure required for coach mode")
            self.procedure = Procedure(procedure_path)

        self._running = False
        self._last_vlm_time: float = 0.0
        self._log_file = None

        if log_path:
            self._log_file = open(log_path, "a", buffering=1)
            log.info("[watcher] Logging observations to %s", log_path)

    def _speak(self, text: str):
        if self.tts and text:
            self.tts.speak(text)

    def _log_observation(self, text: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {text}\n"
        if self._log_file:
            self._log_file.write(line)
        log.info("[obs] %s", text[:120])

    def _handle_describe(self, description: str):
        """In describe mode: speak every observation."""
        self._speak(description)

    def _handle_coach(self, description: str):
        """In coach mode: check procedure progress, speak corrections."""
        if self.procedure is None or self.procedure.is_complete:
            if self.procedure and self.procedure.is_complete:
                self._speak("Procedure complete. All steps have been verified.")
            return

        step = self.procedure.current_step
        step_done = self.procedure.check_step_complete(description)

        if step_done:
            self.procedure.advance()
            if self.procedure.is_complete:
                self._speak(f"Step {step['id']} complete. Procedure finished. Well done.")
            else:
                next_step = self.procedure.current_step
                self._speak(
                    f"Step {step['id']} complete. "
                    f"Next: {next_step['description']}"
                )
        else:
            # Parse the MATCH answer from the VLM response
            lower = description.lower()
            if "match: no" in lower or "match: n" in lower:
                # VLM confirmed step is NOT yet done — coach the user
                self._speak(
                    f"Still on step {step['id']}: {step['description']}. "
                    f"Look for: {step.get('detect', 'the expected screen element')}."
                )
            else:
                # Ambiguous or "YES" — log but don't speak
                log.debug("[coach] Step %d — observation logged", step["id"])

    def _handle_monitor(self, description: str):
        """In monitor mode: only speak anomalies."""
        if description.upper().startswith("ANOMALY"):
            anomaly_text = description[description.find(":") + 1:].strip()
            log.warning("[monitor] ANOMALY: %s", anomaly_text)
            self._speak(f"Anomaly detected. {anomaly_text}")
        else:
            log.debug("[monitor] NORMAL")

    def _process_clip(self, frames: list):
        """Send frames to Cosmos, update context, dispatch to mode handler."""
        n = len(frames)
        if self.mode == "describe":
            prompt = _build_describe_prompt(self.context, n)
        elif self.mode == "coach":
            prompt = _build_coach_prompt(self.context, self.procedure, n)
        else:
            prompt = _build_monitor_prompt(self.context, n)

        t0 = time.time()
        try:
            description = call_cosmos(frames, prompt)
        except Exception as e:
            log.error("[cosmos] Call failed: %s", e)
            return
        elapsed = time.time() - t0
        log.info("[cosmos] %.1fs → %d chars", elapsed, len(description))

        self.context.add(description)
        self._log_observation(description)

        if self.mode == "describe":
            self._handle_describe(description)
        elif self.mode == "coach":
            self._handle_coach(description)
        elif self.mode == "monitor":
            self._handle_monitor(description)

    def run(self):
        """Main blocking loop."""
        self._running = True
        self.capture.start()

        mode_desc = {
            "describe": "Continuous narration",
            "coach":    f"Coaching: '{self.procedure.name}'" if self.procedure else "Coaching",
            "monitor":  "Silent monitor (speaks anomalies only)",
        }.get(self.mode, self.mode)

        log.info("[watcher] Starting — mode=%s  FPS=%.1f  clip_frames=%d",
                 self.mode, self.fps, self.clip_frames)
        log.info("[watcher] %s", mode_desc)

        if self.tts_enabled:
            self._speak(f"Watcher started. Mode: {self.mode}.")
            if self.procedure:
                self._speak(
                    f"Procedure loaded: {self.procedure.name}. "
                    f"Step 1: {self.procedure.steps[0]['description']}"
                )

        try:
            while self._running:
                now = time.time()
                time_since_last = now - self._last_vlm_time

                # Respect minimum VLM interval
                if time_since_last < self.vlm_interval:
                    time.sleep(0.1)
                    continue

                # Need enough frames in buffer
                if self.capture.frame_count < 2:
                    time.sleep(0.1)
                    continue

                frames = self.capture.get_clip(self.clip_frames)
                if not frames:
                    time.sleep(0.1)
                    continue

                # Skip if screen hasn't changed meaningfully
                if not self.change_detector.has_changed(frames):
                    time.sleep(0.2)
                    continue

                self._last_vlm_time = time.time()
                # Process in a thread so capture loop isn't blocked
                t = threading.Thread(target=self._process_clip, args=(frames,), daemon=True)
                t.start()

        except KeyboardInterrupt:
            log.info("[watcher] Interrupted by user.")
        finally:
            self._running = False
            self.capture.stop()
            if self._log_file:
                self._log_file.close()
            log.info("[watcher] Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Vigil — Continuous screen watcher with temporal VLM coaching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python watcher.py --mode describe --fps 2
  python watcher.py --mode coach --procedure watcher_procedures/it_basic.json --fps 2
  python watcher.py --mode monitor --log /tmp/watcher.log --fps 1
  python watcher.py --mode describe --region 0,0,1920,1080 --fps 2 --clip-frames 6
  python watcher.py --mode describe --no-tts   # silent, log only
        """,
    )

    p.add_argument(
        "--mode",
        choices=["describe", "coach", "monitor"],
        default="describe",
        help="Watcher mode (default: describe)",
    )
    p.add_argument(
        "--procedure",
        type=str,
        default=None,
        help="Path to procedure JSON file (required for coach mode)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Screen capture frames per second (default: 2.0)",
    )
    p.add_argument(
        "--clip-frames",
        type=int,
        default=6,
        help="Number of frames per VLM clip (default: 6)",
    )
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="Screen region as left,top,width,height (default: full primary monitor)",
    )
    p.add_argument(
        "--vlm-interval",
        type=float,
        default=3.0,
        help="Minimum seconds between VLM calls (default: 3.0)",
    )
    p.add_argument(
        "--change-threshold",
        type=float,
        default=CHANGE_THRESHOLD,
        help=f"Change detection threshold 0-255 (default: {CHANGE_THRESHOLD}). Lower = more sensitive.",
    )
    p.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to observation log file (optional)",
    )
    p.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech (log only)",
    )
    p.add_argument(
        "--cosmos-url",
        type=str,
        default=COSMOS_URL,
        help=f"Cosmos API base URL (default: {COSMOS_URL})",
    )
    p.add_argument(
        "--cosmos-model",
        type=str,
        default=COSMOS_MODEL,
        help=f"Cosmos model ID (default: {COSMOS_MODEL})",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Apply CLI overrides for Cosmos endpoint
    if args.cosmos_url != COSMOS_URL:
        os.environ["COSMOS_API_URL"] = args.cosmos_url
        # Reload module-level constant for call_cosmos
        import watcher as _self
        _self.COSMOS_URL = args.cosmos_url

    if args.cosmos_model != COSMOS_MODEL:
        os.environ["COSMOS_MODEL"] = args.cosmos_model

    # Parse region
    monitor = None
    if args.region:
        try:
            left, top, width, height = map(int, args.region.split(","))
            monitor = {"left": left, "top": top, "width": width, "height": height}
            log.info("[main] Capture region: %dx%d @ (%d,%d)", width, height, left, top)
        except ValueError:
            log.error("--region must be left,top,width,height (e.g. 0,0,1920,1080)")
            sys.exit(1)

    watcher = Watcher(
        mode=args.mode,
        fps=args.fps,
        clip_frames=args.clip_frames,
        monitor=monitor,
        procedure_path=args.procedure,
        log_path=args.log,
        tts_enabled=not args.no_tts,
        change_threshold=args.change_threshold,
        vlm_interval=args.vlm_interval,
    )

    watcher.run()


if __name__ == "__main__":
    main()
