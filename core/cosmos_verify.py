"""
Cosmos-Reason2-8B Evidence Verifier
====================================
Sends CheatVision evidence clips to Cosmos-Reason2-8B for AI-powered reasoning.
Runs as a background thread — never blocks the capture loop.

Connects to a running NIM container (or any OpenAI-compatible VLM endpoint):
  Default: http://10.0.0.1:8000/v1   (Spark1)
  Fallback: http://localhost:8000/v1  (local)

Usage:
    verifier = CosmosVerifier()
    verifier.submit(clip_path, verdict, score, on_result=callback)
    # callback(clip_path, reasoning_text) fires when done

Environment:
    COSMOS_API_URL   — base URL (default: http://10.0.0.1:8000/v1)
    COSMOS_API_KEY   — API key (default: "nim" — NIM accepts any value)
    COSMOS_ENABLED   — set to "0" to disable without removing the import
"""

import os
import cv2
import time
import base64
import threading
import logging
from pathlib import Path
from typing import Callable, Optional
from queue import Queue, Empty

log = logging.getLogger(__name__)

def _detect_endpoint() -> tuple[str, str, str]:
    """Auto-detect best available vision LLM endpoint. Returns (url, key, model)."""
    import json, urllib.request
    candidates = [
        # (url, key, model_override)
        ("http://localhost:8002/v1",  "local",   "nvidia/cosmos-reason2-8b"),  # vLLM Spark2
        ("http://10.0.0.1:8000/v1",  "local",   "nvidia/cosmos-reason2-8b"),  # NIM Spark1
        ("http://localhost:11434/v1", "ollama",  "qwen3-vl:latest"),           # Ollama (running now)
    ]
    # For Ollama: confirm qwen3-vl model actually exists

    for url, key, model in candidates:
        try:
            req = urllib.request.Request(
                f"{url}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    data = json.loads(r.read())
                    available = [m["id"] for m in data.get("data", [])]
                    # Verify the specific model is present (important for Ollama)
                    if any(model.split(":")[0] in m for m in available) or not available:
                        return url, key, model
        except Exception:
            continue
    return candidates[0]   # default to vLLM even if not yet up

_url, _key, _detected_model = _detect_endpoint()
_DEFAULT_URL   = os.environ.get("COSMOS_API_URL", _url)
_DEFAULT_KEY   = os.environ.get("COSMOS_API_KEY", os.environ.get("NGC_API_KEY", _key))
_MODEL       = os.environ.get("COSMOS_MODEL", _detected_model)
_ENABLED     = os.environ.get("COSMOS_ENABLED", "1") not in ("0", "false", "no")

# How many frames to sample from the clip (4fps is Cosmos training rate)
_SAMPLE_FRAMES = 8
_MAX_TOKENS    = 2048   # reasoning can be verbose
_TIMEOUT_S     = 60     # max wait per verification call

# Prompt tuned for CheatVision evidence clips
_PROMPT = """You are a competitive gaming anti-cheat analyst reviewing evidence.

Analyze this {n_frames}-frame sequence captured from a live game ({game}).
The behavioral detector flagged this clip with verdict: {verdict} (score: {score:.3f}).

Examine the frames carefully and answer:
1. Is the movement/behavior consistent with an aimbot? Look for unnatural snap velocity,
   perfect linearity, or inhuman target acquisition.
2. Is there any ESP/wallhack evidence — colored boxes over walls, pre-aiming through terrain?
3. Is there a no-recoil script? Perfectly flat spray with zero muzzle climb?
4. For Dark War Survival: Is health bar stuck at 100%? Do enemies die impossibly fast?
   Do abilities fire with no cooldown gap?
5. Overall verdict: CONFIRMED CHEAT / LIKELY CHEAT / SUSPICIOUS / CLEAN
6. Confidence: HIGH / MEDIUM / LOW

Be specific — cite frame numbers and describe exactly what you observe.
Do not hallucinate details. If frames are ambiguous, say so."""


def _encode_frame(frame) -> str:
    """Encode a cv2 frame as base64 JPEG for API submission."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _sample_clip(clip_path: str, n: int = _SAMPLE_FRAMES):
    """Extract n evenly-spaced frames from a clip. Returns list of cv2 frames."""
    cap = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _call_cosmos(clip_path: str, verdict: str, score: float,
                 game: str = "Unknown",
                 api_url: str = _DEFAULT_URL,
                 api_key: str = _DEFAULT_KEY,
                 model: str = _MODEL) -> str:
    """
    Send clip frames to Cosmos-Reason2-8B. Returns reasoning text.
    Uses urllib only — no requests lib.
    """
    import json
    import urllib.request

    frames = _sample_clip(clip_path, _SAMPLE_FRAMES)
    if not frames:
        return "Error: could not read clip frames."

    prompt = _PROMPT.format(
        n_frames=len(frames), game=game,
        verdict=verdict, score=score
    )

    # Build content array: alternating images + text
    content = []
    for frame in frames:
        b64 = _encode_frame(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": _MAX_TOKENS,
        "temperature": 0.1,   # low temp for consistent analysis
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

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Cosmos verification failed: {e}"


class CosmosVerifier:
    """
    Background worker that submits evidence clips to Cosmos-Reason2-8B.
    Fires a callback with the reasoning text when done.
    Never blocks — all calls go through a queue.
    """

    def __init__(self,
                 api_url: str = _DEFAULT_URL,
                 api_key: str = _DEFAULT_KEY,
                 game: str = "Unknown"):
        self.api_url = api_url
        self.api_key = api_key
        self.game    = game
        self.enabled = _ENABLED
        self._queue: Queue = Queue()
        self._results: dict = {}   # clip_path -> reasoning text
        self._lock = threading.Lock()

        if self.enabled:
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            log.info("[cosmos] Verifier started → %s", api_url)
        else:
            log.info("[cosmos] Disabled (COSMOS_ENABLED=0)")

    def submit(self,
               clip_path: str,
               verdict: str,
               score: float,
               on_result: Optional[Callable[[str, str], None]] = None):
        """
        Queue a clip for verification.
        on_result(clip_path, reasoning_text) is called in the worker thread when done.
        """
        if not self.enabled:
            return
        if not Path(clip_path).exists():
            log.warning("[cosmos] Clip not found: %s", clip_path)
            return
        self._queue.put((clip_path, verdict, score, on_result))
        log.info("[cosmos] Queued: %s (verdict=%s score=%.3f)", clip_path, verdict, score)

    def get_result(self, clip_path: str) -> Optional[str]:
        """Return cached reasoning text for a clip, or None if not done yet."""
        with self._lock:
            return self._results.get(clip_path)

    def is_ready(self) -> bool:
        """Returns True if Cosmos endpoint is reachable."""
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{self.api_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def set_game(self, game: str):
        self.game = game

    def _worker(self):
        while True:
            try:
                clip_path, verdict, score, callback = self._queue.get(timeout=5)
            except Empty:
                continue
            try:
                log.info("[cosmos] Analyzing: %s", Path(clip_path).name)
                t0 = time.time()
                text = _call_cosmos(
                    clip_path, verdict, score,
                    game=self.game,
                    api_url=self.api_url,
                    api_key=self.api_key,
                )
                elapsed = time.time() - t0
                log.info("[cosmos] Done in %.1fs — %d chars", elapsed, len(text))

                with self._lock:
                    self._results[clip_path] = text

                if callback:
                    try:
                        callback(clip_path, text)
                    except Exception as e:
                        log.error("[cosmos] callback error: %s", e)

            except Exception as e:
                log.error("[cosmos] Worker error: %s", e)
            finally:
                self._queue.task_done()
