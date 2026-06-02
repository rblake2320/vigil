#!/usr/bin/env python3
"""
perception/coach.py — Perception-driven coaching engine
---------------------------------------------------------
Wires spark2_source structured signals into the watcher coaching logic.
No frame capture loop — signals ARE the input. VLM only fires for pixel-only
moments (screen_changed flag from spark2_source).

Usage:
    # Pipe spark2_source into this coach
    python -u perception/spark2_source.py --cosmos | python perception/coach.py

    # Or run standalone (self-contained, starts its own perception loop)
    python perception/coach.py --procedure watcher_procedures/it_basic.json
    python perception/coach.py --procedure watcher_procedures/it_basic.json --watch ~/myproject
    python perception/coach.py --describe   # narrate mode, no procedure
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from watcher import PiperTTS, Procedure, DescriptionContext, call_cosmos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("coach")

COSMOS_URL   = os.environ.get("COSMOS_API_URL", "http://10.0.0.1:8000/v1")
COSMOS_MODEL = os.environ.get("COSMOS_MODEL",   "nvidia/cosmos-reason2-8b")


# ─── Signal → Text bridge ─────────────────────────────────────────────────────

def signals_to_text(signals: dict) -> str:
    """Convert a structured signals dict to a human-readable observation string."""
    parts = []

    title = signals.get("window_title", "")
    proc  = signals.get("process", "")
    if title or proc:
        parts.append(f"Active: {proc} — {title}".strip(" —"))

    if signals.get("git_branch"):
        staged   = signals.get("git_staged", 0)
        unstaged = signals.get("git_unstaged", 0)
        parts.append(f"Git: branch={signals['git_branch']}, staged={staged}, unstaged={unstaged}")

    for fe in signals.get("fs_events", []):
        parts.append(f"File {fe.get('event','?')}: {Path(fe.get('path','')).name}")

    if signals.get("cosmos"):
        parts.append(f"Screen: {signals['cosmos']}")

    if signals.get("screen_changed") and not signals.get("cosmos"):
        parts.append("Screen changed (no VLM description)")

    return " | ".join(parts) if parts else ""


# ─── Coaching loop ─────────────────────────────────────────────────────────────

class PerceptionCoach:
    def __init__(
        self,
        procedure_path: Optional[str] = None,
        describe: bool = False,
        watch_path: Optional[str] = None,
        use_cosmos: bool = True,
        tts: bool = True,
    ):
        self.describe    = describe
        self.use_cosmos  = use_cosmos
        self.context     = DescriptionContext(max_entries=5)
        self.procedure   = Procedure(procedure_path) if procedure_path else None
        self.tts         = PiperTTS() if tts else None
        self.watch_path  = watch_path
        self._last_spoke = 0.0
        self._speak_cooldown = 6.0

    def _speak(self, text: str):
        if not self.tts or not text.strip():
            return
        now = time.time()
        if now - self._last_spoke < self._speak_cooldown:
            return
        self._last_spoke = now
        log.info("[TTS] %s", text[:120])
        self.tts.speak(text)

    def _handle_describe(self, obs: str):
        self.context.add(obs)
        self._speak(obs)

    def _handle_coach(self, obs: str, signals: dict):
        if not self.procedure or self.procedure.is_complete:
            return

        step = self.procedure.current_step
        if not step:
            return

        self.context.add(obs)

        # Check if step signals match directly from structured data
        step_done = self._check_step_from_signals(step, signals)

        if step_done is None:
            # Can't determine from structured signals alone — ask VLM if available
            cosmos_desc = signals.get("cosmos", "")
            if cosmos_desc:
                step_done = self.procedure.check_step_complete(cosmos_desc)

        if step_done:
            self.procedure.advance()
            next_step = self.procedure.current_step
            if next_step:
                msg = f"Good. Next: {next_step['description']}"
            else:
                msg = f"Procedure '{self.procedure.name}' complete."
            log.info("[coach] %s", msg)
            self._speak(msg)
        else:
            hint = step.get("hint", step.get("description", ""))
            log.debug("[coach] step %d still in progress — %s", step["id"], hint[:60])
            # Only re-prompt if screen actually changed (avoid nagging)
            if signals.get("screen_changed") or signals.get("window_changed"):
                self._speak(f"Step {step['id']}: {hint}")

    def _check_step_from_signals(self, step: dict, signals: dict) -> Optional[bool]:
        """
        Try to verify a step using structured signals without VLM.
        Returns True (done), False (not done), or None (can't tell).
        """
        keywords = step.get("signals", [])
        if not keywords:
            return None  # no structured signal hints defined for this step

        title = (signals.get("window_title") or "").lower()
        proc  = (signals.get("process") or "").lower()
        text  = f"{title} {proc}"

        matched = sum(1 for kw in keywords if kw.lower() in text)
        if matched >= len(keywords):
            return True
        return None

    def process(self, signals: dict):
        obs = signals_to_text(signals)
        if not obs:
            return

        log.info("[signals] %s", obs[:140])

        if self.describe:
            self._handle_describe(obs)
        elif self.procedure:
            self._handle_coach(obs, signals)
        else:
            # No procedure — just log and narrate screen changes
            if signals.get("cosmos") or signals.get("screen_changed"):
                self._handle_describe(obs)

    def run_from_stdin(self):
        """Read JSON signal dicts from stdin (piped from spark2_source.py)."""
        log.info("[coach] reading signals from stdin")
        buf = ""
        for line in sys.stdin:
            line = line.strip()
            if line == "─" * 60 or line.startswith("─"):
                if buf.strip():
                    try:
                        signals = json.loads(buf)
                        self.process(signals)
                    except json.JSONDecodeError:
                        pass
                buf = ""
            else:
                buf += line + "\n"

    def run_standalone(self):
        """Start spark2_source as a subprocess and consume its output."""
        cmd = [
            sys.executable, "-u",
            str(Path(__file__).parent / "spark2_source.py"),
            "--threshold", "6",
        ]
        if self.use_cosmos:
            cmd.append("--cosmos")
        if self.watch_path:
            cmd += ["--watch", self.watch_path]

        log.info("[coach] launching perception: %s", " ".join(cmd))
        if self.procedure:
            step = self.procedure.current_step
            if step:
                log.info("[coach] starting procedure '%s' — step 1: %s",
                         self.procedure.name, step["description"])
                self._speak(f"Starting procedure: {self.procedure.name}. Step one: {step['description']}")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        buf = ""
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith("─"):
                    if buf.strip():
                        try:
                            signals = json.loads(buf)
                            self.process(signals)
                        except json.JSONDecodeError:
                            pass
                    buf = ""
                elif line.startswith("[perceive]"):
                    log.info(line)
                else:
                    buf += line + "\n"
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()
            proc.wait()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perception-driven coaching engine")
    parser.add_argument("--procedure", metavar="JSON", help="Procedure file to coach against")
    parser.add_argument("--describe",  action="store_true", help="Narrate mode — no procedure, just describe")
    parser.add_argument("--watch",     metavar="PATH", help="Git repo / dir to watch for fs events")
    parser.add_argument("--no-cosmos", action="store_true", help="Disable VLM (structured signals only)")
    parser.add_argument("--no-tts",    action="store_true", help="Disable TTS (log only)")
    parser.add_argument("--stdin",     action="store_true", help="Read signals from stdin (piped from spark2_source)")
    args = parser.parse_args()

    if not args.procedure and not args.describe:
        parser.error("Provide --procedure <json> or --describe")

    coach = PerceptionCoach(
        procedure_path=args.procedure,
        describe=args.describe,
        watch_path=args.watch,
        use_cosmos=not args.no_cosmos,
        tts=not args.no_tts,
    )

    if args.stdin:
        coach.run_from_stdin()
    else:
        coach.run_standalone()
