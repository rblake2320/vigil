"""
vigil/core/context_session.py
==============================
Context-aware commentary engine for Vigil.

Turns "a man gestures" into "Canelo throws a left hook, Bivol slips right."

Flow:
  1. Bootstrap — VLM reads the scene, classifies it, extracts entities
  2. ContextSession stores type + entities + commentary mode + interval
  3. Domain prompt injected into every VLM call
  4. Continuous mode: timer fires VLM on interval regardless of label changes
"""

import json
import base64
import urllib.request
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

STEP_URL   = "http://localhost:8898/v1/chat/completions"   # primary
COSMOS_URL = "http://10.0.0.1:8000/v1/chat/completions"   # fallback

# ── Bootstrap prompt ──────────────────────────────────────────────────────────
BOOTSTRAP_PROMPT = """Look at this image carefully. Identify what kind of content this is.

Reply with ONLY a JSON object — no explanation, no markdown, just raw JSON:
{
  "type": "boxing" | "sports" | "youtube" | "news" | "tv_show" | "general",
  "channel": "channel or platform name or null",
  "topic": "main subject or show name or null",
  "entities": {
    "fighters": ["name1", "name2"] (if boxing),
    "teams": ["team1", "team2"] (if sports),
    "sport": "sport name" (if sports),
    "score": "score string" (if visible),
    "round": "round number or period" (if visible),
    "presenter": "name if visible" (if youtube/tv),
    "anchor": "name if visible" (if news)
  },
  "commentary_mode": "continuous" | "change_only",
  "interval_seconds": 2 (boxing/action) | 5 (sports) | 8 (youtube/news) | 10 (general)
}

For boxing: continuous=true, interval=2
For live sports: continuous=true, interval=4
For YouTube/news/shows: change_only, interval=8
Read ALL visible text — lower-thirds, score bugs, chyrons, logos — to fill entities."""

# ── Domain prompt templates ───────────────────────────────────────────────────
DOMAIN_PROMPTS = {
    "boxing": (
        "You are a boxing radio commentator describing live action for a blind listener.\n"
        "Fighters: {fighters}. {round_info}\n"
        "Describe THIS MOMENT in one sentence: punches (jab/cross/hook/uppercut/body shot), "
        "defense (slip/bob/weave/clinch/guard), ring movement, who has the advantage.\n"
        "Name the fighter doing the action. Be immediate and specific. Max 20 words."
    ),
    "sports": (
        "You are a sports radio commentator for a blind listener.\n"
        "{sport} — {teams}. {score_info}\n"
        "Describe the current play/action in one sentence. Name players if visible. "
        "What just happened? Max 20 words."
    ),
    "youtube": (
        "You are describing a YouTube video to a blind viewer.\n"
        "Channel: {channel}. Topic: {topic}.\n"
        "What is on screen RIGHT NOW? Read any visible text. "
        "What is the presenter doing or showing? Max 20 words."
    ),
    "news": (
        "You are describing a news broadcast to a blind viewer.\n"
        "Channel: {channel}. {anchor_info}\n"
        "What is on screen? Read any chyrons, headlines, or lower-third text exactly. "
        "Describe footage or graphics. Max 20 words."
    ),
    "tv_show": (
        "You are describing a TV show to a blind viewer.\n"
        "Show: {topic}. Channel: {channel}.\n"
        "What is happening in this scene? Who is speaking? What action is occurring? Max 20 words."
    ),
    "general": (
        "Describe this image in one sentence for a blind person. "
        "Read any visible text. Name people, objects, setting. "
        "Be specific and concrete. Max 20 words."
    ),
}

UPDATE_PROMPTS = {
    "boxing": (
        "Boxing: {fighters}. {round_info}\n"
        "Previously: {context}\n"
        "What changed? New punch, defense, clinch, knockdown, pause? "
        "If same stalemate, say SAME. Otherwise one sentence, max 15 words."
    ),
    "sports": (
        "{sport} — {teams}.\nPreviously: {context}\n"
        "What changed? New play, score, foul, timeout? If same, say SAME. Max 15 words."
    ),
    "youtube": (
        "YouTube — {channel}: {topic}\nPreviously: {context}\n"
        "What changed on screen? New slide, text, action? If same, say SAME. Max 15 words."
    ),
    "news": (
        "News — {channel}\nPreviously: {context}\n"
        "New headline, footage, or chyron? Read it. If same, say SAME. Max 15 words."
    ),
    "tv_show": (
        "TV — {topic}\nPreviously: {context}\n"
        "What changed? New scene, speaker, action? If same, say SAME. Max 15 words."
    ),
    "general": (
        "Previously: {context}\n"
        "What is NEW or DIFFERENT? If nothing changed, say SAME. Max 15 words."
    ),
}


@dataclass
class ContextSession:
    type: str = "general"
    channel: Optional[str] = None
    topic: Optional[str] = None
    entities: dict = field(default_factory=dict)
    commentary_mode: str = "change_only"   # "continuous" or "change_only"
    interval: float = 8.0                  # seconds between VLM calls in continuous mode
    active: bool = False

    def describe(self) -> str:
        parts = [f"Type: {self.type}"]
        if self.channel:
            parts.append(f"Channel: {self.channel}")
        if self.topic:
            parts.append(f"Topic: {self.topic}")
        if self.entities:
            for k, v in self.entities.items():
                if v:
                    parts.append(f"{k}: {v}")
        parts.append(f"Mode: {self.commentary_mode} ({self.interval}s)")
        return " | ".join(parts)

    def build_prompt(self, known_context: str = "") -> str:
        """Build the right domain prompt with entities injected."""
        e = self.entities
        kwargs = {
            "channel":     self.channel or "unknown",
            "topic":       self.topic or "unknown",
            "context":     known_context,
            "fighters":    " vs ".join(e.get("fighters", [])) or "unknown fighters",
            "round_info":  f"Round {e['round']}." if e.get("round") else "",
            "teams":       " vs ".join(e.get("teams", [])) or "unknown teams",
            "sport":       e.get("sport", "sports"),
            "score_info":  f"Score: {e['score']}." if e.get("score") else "",
            "anchor_info": f"Anchor: {e['anchor']}." if e.get("anchor") else "",
            "presenter":   e.get("presenter", "presenter"),
        }
        if known_context and self.type in UPDATE_PROMPTS:
            template = UPDATE_PROMPTS[self.type]
        else:
            template = DOMAIN_PROMPTS.get(self.type, DOMAIN_PROMPTS["general"])
        try:
            return template.format(**kwargs)
        except KeyError:
            return DOMAIN_PROMPTS["general"]


def bootstrap_from_frame(frame, vlm_url: str = STEP_URL) -> Optional[ContextSession]:
    """
    Send a frame to the VLM to classify the scene and extract context.
    Returns a populated ContextSession or None on failure.
    """
    import cv2
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(jpg.tobytes()).decode()

    def _call(url, model):
        p = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": BOOTSTRAP_PROMPT},
        ]}], "max_tokens": 300, "temperature": 0.1}
        req = urllib.request.Request(url, data=json.dumps(p).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        with urllib.request.urlopen(req, timeout=30) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
            return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    # Try Step first, fall back to Cosmos
    text = ""
    try:
        text = _call(STEP_URL, "step-3.7-flash")
    except Exception as e:
        log.warning(f"[Bootstrap] Step failed ({e}), trying Cosmos")
        try:
            text = _call(COSMOS_URL, "nvidia/cosmos-reason2-8b")
        except Exception as e2:
            log.warning(f"[Bootstrap] Cosmos also failed: {e2}")
            return None
    try:

        # Strip markdown code fences if present
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        data = json.loads(text)
        entities = data.get("entities", {})
        # Flatten list values for readability
        flat_entities = {}
        for k, v in entities.items():
            if isinstance(v, list):
                flat_entities[k] = v
            elif v:
                flat_entities[k] = v

        session = ContextSession(
            type=data.get("type", "general"),
            channel=data.get("channel"),
            topic=data.get("topic"),
            entities=flat_entities,
            commentary_mode=data.get("commentary_mode", "change_only"),
            interval=float(data.get("interval_seconds", 8.0)),
            active=True,
        )
        log.info(f"[Bootstrap] {session.describe()}")
        return session

    except Exception as e:
        log.warning(f"[Bootstrap] Failed: {e} — raw: {text[:200] if 'text' in dir() else '?'}")
        return None
