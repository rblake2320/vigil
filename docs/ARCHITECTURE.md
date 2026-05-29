# Vigil — Architecture

## The Chain

```
INPUT (any camera)
   ↓
STAGE 1: Fast heuristic detector (<1ms per frame)
   - fire.py: orange/red pixel saturation spike
   - fall.py: person aspect ratio shift (vertical→horizontal)
   - pool.py: person stationary in water zone
   - (custom): any pixel/shape rule you define
   ↓
STAGE 2: VLM confirmation (triggers only on Stage 1 hit)
   - qwen3-vl:latest via Ollama (always available)
   - Cosmos-Reason2-8B via NIM (Spark1, loading)
   - "Is this actually fire? What do you see?"
   - Eliminates false positives: lamp, sunset, TV screen
   ↓
STAGE 3: Evidence capture
   - auto_evidence.py: 5s pre-trigger clip saved
   - SHA-256 hash computed on clip bytes
   - Chain hash links entries in ledger.jsonl
   - Legally defensible, tamper-evident
   ↓
STAGE 4: Context report (vigil_context.py)
   - Which camera, which room, which address
   - Time of day — is this normal activity hours?
   - How long has event been present?
   - Cross-camera status
   - Full natural-language incident report generated
   ↓
STAGE 5: Escalation cascade (twilio_alert.py)
   -  0s: local alarm + push notification
   - 10s: SMS to owner
   - 30s: automated voice call to owner
   - 60s: no response → call 911 with full incident report
   - 60s: also call fire rescue / EMS / relevant responder
```

## Key Design Decisions

**Two-stage detection** — fast heuristic + VLM confirmation.  
Stage 1 runs every frame in microseconds. Stage 2 (expensive VLM call) only
fires when Stage 1 says something changed. False positive rate is very low.

**SHA-256 chain** — every evidence clip is hashed. Each ledger entry includes
the hash of the previous entry. You cannot alter a clip or delete an entry
without invalidating the chain. Court-admissible.

**Time-of-day baseline** — the system knows what "normal" looks like for each
camera. Fire at 6pm (cooking) has lower urgency score than fire at 3am.
Person in kitchen at 6am (coffee) is normal; at 3am is unusual.

**VLM reasoning in plain language** — when the system calls 911, the dispatcher
script says exactly what the AI saw: "A person is face-down in the pool area
and has not moved for 14 seconds." Not "motion detected."

## Camera Support

Any OpenCV-compatible source:
- `0`, `1` — webcam by index
- `rtsp://ip:port/stream` — IP camera
- `elgato` — HDMI capture via Elgato 4K X (`/dev/video0`)
- `/path/to/video.mp4` — pre-recorded video (for testing)

## Hardware

Proven on: NVIDIA DGX Spark (GB10 GPU, 128GB unified RAM)  
Required: any Linux machine with Python 3.11+, OpenCV, and Ollama  
Optional: NVIDIA GPU for Cosmos-Reason2-8B (better reasoning quality)

## Evidence Format

`evidence_ledger.jsonl` — one JSON line per event:
```json
{
  "timestamp": "2026-05-29T03:06:00",
  "camera_id": "home_office",
  "room": "Home Office",
  "event_type": "fire",
  "duration_s": 14,
  "vlm_reasoning": "A person is holding a lighter that emits a bright flame...",
  "confidence": 94,
  "clip_path": "evidence/20260529_030600.mp4",
  "clip_sha256": "747ac541...",
  "chain_hash": "1e7b09be..."
}
```
