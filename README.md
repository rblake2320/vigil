# Vigil — Vision-to-Action Platform

> **Camera sees any event. AI confirms it. Real-world action fires automatically.**

---

## What This Is

Vigil is a platform that connects any camera to any real-world action — with AI understanding the scene in between. It is not motion detection. It is not a notification app.

**Motion detection:** *"Something moved."*
**Vigil:** *"A child under 4 feet tall has been face-down in the pool area for 8 seconds without swimming motion. Calling 911 now. Evidence clip saved."*

---

## Proven

**May 29, 2026 — 03:06:00 AM**

A lighter was held in front of a desktop webcam. The system:

1. Detected fire pixels in the frame (0.104% — orange/yellow flame signature)
2. Woke up the AI automatically
3. AI responded: *"YES FIRE — a person is holding a lighter that emits a bright flame, which is clearly visible in the center of the frame."*
4. Saved SHA-256 hashed evidence frame to disk
5. Logged timestamped event to tamper-evident chain
6. Would have called 911 with full context script

**The Vision-to-Action chain is real and working.**

---

## Architecture

```
Any Camera / HDMI Feed / RTSP Stream
           ↓
    YOLO Detection (4ms per frame on DGX Spark GB10)
    YOLO11 Pose (fall/person posture)
           ↓
    AI Reasoning Layer
    Cosmos-Reason2-8B or qwen3-vl via Ollama
    "YES FIRE — bright flame visible center frame"
           ↓
    Evidence Engine
    SHA-256 hashed video clip
    HMAC-signed tamper-evident JSONL ledger
           ↓
    Surrounding Awareness Engine
    - Timestamp (exact + duration)
    - Location (camera ID, room, address)
    - Time-of-day context (3am = unusual)
    - First occurrence vs repeat event
    - Cross-camera correlation
    - Behavioral baseline
           ↓
    Escalation Cascade
    0s  → Log + evidence saved
    15s → SMS owner: "VIGIL ALERT: Fire at Kitchen, 3:06 AM..."
    30s → Voice call to owner with situation summary
    60s → Call 911 with complete incident report
           ↓
    Real-World Action
    Twilio Voice (E911 capable) + SMS
    Discord/webhook
    Smart home triggers
    Any HTTP endpoint
```

---

## Use Cases

### Safety / Emergency
| Camera sees | Action |
|---|---|
| Fire / flame | Call 911 fire + SMS owner |
| Person falls (horizontal >3s) | Call 911 EMS + family |
| Child in pool not moving | Call 911 rescue immediately |
| Smoke spreading | Call 911 + cut smart home circuits |
| Person collapses mid-stream | Alert + emergency call |

### Security
| Camera sees | Action |
|---|---|
| Unknown person in zone | Alert owner + log |
| Package theft | Evidence clip + owner alert |
| Weapon visible | Silent 911 alert |
| Door opened at night | Immediate push notification |

### Medical / Elder Care
| Camera sees | Action |
|---|---|
| Patient attempts to leave bed | Alert nurse before fall |
| Person on floor not getting up | Call EMS |
| Person wandering outside safe zone | Alert family |

### Workplace / Industrial
| Camera sees | Action |
|---|---|
| Worker without PPE in hazard zone | Lock machinery + alert supervisor |
| Forklift proximity to pedestrian | Audible warning + log |

### Sports Officiating (RealCall)
| Camera sees | Action |
|---|---|
| Ball crosses goal line | Referee buzzer + timestamped call |
| Player contact / foul | Alert + crypto-logged verdict |
| Ball in/out of bounds | Hawk-Eye style ruling |

---

## What Makes This Different

**1. Hardware-external — no agent on the monitored device**
The camera watches from outside. Nothing runs on the monitored system.
Can't be blocked, bypassed, or detected.

**2. SHA-256 cryptographic evidence chain**
Every flagged event produces a tamper-evident video clip.
Each entry is signed and chained to the previous one.
Court-admissible. No existing consumer product does this.

**3. AI understands context — not just pixels**
Not "motion detected." 
"Small child, face-down in water, no swimming movement for 8 seconds, HIGH CONFIDENCE EMERGENCY."
The VLM reasoning layer explains WHY in natural language.

**4. Configurable escalation to 911**
30s→SMS → 60s→call family → 90s→call 911.
User-defined priority order.
No off-the-shelf system has configurable escalation with automatic 911 capability.

**5. Timestamp + surrounding awareness**
Every alert includes: exact time, duration, camera location, time-of-day context,
first occurrence flag, cross-camera correlation, auto-generated 911 call script.

---

## Hardware

Proven on:
- **DGX Spark GB10** (NVIDIA, 120GB unified memory) — primary inference
- **Elgato 4K X** — HDMI capture (any screen as camera source)
- **Any RTSP IP camera** — direct stream ingestion
- **USB webcam** — /dev/video0

YOLO inference: 4ms per frame = 250 FPS capability on GB10.
VLM reasoning: Cosmos-Reason2-8B (purpose-built for physical world events).

---

## Repository Structure

```
watcher.py                        — Continuous screen watcher (software capture, temporal VLM)
vigil_live.py                     — Real-time YOLO + VLM monitor with browser UI (port 8896)

watcher_procedures/               — JSON procedure files for coach mode
  it_basic.json                   — IT onboarding: AD user creation (8 steps)
  fire_watch.json                 — Industrial boiler startup with SCADA (6 steps)
  README.md                       — Procedure format documentation

core/
  vigil_demo.py                   — Live event detector (fire/fall/person/any)
  vigil_context.py                — SurroundingAwareness engine (timestamp, 911 script, escalation)
  auto_evidence.py                — SHA-256 evidence chain, tamper-evident ledger
  cosmos_verify.py                — Cosmos-Reason2-8B / qwen3-vl VLM reasoning
  realtime_monitor.py             — YOLO11 + VLM two-stage detection loop
  context_session.py              — Session context + bootstrap from frame
  session_report.py               — HTML session report generator

alerts/
  discord_alert.py                — Discord webhook with frame attachments
  twilio_alert.py                 — Voice call + SMS + E911 (TODO: wire in)

detectors/
  fire_detector.py                — Fire/smoke detection (PROVEN)
  fall_detector.py                — YOLO11 Pose fall detection (designed)
  intrusion.py                    — Person-in-zone detection (designed)
  pool_safety.py                  — Child-in-water detection (designed)

cameras/
  rtsp_source.py                  — RTSP stream ingestion
  elgato_source.py                — HDMI capture via Elgato
  webcam_source.py                — USB camera

signatures/
  fps.py                          — FPS game cheat signatures (from CheatVision)
  darkwarsurvival.py              — Dark War Survival RTS signatures
  sports.py                       — Sports officiating call signatures

docs/
  PROOF.md                        — Documentation of what was proven and when
  PATENT.md                       — Patent #13 claim space
  ARCHITECTURE.md                 — Full system design
```

---

## Quick Start

```bash
# --- Continuous screen watcher (new) ---
# Describe what's on screen continuously
python watcher.py --mode describe --fps 2

# Coach mode: follow a procedure step by step with TTS guidance
python watcher.py --mode coach --procedure watcher_procedures/it_basic.json --fps 2

# Monitor mode: silent, log to file, speak only on anomalies
python watcher.py --mode monitor --log /tmp/vigil_watch.log --fps 1

# Watch a specific screen region only
python watcher.py --mode describe --region 0,0,1920,1080 --fps 2 --clip-frames 6

# --- Original camera-based detection ---
# Watch any camera for fire
python core/vigil_demo.py --source 0              # webcam
python core/vigil_demo.py --source rtsp://...     # IP camera
python core/vigil_demo.py --source elgato         # HDMI via Elgato

# Live browser UI (YOLO + VLM two-stage)
python vigil_live.py --source elgato --port 8896 --conf 0.35

# Configure a camera profile
from core.vigil_context import CameraProfile, SurroundingAwareness

awareness = SurroundingAwareness()
awareness.register_camera(CameraProfile(
    camera_id="kitchen_cam",
    room="Kitchen",
    address="123 Main Street",
    normal_hours=(6, 22),
    owner_phone="+1555...",
))
```

---

## Continuous Screen Watcher (watcher.py)

`watcher.py` extends Vigil beyond camera feeds to watch any software screen — no HDMI capture hardware needed. It uses `mss` for zero-dependency software screen capture and sends temporal multi-frame clips to Cosmos-Reason2-8B for deep contextual understanding.

### Architecture

```
mss software capture (any screen region, configurable FPS)
        ↓
  Rolling frame buffer (N seconds)
        ↓
  Change detection (grayscale mean-abs-diff, skip static screens)
        ↓
  Clip sampler (6 frames evenly sampled over last 3s)
        ↓
  Cosmos-Reason2-8B (multi-image API call with rolling 5-description context)
        ↓
  Mode handler
    describe → speak every observation via Piper TTS
    coach    → compare to procedure step, speak only corrections/advances
    monitor  → silent watch, speak only on anomalies
        ↓
  Piper TTS → paplay → HDMI audio
```

### Key design decisions

- **Multi-frame temporal clips** — 6 frames sent as separate `image_url` blocks in one API call so Cosmos sees motion over time, not a single snapshot
- **Rolling description context** — last 5 descriptions injected into each prompt so the model knows what it already saw and focuses on changes
- **Change detection** — computes mean absolute diff on downsampled grayscale to skip VLM calls when screen is static (saves tokens, reduces latency)
- **Non-blocking TTS queue** — Piper synthesis and paplay run in a background thread; new observations queue up without blocking the capture loop
- **Zero extra dependencies** — uses only `mss`, `piper`, `cv2`, `numpy`, `urllib.request` (all already installed)

### Coach mode and Procedure JSON

Coach mode loads a JSON procedure file that describes a multi-step task. The watcher:
1. Sends each clip to Cosmos with the current step's description and detect keyword
2. If Cosmos sees screen evidence of completion, advances to the next step and speaks it
3. If not, speaks a coaching reminder with the step's `hint` text

See `watcher_procedures/README.md` for the full format specification.

---

## Status

| Component | Status |
|---|---|
| Fire detection | ✅ PROVEN (2026-05-29) |
| SHA-256 evidence chain | ✅ Working |
| AI reasoning (qwen3-vl) | ✅ Active via Ollama |
| AI reasoning (Cosmos-Reason2-8B) | ✅ 0.7s on Spark1 NIM :8000 |
| Surrounding awareness engine | ✅ Built |
| 911 call script generation | ✅ Built |
| **Continuous screen watcher** | ✅ Built (watcher.py, 2026-06-02) |
| **Multi-frame temporal VLM** | ✅ Built (6-frame clips to Cosmos) |
| **Coach mode + procedure JSON** | ✅ Built (watcher_procedures/) |
| **Software screen capture (mss)** | ✅ Built (no Elgato hardware needed) |
| Fall detection | 🔵 Designed, not deployed |
| Pool safety | 🔵 Designed, not deployed |
| Twilio voice call | 🔵 Wired, not connected |

---

## Related

- **CheatVision** (original proof): `rblake2320/behaviorshield-anticheat`
  The HDMI-capture anti-cheat system where this architecture was first proven.
  Vigil is the generalization of that architecture to all verticals.

- **Session handoff**: See `behaviorshield-anticheat/SESSION_HANDOFF_2026-05-29.md`
  for complete state of all systems and next steps.

---

*Craig — DGX Spark, 2026*
