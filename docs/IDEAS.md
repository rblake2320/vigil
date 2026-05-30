# Vigil — Ideas & Future Exploration

> Running notes on ideas to explore. Not a roadmap — a thinking board.

## Meta Glasses as Camera Source (HIGH PRIORITY)

**The idea:** Replace the Elgato capture card with Meta Ray-Ban glasses as the camera.
The glasses see what the wearer sees → Vigil describes it in real time via Piper TTS in their ear.

**Why this is big:**
- Meta glasses stream first-person POV video
- Vigil already: detects objects → VLM describes scene → Piper speaks it
- Target users: partially sighted people (not completely blind — can see shapes but not detail)
  - "There's a person in a red shirt about 6 feet ahead"
  - "The sign says 'Exit' pointing left"
  - "There's a car turning toward you"
- No product does this passively and automatically today
- Audio description (AD) exists for TV/film but NOT for live first-person daily life

**Technical path:**
- Meta glasses → RTSP stream or USB tether → Vigil `--source rtsp://...`
- OR: Meta glasses companion app → push frames to Vigil HTTP endpoint
- Already works: any camera source → YOLO → Cosmos → Piper → audio

**Gap in market confirmed by research:**
- Microsoft Seeing AI: must actively point phone, not passive
- Google Lookout: same — active, phone-based
- Nobody has: always-on, wearable, first-person passive description

---

## UI Features (IN PROGRESS)

### Chat Interface ✅ BUILT
- Text input in Vigil UI: type "what is that person doing?" → VLM answers → spoken aloud
- Endpoint: `POST /ask {"question": "..."}`
- Answer appears in event log + TTS speaks it

### Draw ROI (Region of Interest) ✅ BUILT
- Click "Draw ROI" button → draw a box on the feed
- YOLO detection focuses only on that area
- Useful for: watching a doorway, a specific shelf, a person's hands

### Hover Mode ✅ BUILT
- Click "Hover Mode" → move mouse over feed
- 50%×50% focus box follows your cursor
- YOLO focuses where you point
- Useful for: guided exploration, pointing at something to ask about it

---

## Two-Spark Architecture (NEXT)

**Current problem:** Both Sparks saturated (~114GB / 121GB each).

**Proposed split:**
```
Spark2 (192.168.1.72)          Spark1 (10.0.0.1)
────────────────────           ────────────────────
Vigil (camera + YOLO)          Cosmos-Reason2-8B (vLLM :8000)
Piper TTS                      UltraRAG + PostgreSQL
Qwen2-VL via Ollama ← new      [freed capacity for larger models]
```

**To do:**
1. Kill Step-3.7-Flash on Spark2 (text-only, wastes GPU, returns 500 on vision)
2. Kill gnome-system-monitor memory leak on Spark2 (~17GB)
3. Pull `qwen2-vl:7b` via Ollama on Spark2
4. Point Vigil Stage 2 at local Ollama for fast vision description
5. Keep Cosmos on Spark1 for heavy reasoning
6. 200GbE link (192.168.100.x) between Sparks for fast cross-inference

---

## Weapon / Object Training (PLANNED)

- Use `/capture/start` endpoint to build custom YOLO classes
- Session: hold up each weapon for 10s → 50 labeled frames per class
- Classes wanted: pistol, rifle, knife, shotgun, taser
- After collection: `yolo train data=vigil_training/data.yaml model=yolo11n.pt epochs=50`
- Fine-tuned model loads into Vigil → actual weapon detection (not "baseball bat")

---

## Audio Description Modes (IDEAS)

- **Continuous mode:** describe every N seconds even without detection trigger
- **Change mode:** only speak when something changes in the scene
- **Focus mode:** describe only the thing the user points at (hover mode → ask)
- **Alert mode:** current behavior — only speak on YOLO trigger
- **Persona:** choose voice/style — "clinical security" vs "friendly assistant" vs "navigation guide"

---

## Voice Options

- Current: `en_US-ryan-high` (male, natural)
- Available: `en_US-amy-medium` (female), `en_US-lessac-high` (female, high quality)
- Future: voice cloning via XTTS — let user pick their preferred voice
- Speed control: piper synthesis rate adjustable

---

## Patent Angle — What's Novel

Patent #12 target: Passive real-time visual description for the partially sighted using
wearable camera + edge AI + neural TTS — always-on, no user action required.

Key differentiators from prior art:
1. **Passive** — no button press, no phone pointing
2. **Wearable first-person POV** — glasses, not phone camera
3. **Two-stage** — fast YOLO trigger (<20ms) + rich VLM description (async)
4. **Personalized focus** — hover/draw ROI lets user direct AI attention
5. **Conversational** — ask questions about what you're seeing
6. **Edge + cloud hybrid** — YOLO local, VLM can be local or remote
7. **Training flywheel** — auto-collects labeled data from real use

---

## Other Camera Sources to Test

- `--source 0` : webcam
- `--source elgato` : HDMI capture (current)
- `--source rtsp://IP:PORT/stream` : IP camera (Craig's .119 camera — needs creds)
- `--source http://URL/stream` : MJPEG stream (CheatVision stream works this way)
- Meta Ray-Ban: investigate RTSP output or companion API
