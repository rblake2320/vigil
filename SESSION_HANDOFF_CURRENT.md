# Session Handoff — 2026-05-30 09:39:46

> **READ THIS FIRST.** Written by `write_handoff.py`. Captures live system state at time of writing.
> Next session: run `cat ~/ai-business/SESSION_HANDOFF_CURRENT.md` before doing anything.

## What Was Happening This Session

<!-- AUTO-FILLED: edit manually before exiting if you have context -->

### In Progress
- Vigil audio-description platform: WORKING ✅
  - Piper TTS (en_US-ryan-high) → paplay → HDMI → audible ✅
  - YOLO11n on CPU (device="cpu") — avoids GB10 CUDA-12.1 / PyTorch-12.0 mismatch ✅
  - 37-class detection (people, vehicles, weapons, devices, animals)
  - Auto-labeling: saves JPEG + YOLO .txt + Cosmos .meta.json on every alert
  - Manual capture: POST /capture/start {"label":"pistol","seconds":10} for weapon training
  - Elgato 4K X as live camera source (--source elgato)
- CheatVision: NOT running (done gaming) — starts with `cd ~/ai-business/cheatvision && python viewer.py`
- Step-3.7-Flash (port 8898): running but text-only, 500s on vision, Vigil falls back to Cosmos

### Next Up
- Craig to do weapon capture session (pistol, rifle, knife, etc.) using /capture/start endpoint
- Clean up both Sparks (Spark1 + Spark2) to free 256GB unified RAM
- YOLO fine-tune on collected training data when weapon samples are ready
- Patent #12 ($65, patentcenter.uspto.gov) — Vigil passive audio description for blind

## Service Status at 2026-05-30 09:39:46

| Port | Service | Status |
|---|---|---|
| 8765 | AI Army Hub | ✅ UP |
| 8891 | CheatVision viewer | ✅ UP |
| 8892 | QuoteHub Gary Vee | ✅ UP |
| 8895 | Cosmos Chat Monitor | ✅ UP |
| 8896 | Vigil Live Monitor | ✅ UP |
| 8898 | Step-3.7-Flash (llama-server) | ✅ UP |
| 8000 | Cosmos-Reason2-8B (Spark1 :8000) | ✅ UP |
| 8881 | VSS Frontend | ✅ UP |
| 8880 | VSS Backend | ✅ UP |

## System Resources

```
Mem:           121Gi       115Gi       7.0Gi       249Mi       6.0Gi       6.4Gi
GPU (MB used/free/total): [N/A], [N/A], [N/A]
```

**Top Python processes:**
```
2379556 1.0 /home/rblake2320/miniconda3/bin/python
3461404 0.0 /home/rblake2320/miniconda3/bin/python
3014184 0.0 /home/rblake2320/miniconda3/bin/python
3014162 0.0 /usr/bin/python3
1012296 0.0 /home/rblake2320/miniconda3/bin/python
2615184 0.0 /usr/bin/python3
```

## Start Commands

```bash
# Vigil (primary — audio description, real-time monitor)
cd ~/ai-business/vigil && nohup ~/miniconda3/bin/python vigil_live.py --source elgato --port 8896 --conf 0.35 > /tmp/vigil_live.log 2>&1 &

# CheatVision (gaming only)
cd ~/ai-business/cheatvision && nohup ~/miniconda3/bin/python viewer.py > /tmp/viewer.log 2>&1 &

# Dashboard (side-by-side) — http://192.168.1.72:8891/dashboard
# Vigil UI — http://192.168.1.72:8896
# Training stats — curl http://192.168.1.72:8896/training
# TTS toggle — curl -X POST http://192.168.1.72:8896/tts/toggle

# Weapon capture example
curl -X POST http://192.168.1.72:8896/capture/start \
  -H "Content-Type: application/json" \
  -d '{"label":"pistol","seconds":10,"fps":5}'
```

## Key Files

| File | Purpose |
|---|---|
| `~/ai-business/vigil/vigil_live.py` | Main Vigil server (TTS, capture, endpoints) |
| `~/ai-business/vigil/core/realtime_monitor.py` | YOLO detector, Cosmos/Step VLM, DataCollector |
| `~/ai-business/vigil/vigil_training/` | Auto-collected training data (images/labels/meta) |
| `~/piper-voices/en_US-ryan-high.onnx` | Piper TTS neural voice model |
| `~/ai-business/cheatvision/viewer.py` | CheatVision viewer + /dashboard route |

## Architecture

```
Elgato 4K X → YOLO11n (CPU, ~110ms) → ALERT fires immediately
                                      ↓ async
                    Step-3.7-Flash (500s on vision) → fallback →
                    Cosmos-Reason2-8B (10.0.0.1:8000) → description
                                      ↓
                    Piper TTS (en_US-ryan-high) → paplay → HDMI audio
                                      ↓
                    DataCollector → vigil_training/ (YOLO format)
```

## Recent Git History

### vigil (rblake2320/vigil)
```
055640d Add audio description pipeline, expanded detection, auto-labeling, and Step-3.7-Flash VLM
34c4c14 Add real-time two-stage pipeline: YOLO fast trigger + Cosmos async reasoning
a2a4767 Research: real-time vision AI latency — YOLO vs Cosmos on GB10
3be064f Complete Vigil platform: detectors, cameras, alerts, architecture docs
b885199 feat: Vigil Vision-to-Action — fire detection PROVEN 2026-05-29 03:06 AM
```

### behaviorshield-anticheat (rblake2320/behaviorshield-anticheat)
```
14e673f Add /dashboard route and fix stream viewport overflow
015046c Session handoff 2026-05-29b: Cosmos fixed, real-time pipeline built, xdotool unblocked
053858e Add cosmos_chat: live monitor UI + proxy server for Cosmos-Reason2-8B
6e1f8f2 SESSION_HANDOFF_2026-05-29: Complete state, both products, all proof
261d011 Add Vigil Vision-to-Action engine: fire detection PROVEN + context system
```

## Recent Log Tails

### Vigil (/tmp/vigil_live.log)
```
2026-05-30 09:39:41,956 WARNING [Step] Failed (HTTP Error 500: Internal Server Error), falling back to Cosmos
2026-05-30 09:39:43,753 INFO [VLM] SAFE 
 Person in Argentina jersey on YouTube, terminal windows visible, no immediate threat detected.
2026-05-30 09:39:44,980 INFO [ALERT] HIGH 
 [{'label': 'person', 'conf': 0.89}]
2026-05-30 09:39:45,016 WARNING [Step] Failed (HTTP Error 500: Internal Server Error), falling back to Cosmos
2026-05-30 09:39:46,592 INFO [VLM] SAFE. No immediate threat detected. Monitor for unusual activity or behavior changes.
```

## Training Data Collected
```
samples: 2173 | classes: ['person', 'tv', 'suitcase', 'truck', 'car', 'laptop', 'clock', 'motorcycle']
```

## Critical Facts (do not forget)

- **Owner:** Craig (NOT Ryan — that was a hallucination)
- **Spark2 IP:** 192.168.1.72 (NOT 192.168.12.223 — that IP is gone)
- **Spark1 IP:** 10.0.0.1 / 192.168.12.132
- **Cosmos on Spark1:** http://10.0.0.1:8000/v1 — VERIFIED WORKING
- **Audio path:** Piper → paplay → PulseAudio compat socket → HDMI (XV271 Z)
- **YOLO must run on CPU** (`device="cpu"`) — GB10 CUDA 12.1 > PyTorch max 12.0
- **Piper API:** `synthesize()` returns `Iterable[AudioChunk]` (piper-tts 1.4.2+), NOT a wave writer
- **Elgato:** needs MJPG format at 1280x720, `cv2.VideoCapture(0)` no backend flag
- **parecord monitor:** works for espeak but capture format quirks make RMS checks unreliable
- **Step-3.7-Flash:** text-only model, always 500s on vision — keep Cosmos fallback

---
*Written by write_handoff.py at 2026-05-30 09:39:46*
