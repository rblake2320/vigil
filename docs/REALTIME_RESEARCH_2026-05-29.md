# Real-Time Vision AI Research — May 29, 2026

## The Problem Craig Identified
Cosmos-Reason2-8B takes 3-8 seconds per frame. If someone appears behind you,
the alert fires 3-8 seconds later. That's too slow for personal safety.

## Verified Latency Numbers (Ultralytics official DGX Spark guide)

| Model | Format | Latency |
|---|---|---|
| YOLO11n | PyTorch BF16 | 2.67ms |
| YOLO11n | TensorRT FP32 | 1.95ms |
| YOLO11n | **TensorRT FP16** | **1.01ms** |
| YOLO26 (batch) | TensorRT | sub-1ms |
| Cosmos-Reason2-8B | vLLM BF16 | 3,000-8,000ms |
| StreamingVLM (H100) | — | 125ms (8 FPS) |
| qwen2.5-vl-7b Ollama | — | ~200ms/crop |

**YOLO11n TensorRT FP16 on GB10 = ~1ms = ~990 FPS**

## The Right Architecture: Two-Stage Pipeline

```
Camera frame (5-15ms capture)
        ↓
YOLO11n TensorRT FP16 (~1ms)
Person/fire/object detected?
        │ YES — in ~10-20ms total
        ↓
ALERT FIRES IMMEDIATELY
(SMS, sound, push notification)
        │ PARALLEL (async)
        ↓
Cosmos-Reason2-8B reasoning (3-5s)
"Unknown male, 6ft, approaching from rear doorway"
        ↓
Enhanced alert update / 911 script generated
```

**End-to-end person-detected to alert: 12-43ms** — well under 200ms.

Cosmos is NOT the trigger. Cosmos is the explainer that runs AFTER the alert fires.

## Who Else Is Doing This (2025-2026)

| Player | What | Gap |
|---|---|---|
| Google Gemini Live | Real-time camera analysis on phone | Tells you, doesn't act |
| GPT-4o Realtime API | <320ms via WebSocket, frame-by-frame | No action pipeline |
| Cosmos-Reason2-8B | Edge VLM, spatial reasoning | No action pipeline (we're building it) |
| C2AI (Leidos+NVIDIA) | 911 visual coordination | Gov only, human-in-loop |
| Hanwha Vision | AI → door locks, HVAC | No 911, no escalation |
| StreamingVLM | 8 FPS streaming VLM | Research paper, not a product |

## Five Gaps Nobody Has Filled

1. True end-to-end vision → autonomous 911 dispatch (legal barriers)
2. Edge + cloud hybrid with action pipeline (Cosmos does vision, nobody built the action layer)
3. Multi-camera cross-correlation (each camera is isolated)
4. Streaming VLMs at 30fps in production (StreamingVLM is a paper only)
5. Multi-modal fusion (video + audio + thermal) — most systems handle one sensor

**Vigil is building gap #1 and #2. The SHA-256 evidence chain addresses the legal/liability question.**

## Implementation Path for GB10

```bash
# Step 1: Install YOLO with TensorRT export
pip install ultralytics

# Step 2: Export to TensorRT FP16 (one-time, ~5 min)
yolo export model=yolo11n.pt format=engine half=True device=0

# Step 3: Run at ~1ms per frame
yolo predict model=yolo11n.engine source=0  # webcam or RTSP
```

Then wire the YOLO detection event → alert chain (already built in vigil/).

## Cosmos Role Going Forward

- NOT the trigger (too slow)
- IS the reasoning/context layer after alert fires
- Cap max_new_tokens=64 for security decisions → reduces 3-8s to ~1-2s
- Optional: quantize to W4A16 INT4 for ~2x speedup (embedl/Cosmos-Reason2-8B)

## The Legal Moat

"Can an AI call 911 without human confirmation?" — nobody has solved this framework yet.
Whoever solves it first and gets it into a product wins the category.
The SHA-256 evidence chain (already in Vigil) is the trust layer that makes autonomous dispatch defensible.

## Sources
- Ultralytics DGX Spark Guide: https://docs.ultralytics.com/guides/nvidia-dgx-spark
- StreamingVLM: arXiv 2510.09608
- Cerberus cascade VLM: arXiv 2510.16290
- DEV.to security camera VLM build: real-world validation of 200ms VLM latency
- LMSYS DGX Spark review: GB10 decode ~45-52 tok/s for 7-8B models
