# Spark Vision Pipeline — Blink → Local LLM

## Architecture

```
Blink Cameras (cloud)
        │
        │  blinkpy (poll every 60s)
        ▼
  blink_snaps/*.jpg          ← snapshots saved locally
        │
        │  base64 encode
        ▼
Spark-1 Ollama :11434        ← vision model inference
        │
        │  JSON response
        ▼
  Vigil alert pipeline       ← action dispatch
```

## Vision Models on Spark-1

| Alias | Model | Size | Speed | Best for |
|-------|-------|------|-------|----------|
| `fast` | `moondream:latest` | 1.7GB | ~1-2s | Quick presence checks, high-frequency polling |
| `default` | `llama3.2-vision:latest` | 7.8GB | ~3-5s | General scene analysis |
| `smart` | `qwen3-vl:latest` | 6.1GB | ~4-6s | Detailed descriptions, multilingual |
| `deep` | `llava:34b` | 20GB | ~10-20s | Deep analysis, edge cases |

## Quick Start

```python
from perception.spark_vision import SparkVision

sv = SparkVision(model="fast")   # or "default", "smart", "deep"

# Analyze one snapshot
result = await sv.analyze("blink_snaps/Door.jpg", "Is anyone at the door?")
print(result["response"])

# Analyze all cameras at once
results = await sv.analyze_all(prompt="Any people or unusual activity?")
for r in results:
    print(f"[{r['camera']}] {r['response']}")
```

## Run the test

```bash
# From vigil root — must have blink_snaps populated first
python perception/spark_vision.py
```

## Continuous Watch Loop

```python
from perception.blink_source import BlinkSource
from perception.spark_vision import SparkVision

bs = BlinkSource(cred_file="~/.config/blink/credentials.json")
sv = SparkVision(model="fast")

async for alert in sv.watch_loop(bs, interval=60):
    if "person" in alert["response"].lower():
        print(f"ALERT [{alert['camera']}]: {alert['response']}")
```

## Prompt Ideas by Camera

| Camera | Suggested Prompt |
|--------|-----------------|
| Door | `"Is anyone at the door? Note if person is facing camera or walking away."` |
| Living room | `"Any people or unusual activity in this living space?"` |
| Kitchen | `"Is anyone in the kitchen? Any safety concerns (fire, water)?"` |
| My room | `"Presence check — is anyone in this room?"` |
| GNT1-9001-3364-8H30 | `"Describe the scene. Note people, objects, activity level."` |

## Extending

- **Add YOLO + LLM**: Use existing YOLO detectors (Vigil core) for fast bounding boxes,
  then only send crops of detected objects to the LLM for description.
- **Alert routing**: Wire `spark_vision.py` results into `alerts/` directory for
  downstream action dispatch.
- **Multi-model consensus**: Run `moondream` for speed, escalate to `llava:34b`
  only when moondream returns a positive detection.

## Spark-2 Note

Spark-2 (10.0.0.2) runs the primary Blink poller and Vigil services. Spark-1
(192.168.12.132) has the heavy Ollama models. Use Spark-1 for vision inference.
Spark-2 Ollama is a lighter instance — check before routing there.
