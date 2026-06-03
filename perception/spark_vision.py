"""
Spark Vision Pipeline
---------------------
Sends Blink camera snapshots to local vision models running on Spark-1 via Ollama.

Available vision models on Spark-1 (:11434):
  - moondream:latest       1.7GB  — fastest, good for presence/object detection
  - llama3.2-vision:latest 7.8GB  — solid all-rounder
  - qwen3-vl:latest        6.1GB  — strong multilingual, good scene description
  - llava:34b              20GB   — most powerful, use for deep analysis

Usage:
  from perception.spark_vision import SparkVision
  sv = SparkVision()
  result = await sv.analyze("blink_snaps/Door.jpg", prompt="Is anyone at the door?")
"""

import asyncio
import base64
import pathlib
import httpx

SPARK1_OLLAMA = "http://192.168.12.132:11434"

# Model tiers — pick based on speed vs accuracy need
MODELS = {
    "fast":    "moondream:latest",        # ~1-2s, lightweight
    "default": "llama3.2-vision:latest",  # ~3-5s, balanced
    "smart":   "qwen3-vl:latest",         # ~4-6s, strong reasoning
    "deep":    "llava:34b",               # ~10-20s, most capable
}


class SparkVision:
    def __init__(self, spark_url: str = SPARK1_OLLAMA, model: str = "default"):
        self.base_url = spark_url
        self.model = MODELS.get(model, model)

    async def analyze(self, image_path: str, prompt: str, model: str | None = None) -> dict:
        """Send an image to Spark vision model and return the response."""
        img = pathlib.Path(image_path)
        if not img.exists():
            return {"error": f"Image not found: {image_path}"}

        img_b64 = base64.b64encode(img.read_bytes()).decode()
        model_name = MODELS.get(model, model) if model else self.model

        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return {
                "camera": img.stem,
                "model": model_name,
                "prompt": prompt,
                "response": data.get("response", ""),
                "done": data.get("done", False),
            }

    async def analyze_all(
        self,
        snap_dir: str = "blink_snaps",
        prompt: str = "Describe what you see. Note any people, movement, or unusual activity.",
        model: str | None = None,
    ) -> list[dict]:
        """Analyze all snapshots in a directory concurrently."""
        snaps = list(pathlib.Path(snap_dir).glob("*.jpg"))
        if not snaps:
            return [{"error": f"No snapshots found in {snap_dir}"}]

        tasks = [self.analyze(str(s), prompt, model) for s in snaps]
        return await asyncio.gather(*tasks)

    async def watch_loop(
        self,
        blink_source,
        prompt: str = "Is there a person or unusual activity? Answer yes/no then explain briefly.",
        model: str = "fast",
        interval: int = 60,
    ):
        """
        Continuous loop: pull Blink snapshots, run vision analysis, yield alerts.

        Example:
            from perception.blink_source import BlinkSource
            bs = BlinkSource(...)
            sv = SparkVision(model="fast")
            async for alert in sv.watch_loop(bs):
                print(alert)
        """
        import time
        sv_model = MODELS.get(model, model)
        while True:
            try:
                frames = await blink_source.get_frames()
                for frame in frames:
                    result = await self.analyze(frame["snapshot"], prompt, sv_model)
                    result["ts"] = time.time()
                    result["source"] = "blink"
                    result["camera"] = frame.get("camera", pathlib.Path(frame["snapshot"]).stem)
                    yield result
            except Exception as e:
                yield {"error": str(e)}
            await asyncio.sleep(interval)


# ── Quick test ──────────────────────────────────────────────────────────────

async def _test():
    sv = SparkVision(model="fast")

    # Health check
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{SPARK1_OLLAMA}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            vision = [m for m in models if any(v in m for v in ["vision", "llava", "moondream", "vl"])]
            print(f"Spark-1 Ollama up. Vision models: {vision}")
        except Exception as e:
            print(f"Spark-1 unreachable: {e}")
            return

    # Analyze whatever snaps exist
    snaps = list(pathlib.Path("blink_snaps").glob("*.jpg"))
    if not snaps:
        print("No snapshots in blink_snaps/ — run blink_auth_finish.py first")
        return

    print(f"\nAnalyzing {len(snaps)} snapshot(s) with {sv.model}...\n")
    results = await sv.analyze_all(prompt="Who or what do you see? Be concise.")
    for r in results:
        print(f"[{r['camera']}] {r['response']}\n")


if __name__ == "__main__":
    asyncio.run(_test())
