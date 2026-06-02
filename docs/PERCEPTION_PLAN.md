# Vigil Perception Architecture — Full Plan

**Status:** In Build  
**Last Updated:** 2026-06-02  
**Owner:** Craig

---

## What We're Building

A multi-modal perception system that feeds a coaching engine. The engine already exists and is running. This plan covers the perception layer that sits in front of it — how signals get collected, fused, and routed to produce coaching decisions without a human in the loop.

The thesis: **don't make a VLM your only sense**. Pixels are the most expensive and least reliable signal on a computer. Structured signals (window state, process, terminal, git, WiFi) are cheaper, faster, and often more precise. The VLM is gated for moments that are genuinely pixel-only.

---

## Architecture

```
PERCEPTION LAYER
────────────────────────────────────────────────────────────────

PRIMARY (always-on, near-zero cost)
  ├── Window / process state     xdotool + /proc           ~0ms
  ├── Terminal stdout            inotify on log files      ~0ms
  ├── Git events                 watchdog inotify          ~0ms
  └── WiFi CSI sensing           ESP32-S3 → UDP            ~1ms
         │
         ▼
CHANGE GATE (decides when to wake expensive sensors)
  ├── Perceptual hash diff       mss frame → imagehash     ~5ms
  └── WiFi motion delta          variance spike            ~1ms
         │
    YES  │  NO → pass structured signals only, skip VLM
         ▼
SECONDARY (gated, expensive)
  └── Cosmos VLM                 Spark1 :8000              ~2-4s

         │
         ▼
SIGNAL FUSION
  signals = {
    window_title, process, git_branch, git_staged,
    wifi_presence, wifi_motion, wifi_breathing_bpm,
    screen_changed, cosmos,
    fs_events, ...
  }
         │
         ▼
COACHING ENGINE  (perception/coach.py — already built)
  ├── Procedure step matching    structured signals first
  ├── VLM confirm                cosmos description
  └── TTS coaching output        Piper → paplay → HDMI

```

---

## Sources — Build Status

### ✅ DONE: spark2_source.py
Spark2-native perception: xdotool, /proc, watchdog inotify, mss screen capture, perceptual hash gate, Cosmos VLM.

**Run:**
```bash
python -u perception/spark2_source.py
python -u perception/spark2_source.py --cosmos --watch ~/myproject
```

### ✅ DONE: coach.py
Coaching engine that consumes signal dicts. Runs spark2_source as subprocess or accepts piped input. Procedure-based step matching + TTS output.

**Run:**
```bash
python -u perception/coach.py --describe
python -u perception/coach.py --procedure watcher_procedures/it_basic.json
```

### 🔲 IN BUILD: wifi_source.py
WiFi CSI sensing via ESP32-S3. Emits presence, motion level, and breathing rate into the signals dict. See section below.

### 🔲 PLANNED: WindowsUIASource
Win32 UIA accessibility tree diff for the RTX 5090 machine. Structured signal extraction from any Windows app without pixels. Feeds signals to Spark2 brain over HTTP.

### 🔲 PLANNED: FusedSource
Merges all active sources (spark2 + wifi + UIA) into a single signal stream. Priority rules: structured overrides inferred; WiFi presence gates VLM.

---

## WiFi Sensing — Integration Plan

### Why

Camera sees the desk. WiFi sees through walls. Together they cover the room.

Specific value-adds for the coaching use case:
- Detect person approaching before they sit down → prep context
- Detect person leaving → pause/summarize
- Breathing rate as stress/focus proxy during sessions
- Through-wall presence when no camera line-of-sight

### How It Works

ESP32-S3 ($8) sends 802.11 Channel State Information (CSI) packets over UDP at 20 Hz. CSI captures how the WiFi signal is distorted across 52–192 subcarriers. A person in the room changes those distortions. We process the amplitude time series to extract:

- **Presence**: variance of CSI amplitudes above a threshold
- **Motion level**: rate of change of variance (none / low / high)
- **Breathing rate**: FFT peak in 0.2–0.5 Hz band (12–30 BPM)

### Hardware

| Item | Source | Cost |
|---|---|---|
| ESP32-S3 dev board | Amazon / Aliexpress | ~$8–12 |
| Firmware | github.com/espressif/esp-csi | free |

One ESP32-S3 on the same WiFi network as Spark2. Flashed with esp-csi example firmware. Sends CSI data to Spark2 UDP port 5500.

### Signal Output

```python
signals["wifi_presence"]      = True / False
signals["wifi_motion"]        = "none" / "low" / "high"
signals["wifi_breathing_bpm"] = 16          # float, 0 if not detected
signals["wifi_rssi"]          = -62         # dBm
signals["wifi_source"]        = "csi"       # or "rssi" or "sim"
```

### File: perception/wifi_source.py

Modes:
- `--mode csi` — live ESP32 CSI via UDP (real hardware)
- `--mode rssi` — passive RSSI via any NIC in monitor mode (no ESP32 needed)
- `--mode sim` — simulation for testing without hardware

Runs standalone or integrates into FusedSource.

---

## Windows 5090 — Integration Plan

### The Gap

Spark2 watches Spark2's own screen. The RTX 5090 is Craig's primary workstation. To coach Craig at his desk, perception needs to run on Windows or receive signals from it.

### Architecture

```
Windows 5090 (lightweight agent)
  ├── Win32 UIA tree diff        → structured signals
  ├── dxcam GPU frame capture   → JPEG frames on change
  └── POST to Spark2 :8897      → signal receiver

Spark2 (brain)
  ├── /perceive endpoint         receives Windows signals
  ├── Cosmos VLM                 processes frames
  └── coaching engine            same engine, different source
```

### Files Needed

- `windows/uia_source.py` — Win32 UIA collector (runs on 5090)
- `windows/frame_sender.py` — dxcam + HTTP frame pusher (runs on 5090)
- `perception/remote_receiver.py` — FastAPI endpoint on Spark2

---

## Procedure Library — Planned

Current procedures are IT-focused demos. Target procedure categories:

| Category | Examples |
|---|---|
| IT Ops | AD user setup, server deployment, incident response |
| Software Dev | PR review checklist, deployment runbook, code review |
| Security | Pen test workflow, incident triage, vuln scan |
| Medical | Triage protocol, medication checklist |
| Industrial | Equipment startup, safety inspection |
| Finance | Trade execution checklist, audit workflow |

Each procedure JSON: steps with `detect` keyword, `hint`, `timeout_seconds`, and optional `signals` list for structured-signal matching (no VLM needed).

---

## Tested & Verified

| Component | Date | Result |
|---|---|---|
| spark2_source.py signals | 2026-06-02 | ✅ window, process, hash gate, inotify all firing |
| Cosmos VLM gate | 2026-06-02 | ✅ fires on screen change, ~5s latency |
| coach.py → TTS pipeline | 2026-06-02 | ✅ Piper voice loads, paplay → HDMI, audio confirmed |
| Procedure step advance | 2026-06-02 | ✅ detect keyword match advances step |
| wifi_source.py | 🔲 in build | — |
| Windows UIA source | 🔲 planned | — |

---

## Run the Stack Today

```bash
# Full describe mode (screen narration + TTS)
cd ~/ai-business/vigil
python -u perception/coach.py --describe

# Coach against a procedure
python -u perception/coach.py --procedure watcher_procedures/it_basic.json

# With git repo watching
python -u perception/coach.py --describe --watch ~/ai-business/vigil
```
