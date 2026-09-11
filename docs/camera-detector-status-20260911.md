# Camera-independent detector work: measured status

This branch is an evaluation build. It does not enable automatic fall, breathing, fire or face-identity alerts on the owner's Blink cameras.

## Implemented and executed

- `tools/fall_clip.py` accepts a local decoded video independently of camera brand, verifies the supplied YOLO11 pose weights hash, runs on CPU with two threads, and saves per-frame timestamps, boxes, torso visibility inputs, track IDs, frame hashes and a result receipt. Missing observations are not safety verdicts. External process timeout bounds the run.
- `detectors/temporal_fall.py` evaluates per-track upright/drop/horizontal transitions in media time. Eleven synthetic kinematics regressions passed, including preexisting horizontal pose, identity changes, gaps, malformed input and portrait/landscape equivalence. These are logic tests, not a real-fall accuracy benchmark.
- One actual Blink recording was downloaded through the signed-in Android app. The first 15 seconds produced 91 inference observations in 1.657 seconds of processing, with 43 insufficient observations and 48 no-candidate observations. No fall candidate was observed. This is a real video-path result, not a positive fall proof. Private video and detailed traces stay outside Git.
- The existing demo's unconditional escalation after model prose was replaced with record-only default. An optional local test file requires strict event-bound confirmation; it cannot invoke webhooks or telephone endpoints. Eighteen focused refusal/local-file tests passed.

## Reuse direction

Keep camera acquisition separate from temporal detection and from authorized notification delivery. RTSP/ONVIF/WebRTC cameras and proprietary cloud cameras require different access adapters; a common decoded-video interface does not grant access to every camera.

- [go2rtc](https://github.com/AlexxIT/go2rtc): candidate multi-protocol streaming layer, not a fall detector; not installed here.
- [Blink clip export](https://support.blinkforhome.com/video-clips/how-do-i-share-my-motion-clips): supported manual acquisition used in this run.
- [blinkpy](https://github.com/fronzbot/blinkpy): unofficial authenticated adapter. The saved desktop login failed once; no repeated login loop or camera settings change. Phone login remained functional.
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/): pose inference used here, not a trained fall classifier. Model/source licensing must be respected before product distribution.
- [Nanit breathing-motion mechanism](https://www.nanit.com/pages/nanit-breathing-motion-monitoring): uses a dedicated visible fabric pattern and compatible camera. This cannot be assumed equivalent to generic Blink footage.

## Remaining acceptance gates and actions

1. Evaluate a licensed temporal fall model on staged fall and ordinary-activity recordings; retain misses and nuisance alerts. The current box-transition heuristic is only a candidate baseline.
2. Establish unattended authenticated Blink clip acquisition or another supported stream. Manual export is not continuous monitoring.
3. Bind candidate, source freshness and camera identity to the existing check-in approval/delivery path; measure delivery and recovery. This branch sends no phone alert.
4. Evaluate breathing only against an independent reference measurement with applicable recording conditions; no infant safety claim or life-safety substitution.
5. Face identification requires a separately selected opt-in enrollment and unknown-person testing; no face identity module was added by this branch.

No emergency calls, cloud image uploads, GPU jobs, or persistent new Windows services were run for this evaluation.
