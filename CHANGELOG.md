# Changelog

## 2026-09-11 — person-corroborated posture candidate

- Added a pinned, restricted loader for the AGPL-3.0 licensed
  `melihuzunoglu/human-fall-detection` checkpoint (weights not bundled).
- Added optional learned-posture mode: same-frame human pose, competing posture
  classes, ambiguity refusal, and a sustained observation gate. One live run
  dispatches at most its first candidate and then stops.
- Actual video comparison: four ordinary-activity clips produced zero candidates;
  heldout fall and retained owner fall footage produced candidates. This is a
  small experimental comparison, not medical validation or proof of a live SMS.
- Simple descent-only mode failed real negatives and cannot dispatch alerts.
- Readiness now requires sustained recent analyzed frames; a paused Blink trace
  refuses. Coded Google Messages replies accept keyboard trailing ASCII spaces.

## 2026-09-11 — test reply readback and failed-trial evidence

- Added scoped Google Messages reply readback with fresh nonce, sender/thread
  checks, tamper checks, and 11 focused regressions. Handles UI-visible replies
  that the SMS-only test coordinator missed.
- Recorded two unsuccessful live fall-alert trials and their retained-image
  analysis. No live fall-alert success or clinical reliability claim.
- See `docs/fall-repeat-readback-20260911.md` for exact proof boundaries.
