# Changelog

## 2026-09-11 — test reply readback and failed-trial evidence

- Added scoped Google Messages reply readback with fresh nonce, sender/thread
  checks, tamper checks, and 11 focused regressions. Handles UI-visible replies
  that the SMS-only test coordinator missed.
- Recorded two unsuccessful live fall-alert trials and their retained-image
  analysis. No live fall-alert success or clinical reliability claim.
- See `docs/fall-repeat-readback-20260911.md` for exact proof boundaries.
