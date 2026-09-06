# Android observations and checkpoints from Design Beast

Vigil consumes the shared `beast_studio_client.mobile` implementation from
Design Beast SDK 1.1.0. PhoneClaw is an upstream reference only; its repository
is untouched and no source was copied. This lane adds local Android UI and
screenshot evidence without another assistant, cloud service or script engine.

Install the pinned shared SDK in a Vigil virtual environment:

```powershell
python -m pip install -r requirements-mobile.txt
python -m perception.mobile_watch --serial YOUR_SERIAL --package com.android.settings --output mobile-evidence --duration 30
python -m perception.mobile_watch --serial YOUR_SERIAL --package com.android.settings --procedure watcher_procedures/android-settings.json --duration 30
```

The requirements file pins the companion Beast commit and its mobile extra.
Deployments may instead install the wheel built from that commit. ADB and an authorized
device must be supplied separately. Open the allowed app yourself. Nothing in
this observer launches apps, taps, types, calls a model, speaks or publishes.

Each observation is retained by the SDK and emitted as JSON. A procedure uses
unique string step IDs and exact `expected` selectors (text/resource_id/description).
At most one checkpoint advances per fresh snapshot. Missing or ambiguous state
keeps the checkpoint pending. Expiry before all checkpoints yields exit code 1.
Transport/capture errors end with a classified error. Visible checkpoints do not
prove persisted application data or overall task correctness.

## Desktop coach behavior change

The old substring matcher could advance on `MATCH: NO. The saved successfully
message is not visible` and on a missing/empty detect keyword. Both watcher and
signal coach now treat model prose and window-title keywords as observations,
never completion receipts. Legacy procedures require nonempty steps/detect fields.
Their hints still run, but automatic advancement is deliberately contained until
a native desktop postcondition adapter is available. This is not a claim that
the desktop coach has been fully repaired or verified end to end.

The Android observer is the first independently checked state lane. Do not weaken
desktop verification to restore keyword-driven advancement. See
`tests/test_completion_boundaries.py` for the negative-control regressions.

No real Android capture has been proven by the CPU tests. A physical-device
acceptance run must retain the real snapshots and verify an intended checkpoint,
a wrong-package refusal, ambiguous text, and a disconnected device. Screenshots
and UI text are private local artifacts; `mobile-evidence/` is ignored by Git.
