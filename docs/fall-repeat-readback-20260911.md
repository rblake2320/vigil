# Repeat trial: live alert not demonstrated

Two finite physical test captures ended normally. Neither armed the detector or
sent a fall alert. The first captured real descent and a seated-on-floor recovery;
the second predominantly captured a paused/buffering Blink view.

The owner did reply START. An inbound START was directly observed in the correct
Google Messages conversation while the SMS content provider returned no new rows.
The SMS-only coordinator missed it. An RCS compose indicator was present; the new
reader deliberately reports UI observation, not an inferred transport or a
cryptographic sender identity. Status and result updates were subsequently sent
to the owner through the governed SMS adapter, separately from fall alerts.

## Permanent coordination change

`tools/phone_reply_readback.py` reads a fresh, uniquely named UI dump before the
camera starts. It requires the intended thread header, inbound sender attribution,
exact `START <nonce>` text, an unambiguous visible message, and a 120-second
challenge window. Outgoing echoes, wrong threads/senders, duplicate matches,
unqualified old START messages, changed evidence, and expired challenges refuse.
It performs no sends or detector activation. The private test coordinator accepts
this receipt before opening the camera and sends an explicit ARMED notice after
live health succeeds, before activation. This revised composed path still needs
a fresh on-device challenge test; it was not claimed tested by parser coverage.

Validation: 11 focused parser/gate tests pass; one real retained Google Messages
snapshot produced an inbound START observation (not fresh-challenge authority).
Private controller changes passed Python compilation only. Household snapshots,
messages, phone numbers, credentials, and keypoints remain outside this repository.

## Detector failure remains

An independent offline replay using actual confidence-filtered recorded tracks
produced zero candidates for both trials. The first reached only 0.224 seconds of
horizontal credit, then a square posture box and overlapping detections reset it.
Further bounded CPU inference of retained descent frames measured a substantial
shoulder/hip descent followed by upright sitting on the floor. Persistent
horizontal posture is therefore the wrong condition for this observed motion.
Duplicate-box suppression was not applied: boxes alone cannot safely distinguish
same-person detections from two overlapping people. Thresholds were not lowered
to turn the failed example into a pass.

Next detector work must evaluate descent followed by a low body position against
the saved images and sitting/kneeling/bending/camera-motion negatives. No additional
owner fall is needed to investigate the existing failure. This is not operational
fall monitoring, medical validation, or successful end-to-end alert delivery.

Private trace SHA-256s:

- First: `e45712b4df222ccd4a88bab224a63c258b2a85ed2c5dd5ca139e4229a0f2de47`
- Second: `e1d4b53427f94b5edc414014bf51a4ae91aa7e4f8bbdde6d193a6d047853c157`
- Scoped message UI: `11cf22a7809b448c4d1dbb22b21cc8acbcd09f566169ec645fdd0b83cd74815a`
