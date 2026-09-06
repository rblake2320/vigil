# Shared mobile observer and coaching containment

Base `e33834a485e8cf2bd6df3fe167497cac064d8586`; Beast companion branch
`codex/mobile-evidence-20260906`. No PhoneClaw files or upstream repository changed.

Observed defect: `MATCH: NO. The saved successfully message is not visible.`
advanced an actual `Procedure` and `Watcher._handle_coach` to completion. A missing
detect keyword also matched. Root cause: substring matching before negative verdict
handling, with empty-string default. The signal coach used the same helper and
also treated window/process title matches as success. No relevant conventional
regression suite was present in the inspected base tree.

Both automatic advancement paths are now contained: prose and titles are contextual
observations, not proof. Empty/malformed procedures reject. This deliberately leaves
desktop auto-advancement unavailable until a native result adapter exists; it is
not presented as a fully fixed desktop coaching workflow.

New `perception.mobile_watch` uses the single Beast SDK to capture an explicitly
selected Android device/package and check exact visible selectors. One checkpoint
can advance per fresh snapshot. It never executes phone actions. SDK authority,
artifact hash, freshness and target checks are reused, not copied.

Executed: nine Vigil regression tests passed in 0.35 seconds, CLI help succeeded,
and the actual observer imported the built SDK from an isolated installation.
Missing-ADB invocation returned a classified error. Independent reviewer reran nine
Vigil tests and reported no remaining code blockers after companion fixes.

No physical device, camera, speech output, model, emergency action or remote service
was exercised. Android happy-path and actual desktop end-to-end proof remain absent.
See `docs/MOBILE-BEAST.md` for owner verification and integration instructions.
