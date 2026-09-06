"""Bounded Android observation/coaching using Design Beast's shared SDK.

No actions, provider calls, credential loading, device selection, or publishing.
Install ../design-beast/sdk/python first. All snapshots remain in --output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def load_procedure(path: str | None) -> list[dict]:
    if not path:
        return []
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("mobile procedure must be an object")
    steps = document.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("mobile procedure requires nonempty steps")
    ids = set()
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("id"), str) or not step["id"]:
            raise ValueError("step requires a nonempty string id")
        if step["id"] in ids:
            raise ValueError("duplicate step id")
        ids.add(step["id"])
        expected = step.get("expected")
        if (not isinstance(expected, dict) or not expected or
                set(expected) - {"text", "resource_id", "description"} or
                any(not isinstance(v, str) or not v.strip() for v in expected.values())):
            raise ValueError("step expected must be an exact nonempty UI selector")
    return steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--output", default="mobile-evidence")
    parser.add_argument("--procedure")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--interval", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.duration <= 3600 or not 1 <= args.interval <= 60:
            raise ValueError("duration must be 1..3600 and interval 1..60 seconds")
        steps = load_procedure(args.procedure)
        from beast_studio_client.mobile import AndroidDevice, MobileError, check_observation
        device = AndroidDevice(args.serial, args.package, adb=args.adb)
        deadline, index = time.monotonic() + args.duration, 0
        while time.monotonic() < deadline:
            path = device.observe(args.output)
            event = {"schema": "vigil.mobile-observation/v1", "observation": str(path),
                     "serial": args.serial, "package": args.package, "step_observed": False}
            if steps and index < len(steps):
                step = steps[index]
                event["step_id"] = step["id"]
                try:
                    event["check"] = check_observation(path, serial=args.serial,
                        package=args.package, selector=step["expected"])
                    event["step_observed"] = True
                    index += 1
                except MobileError as exc:
                    event["reason"] = str(exc)
            print(json.dumps(event), flush=True)
            if steps and index == len(steps):
                print(json.dumps({"visible_checkpoints_observed": True,
                    "boundary": "Visible UI checkpoints only; persistence and overall task correctness not verified."}))
                return 0
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
        return 1 if steps and index < len(steps) else 0
    except ImportError:
        print(json.dumps({"error": "Install the reviewed beast-studio-client SDK with mobile support."}))
        return 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
