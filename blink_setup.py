#!/usr/bin/env python3
"""
Blink camera setup + auth. Run once interactively to authenticate and save
credentials to ~/.config/blink/credentials.json for future use.
"""
import asyncio
import json
import getpass
import pathlib
import sys
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth, BlinkTwoFARequiredError

def tty_input(prompt):
    with open("/dev/tty", "r") as tty:
        sys.stderr.write(prompt)
        sys.stderr.flush()
        return tty.readline().rstrip("\n")

def tty_getpass(prompt):
    with open("/dev/tty", "w") as tty_out:
        return getpass.getpass(prompt, stream=tty_out)

CRED_FILE = pathlib.Path.home() / ".config/blink/credentials.json"

async def main():
    CRED_FILE.parent.mkdir(parents=True, exist_ok=True)

    session = ClientSession()
    try:
        blink = Blink(session=session)

        if CRED_FILE.exists():
            print(f"Loading saved credentials from {CRED_FILE}")
            creds = json.loads(CRED_FILE.read_text())
            blink.auth = Auth(creds, no_prompt=True)
        else:
            email = tty_input("Blink/Amazon email: ")
            password = tty_getpass("Password (hidden): ")
            blink.auth = Auth({"username": email, "password": password}, no_prompt=True)

        try:
            await blink.start()
        except BlinkTwoFARequiredError:
            pin = tty_input("2FA PIN (check your email/phone): ").strip()
            await blink.send_2fa_code(pin)

        # Save credentials for future use
        await blink.save(str(CRED_FILE))
        if CRED_FILE.exists():
            CRED_FILE.chmod(0o600)
        print(f"Credentials saved to {CRED_FILE}")

        print(f"\nFound {len(blink.cameras)} camera(s):")
        for name, cam in blink.cameras.items():
            print(f"  - {name}")

        # Grab a snapshot from each camera
        snap_dir = pathlib.Path.home() / "ai-business/vigil/blink_snaps"
        snap_dir.mkdir(parents=True, exist_ok=True)

        for name, cam in blink.cameras.items():
            print(f"Snapping {name}...")
            await cam.snap_picture()

        await blink.refresh(force=True)

        for name, cam in blink.cameras.items():
            out = snap_dir / f"{name.replace(' ', '_')}.jpg"
            await cam.image_to_file(str(out))
            print(f"  Saved: {out}")

        print("\nBlink setup complete.")
    finally:
        await session.close()

if __name__ == "__main__":
    asyncio.run(main())
