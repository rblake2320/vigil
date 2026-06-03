"""
Complete Blink 2FA auth and snap all cameras.

Usage:
  python blink_auth_finish.py <PIN>     # pass PIN as arg
  python blink_auth_finish.py           # prompts for PIN

Run blink_auth.py first to trigger a new 2FA code to your email/phone.
"""
import asyncio
import json
import pathlib
import sys

from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth, BlinkTwoFARequiredError

CRED_FILE = pathlib.Path.home() / ".config/blink/credentials.json"
SNAP_DIR = pathlib.Path.home() / "blink_snaps"
SNAP_DIR.mkdir(exist_ok=True)

CREDS = {
    "username": "rblake2320@me.com",
    "password": "?Booker78!",
}


async def main():
    pin = sys.argv[1] if len(sys.argv) > 1 else input("PIN: ").strip()

    async with ClientSession() as session:
        blink = Blink(session=session)
        blink.auth = Auth(CREDS, no_prompt=True)
        try:
            await blink.start()
            print("Logged in without 2FA!")
        except BlinkTwoFARequiredError:
            print(f"Submitting PIN...")
            result = await blink.send_2fa_code(pin)
            print(f"2FA result: {result}")

        print(f"Connected! {len(blink.cameras)} cameras:")
        for name in blink.cameras:
            print(f"  - {name}")

        if blink.cameras:
            for name, cam in blink.cameras.items():
                await cam.snap_picture()
            await blink.refresh(force=True)

            for name, cam in blink.cameras.items():
                out = SNAP_DIR / f"{name.replace(' ', '_')}.jpg"
                await cam.image_to_file(str(out))
                size = out.stat().st_size if out.exists() else 0
                print(f"  Saved {name} -> {out} ({size} bytes)")

        await blink.save(str(CRED_FILE))
        print("Credentials saved.")


asyncio.run(main())
