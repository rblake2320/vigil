"""
Step 1: Run this to trigger a new 2FA code to your email/phone.
Step 2: Run blink_auth_finish.py with the new PIN.
"""
import asyncio, json, pathlib
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth, BlinkTwoFARequiredError

CRED_FILE = pathlib.Path.home() / ".config/blink/credentials.json"

async def main():
    async with ClientSession() as session:
        blink = Blink(session=session)
        blink.auth = Auth({
            "username": "rblake2320@me.com",
            "password": "?Booker78!"
        }, no_prompt=True)
        try:
            await blink.start()
            print("Logged in without 2FA!")
        except BlinkTwoFARequiredError:
            print("2FA code sent to your email/phone.")
            print("Now run: python blink_auth_finish.py <PIN>")
        finally:
            # Save partial auth state so finish script can reuse it
            state = blink.auth.login_attributes if blink.auth else {}
            pathlib.Path(CRED_FILE.parent / "auth_state.json").write_text(json.dumps(state))

asyncio.run(main())
