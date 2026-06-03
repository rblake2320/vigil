# Blink Camera Setup — How We Got Here

## Overview

5 Blink cameras connected to Vigil, accessible from both Spark-2 (primary) and
Windows 5090 (secondary). This doc covers the full auth flow, credential layout,
camera inventory, and the local-model vision pipeline built on top.

---

## Camera Inventory

| Name | What it covers |
|------|---------------|
| `My room` | Bedroom |
| `Kitchen ` | Kitchen area |
| `Living room` | Main living space |
| `GNT1-9001-3364-8H30` | Secondary room (person detected frequently) |
| `Door` | Entry/front door |

---

## Auth Flow (blinkpy 0.25.5)

Blink uses **Amazon OAuth v2** with a 4-hour access token and 30-day refresh token.
New devices require email/phone 2FA.

### First-time auth on a new machine

```python
# Step 1 — trigger 2FA code (sends to email/phone)
blink.auth = Auth({"username": "...", "password": "..."}, no_prompt=True)
await blink.start()   # raises BlinkTwoFARequiredError

# Step 2 — complete with PIN
await blink.send_2fa_code("123456")   # handles everything internally

# Step 3 — save credentials (do this every session to refresh tokens)
await blink.save("~/.config/blink/credentials.json")
```

### Subsequent runs (token still valid)

```python
creds = json.loads(Path("~/.config/blink/credentials.json").read_text())
blink.auth = Auth(creds, no_prompt=True)
await blink.start()   # no 2FA needed
```

### Token expiry handling

- **Access token**: 4 hours. After expiry, `blink.start()` re-triggers OAuth login
  which throws `BlinkTwoFARequiredError` again. Handle it the same way.
- **Refresh token**: 30 days. Within that window `blinkpy` auto-refreshes silently.
- **Always call `blink.save()`** after a session to persist the refreshed token.

### Key blinkpy API (v0.25.5)

```python
blink.send_2fa_code(pin)          # catch BlinkTwoFARequiredError then call this
blink.auth.complete_2fa_login(pin) # lower-level — used internally by send_2fa_code
blink.refresh(force=True)          # re-poll all cameras
cam.snap_picture()                 # trigger new snapshot on camera
cam.image_to_file(path)            # download snapshot to disk
blink.save(path)                   # persist credentials JSON
```

### Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `BlinkTwoFARequiredError` | Token expired or new device | Call `blink.send_2fa_code(pin)` after catching |
| `AttributeError: 'Auth' has no 'send_auth_key'` | Wrong method name | Use `blink.send_2fa_code()` not `auth.send_auth_key()` |
| `AttributeError: 'Blink' has no 'setup_login'` | Wrong method name | Use `blink.send_2fa_code()` — it handles full post-2FA flow |
| `NoneType has no 'base_url'` | 2FA failed silently (bad/expired PIN) | Request a fresh code and retry |
| `2FA verification failed` | PIN already used or expired | Call `blink_auth.py` again to get a new PIN |

---

## Credential Files

| Machine | Path |
|---------|------|
| Windows 5090 | `~/.config/blink/credentials.json` |
| Spark-2 | `~/.config/blink/credentials.json` |

Account: `rblake2320@me.com` | Region: `u025.immedia-semi.com`

Each machine gets its own `hardware_id` and `client_id` from Blink — they co-exist
fine. Saving credentials from one machine does NOT invalidate the other's session.

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `blink_auth.py` | Triggers fresh 2FA code (run when token expired) |
| `blink_auth_finish.py` | Completes auth with PIN, snaps all cameras |
| `blink_setup.py` | Original interactive setup (prompts for email/pass + PIN) |
| `perception/blink_source.py` | Production Blink source used by Vigil |

---

## Snap All Cameras (quick test)

```bash
python blink_auth_finish.py   # after editing PIN at top of file
# or
python - <<'EOF'
import asyncio, json, pathlib
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth, BlinkTwoFARequiredError

async def snap():
    creds = json.loads((pathlib.Path.home() / ".config/blink/credentials.json").read_text())
    async with ClientSession() as s:
        b = Blink(session=s)
        b.auth = Auth(creds, no_prompt=True)
        try:
            await b.start()
        except BlinkTwoFARequiredError:
            await b.send_2fa_code(input("PIN: "))
        for name, cam in b.cameras.items():
            await cam.snap_picture()
        await b.refresh(force=True)
        for name, cam in b.cameras.items():
            out = pathlib.Path(f"blink_snaps/{name.replace(' ','_')}.jpg")
            await cam.image_to_file(str(out))
            print(f"{name}: {out.stat().st_size} bytes")
        await b.save(str(pathlib.Path.home() / ".config/blink/credentials.json"))

asyncio.run(snap())
EOF
```
