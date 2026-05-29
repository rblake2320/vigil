"""
alerts/twilio_alert.py — SMS + Voice call alerts via Twilio
This is the final step that makes Vision-to-Action REAL.

Status: WIRED, NOT YET CONNECTED (no Twilio credentials configured)
To activate: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER in env.

The 911 escalation cascade (from vigil_context.py) calls these functions.
Replace "WOULD FIRE" with actual Twilio API calls below.

Twilio free trial: $15 credit, enough for ~500 SMS or ~200 calls.
E911 requires address registration: twilio.com/docs/voice/api/dialing-911
"""
import os
import logging
import time

log = logging.getLogger(__name__)

_SID   = os.environ.get("TWILIO_ACCOUNT_SID")
_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
_FROM  = os.environ.get("TWILIO_FROM_NUMBER")
_READY = bool(_SID and _TOKEN and _FROM)


def send_sms(to: str, body: str) -> bool:
    """Send an SMS alert. Returns True on success."""
    if not _READY:
        log.warning(f"[Twilio] SMS NOT SENT (no credentials). Would send to {to}: {body[:80]}")
        return False
    try:
        from twilio.rest import Client
        client = Client(_SID, _TOKEN)
        msg = client.messages.create(body=body, from_=_FROM, to=to)
        log.info(f"[Twilio] SMS sent to {to}: SID {msg.sid}")
        return True
    except Exception as e:
        log.error(f"[Twilio] SMS failed: {e}")
        return False


def make_call(to: str, script: str) -> bool:
    """
    Make an automated voice call reading script via TTS.
    Returns True on success.
    """
    if not _READY:
        log.warning(f"[Twilio] CALL NOT MADE (no credentials). Would call {to}.")
        log.warning(f"[Twilio] Script: {script[:120]}")
        return False
    try:
        from twilio.rest import Client
        twiml = f"<Response><Say voice='alice'>{script}</Say></Response>"
        # Twilio requires a URL — host TwiML via ngrok or Twilio Studio
        # For demo: use TwiML Bins (free in Twilio console)
        # twiml_bin_url = "https://handler.twilio.com/twiml/YOUR_BIN_ID"
        client = Client(_SID, _TOKEN)
        call = client.calls.create(
            twiml=twiml,
            from_=_FROM,
            to=to,
        )
        log.info(f"[Twilio] Call initiated to {to}: SID {call.sid}")
        return True
    except Exception as e:
        log.error(f"[Twilio] Call failed: {e}")
        return False


def escalate(
    contacts: list[str],
    sms_body: str,
    call_script: str,
    call_911_script: str | None = None,
) -> dict:
    """
    Full escalation cascade:
      0s  — SMS to all contacts
      30s — Voice call to primary contact
      60s — Voice call to 911 (if call_911_script provided + E911 registered)

    Returns dict of actions taken.
    """
    results = {"sms": [], "calls": [], "911": None}

    log.info("[Vigil] Escalation cascade starting")

    # Immediate SMS
    for contact in contacts:
        ok = send_sms(contact, sms_body)
        results["sms"].append({"to": contact, "sent": ok, "at": time.time()})

    # 30s: voice call
    time.sleep(30)
    if contacts:
        ok = make_call(contacts[0], call_script)
        results["calls"].append({"to": contacts[0], "connected": ok, "at": time.time()})

    # 60s: 911
    if call_911_script:
        time.sleep(30)
        e911_number = os.environ.get("TWILIO_E911_NUMBER", "+19999999999")
        ok = make_call(e911_number, call_911_script)
        results["911"] = {"connected": ok, "at": time.time()}
        log.info(f"[Vigil] 911 call {'placed' if ok else 'FAILED'}")

    return results
