"""One-shot test alert bridge. Existing care engine owns authorization and SMS.
No command here arms a camera or proves delivery/reply. No automatic retries.
"""
import argparse
import hashlib
import json
import math
import sqlite3
import time


def validate_single_segment(message):
    """Conservative GSM-7 ASCII subset; extensions cost two septets.

    Non-ASCII and unsupported controls refuse rather than guessing UCS-2 size.
    This adapter uses single-part sendTextMessage, never multipart fallback.
    """
    if not isinstance(message, str) or not message:
        raise ValueError("empty SMS")
    extensions = set("^{}" + chr(92) + "[~]|")
    total = 0
    for char in message:
        if char not in "\r\n" and not (32 <= ord(char) <= 126 and char != "`"):
            raise ValueError("unsupported SMS encoding")
        total += 2 if char in extensions else 1
    if total > 160:
        raise ValueError("single-segment SMS exceeds 160 septets")
    return total


def dispatch(event, *, journal, alert_store, transport, session_id, consent_path,
             expected_source, expected_destination, now=None):
    from care_notification_engine.delivery import create_and_send_authorized_alert
    now = time.time() if now is None else now
    required = {"event_id", "event", "observed_at", "source_id", "mode", "origin"}
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("invalid event fields")
    if event["event"] != "possible_fall" or event["mode"] != "test":
        raise ValueError("only possible_fall test alerts allowed")
    if event["origin"] not in {"live_capture", "transport_test"}:
        raise ValueError("offline clips cannot dispatch")
    if not isinstance(event["event_id"], str) or not 1 <= len(event["event_id"]) <= 128:
        raise ValueError("invalid event id")
    if event["source_id"] != expected_source or not expected_source:
        raise ValueError("wrong source")
    observed = event["observed_at"]
    if type(observed) not in (int, float) or not math.isfinite(observed) or not 0 <= now-observed <= 30:
        raise ValueError("event not fresh")
    if not expected_destination or not hasattr(transport, "send_with_authorization"):
        raise ValueError("governed transport required")
    # Wrapper must enforce destination at the actual send boundary.
    if transport.expected_destination != expected_destination:
        raise ValueError("destination binding mismatch")
    message = ("TRANSPORT TEST ONLY. Reply RECEIVED." if event["origin"] == "transport_test"
               else "LIVE CAMERA TEST: possible fall, not confirmed. Please check and reply RECEIVED.")
    validate_single_segment(message) # before journal/INTENT or transport
    canonical = json.dumps({"event":event,"session":session_id,"destination":expected_destination},sort_keys=True,separators=(",",":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    alert_id = "vigil-test-" + hashlib.sha256((session_id+":"+event["event_id"]).encode()).hexdigest()
    with sqlite3.connect(journal, timeout=10) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS fall_attempt (slot INTEGER PRIMARY KEY CHECK(slot=1), digest TEXT NOT NULL, alert_id TEXT NOT NULL, state TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        old = db.execute("SELECT digest,alert_id,state FROM fall_attempt WHERE slot=1").fetchone()
        if old:
            return {"status":"duplicate_no_replay" if old[0]==digest else "attempt_cap_refused", "alert_id":old[1], "prior_state":old[2]}
        db.execute("INSERT INTO fall_attempt VALUES(1,?,?, 'INTENT')",(digest,alert_id))
        db.commit() # before any transport side effect
        try:
            result = create_and_send_authorized_alert(alert_store, transport, message=message, session_id=session_id, alert_id=alert_id, consent_state_path=consent_path, now=now)
            state = "AUTH_DENIED" if not result.authorization.ok else "UNKNOWN"
            if result.transport is not None and result.transport.status in {"accepted","failed","unknown"}:
                state = result.transport.status.upper()
            detail = {"status":state,"alert_id":alert_id,"authorization_reason":result.authorization.reason,"origin":event["origin"],"live_ready":False}
        except Exception:
            state = "UNKNOWN"
            detail = {"status":state,"alert_id":alert_id,"live_ready":False}
        db.execute("UPDATE fall_attempt SET state=? WHERE slot=1",(state,))
        db.commit()
        return detail


class DestinationBoundTransport:
    def __init__(self, inner, expected_destination):
        self.inner = inner
        self.expected_destination = expected_destination
        self.inner._shell = DestinationGuardShell(inner._shell, expected_destination)

    def send_with_authorization(self, alert_id, message, authorization_receipt):
        from care_notification_engine.contact_directory import resolve_destination
        from care_notification_engine.transport import TransportAcceptance
        validate_single_segment(message)
        destination = resolve_destination(self.inner._contact_id, directory_path=self.inner._directory_path)
        if destination != self.expected_destination:
            return TransportAcceptance(accepted=False,status="failed",detail="destination_mismatch")
        return self.inner.send_with_authorization(alert_id,message,authorization_receipt)


class DestinationGuardShell:
    """Validate actual outgoing envelope after directory resolution, before ADB."""
    def __init__(self, shell, destination):
        self.shell_delegate, self.destination = shell, destination

    def shell(self, *args):
        import base64
        index = args.index("envelope_b64")
        envelope = json.loads(base64.b64decode(args[index+1], validate=True))
        if envelope.get("destination") != self.destination:
            raise ValueError("outgoing envelope destination mismatch")
        return self.shell_delegate.shell(*args)

    def exec_out(self, *args):
        return self.shell_delegate.exec_out(*args)


def main():
    from care_notification_engine.alert_store import AlertStore
    from care_notification_engine.transport_sms_companion import EnvConfiguredSmsCompanionTransportAdapter
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("event-file","journal","alerts-db","consent-path","session-id","source-id","destination"):
        parser.add_argument("--"+flag,required=True)
    args=parser.parse_args()
    from pathlib import Path
    event=json.loads(Path(args.event_file).read_text(encoding="utf-8"))
    adapter=DestinationBoundTransport(EnvConfiguredSmsCompanionTransportAdapter(),args.destination)
    result=dispatch(event,journal=args.journal,alert_store=AlertStore(args.alerts_db),transport=adapter,session_id=args.session_id,consent_path=Path(args.consent_path),expected_source=args.source_id,expected_destination=args.destination)
    print(json.dumps(result))

if __name__=="__main__":
    main()
