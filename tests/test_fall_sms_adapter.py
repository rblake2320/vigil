import importlib.util
import json
import sqlite3
from pathlib import Path
import pytest
from care_notification_engine.alert_store import AlertStore
from care_notification_engine.authorization import ConsentStore
from care_notification_engine.transport import TransportAcceptance
spec=importlib.util.spec_from_file_location("bridge",Path(__file__).resolve().parents[1]/"tools/fall_sms_adapter.py")
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)

class Transport:
    expected_destination="+19194327655"
    def __init__(self,status="accepted"):
        self.calls=[]; self.status=status
    def send_with_authorization(self,*args):
        self.calls.append(args)
        if self.status=="throw": raise TimeoutError()
        return TransportAcceptance(accepted=self.status=="accepted",status=self.status,detail="stub only")

def setup(tmp_path,status="accepted",grant=True):
    path=tmp_path/"consent.json"; store=ConsentStore(path)
    store.create_contact("owner");store.confirm_contact("owner","device_owner");store.confirm_contact("owner","contact")
    session=store.request_support("owner","view",1000)
    if grant: store.grant_session(session,"device_owner",2000,1000)
    transport=Transport(status)
    kwargs=dict(journal=tmp_path/"journal.db",alert_store=AlertStore(str(tmp_path/"alerts.db")),transport=transport,session_id=session,consent_path=path,expected_source="camera-test",expected_destination=transport.expected_destination,now=1500)
    event=dict(event_id="event1",event="possible_fall",observed_at=1499,source_id="camera-test",mode="test",origin="transport_test")
    return event,kwargs,transport

@pytest.mark.parametrize("change",[{"event":"fire"},{"mode":"production"},{"origin":"offline_clip"},{"observed_at":1000},{"observed_at":True},{"source_id":"other"},{"confirmed":True}])
def test_invalid(tmp_path,change):
    event,kwargs,t=setup(tmp_path);event.update(change)
    with pytest.raises(ValueError): bridge.dispatch(event,**kwargs)
    assert not t.calls

def test_real_consent_denial(tmp_path):
    event,kwargs,t=setup(tmp_path,grant=False)
    assert bridge.dispatch(event,**kwargs)["status"]=="AUTH_DENIED"
    assert not t.calls

@pytest.mark.parametrize("status",["accepted","unknown","throw"])
def test_durable_duplicate_no_replay(tmp_path,status):
    event,kwargs,t=setup(tmp_path,status)
    result=bridge.dispatch(event,**kwargs)
    assert result["status"]==("ACCEPTED" if status=="accepted" else "UNKNOWN")
    assert result["live_ready"] is False
    assert bridge.dispatch(event,**kwargs)["status"]=="duplicate_no_replay"
    event["event_id"]="second"
    assert bridge.dispatch(event,**kwargs)["status"]=="attempt_cap_refused"
    assert len(t.calls)==1 and isinstance(t.calls[0][2],dict)
    assert "TRANSPORT TEST ONLY" in t.calls[0][1]

def test_crash_intent_blocks(tmp_path):
    event,kwargs,t=setup(tmp_path)
    bridge.dispatch(event,**kwargs)
    with sqlite3.connect(kwargs["journal"]) as db: db.execute("UPDATE fall_attempt SET state='INTENT'")
    result=bridge.dispatch(event,**kwargs)
    assert result["prior_state"]=="INTENT" and len(t.calls)==1

def test_actual_destination_guard(tmp_path):
    import base64
    class Shell:
        def shell(self,*args): raise AssertionError("native entry")
    guard=bridge.DestinationGuardShell(Shell(),"+19194327655")
    raw=base64.b64encode(json.dumps({"destination":"+19999999999"}).encode()).decode()
    with pytest.raises(ValueError): guard.shell("am","broadcast","--es","envelope_b64",raw)
