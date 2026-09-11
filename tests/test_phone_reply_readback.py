import xml.etree.ElementTree as ET
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from phone_reply_readback import observe_reply, confirm_challenge, validate_saved, PACKAGE

SENDER='+12025550123'
NONCE='abcdef123456'

def snapshot(body='START '+NONCE, attribution='(202) 555-0123', header='(202) 555-0123', duplicate=False):
    root=ET.Element('hierarchy')
    ET.SubElement(root,'node',{'package':PACKAGE,'resource-id':'thread_title','text':header})
    attrs={'package':PACKAGE,'resource-id':'message_text','text':body,'content-desc':attribution+' said  '+body+' 4:55 PM .','enabled':'true','bounds':'[10,10][200,60]'}
    ET.SubElement(root,'node',attrs)
    if duplicate:ET.SubElement(root,'node',attrs)
    return ET.tostring(root)

def test_inbound_ui_reply_without_sms_store():
    assert confirm_challenge(snapshot(),SENDER,NONCE,100,110)['source']=='google_messages_ui'

@pytest.mark.parametrize('kwargs',[{'attribution':'You'},{'attribution':'(202) 555-0999'}, {'header':'(202) 555-0999'}, {'duplicate':True}, {'body':'START'}, {'body':'START badbadbadbad'}])
def test_wrong_echo_stale_body_or_ambiguity_refuses(kwargs):
    with pytest.raises(ValueError):confirm_challenge(snapshot(**kwargs),SENDER,NONCE,100,110)

@pytest.mark.parametrize('now',[99,221])
def test_clock_window_refuses(now):
    with pytest.raises(ValueError):confirm_challenge(snapshot(),SENDER,NONCE,100,now)

def test_legacy_start_is_observation_only():
    assert observe_reply(snapshot(body='START'),SENDER,'START')['observation_only'] is True

def test_mobile_keyboard_trailing_ascii_space():
    assert confirm_challenge(snapshot(body='START '+NONCE+' '),SENDER,NONCE,100,110)['nonce']==NONCE

def test_saved_snapshot_reverified_and_tamper_refused():
    raw=snapshot();receipt=confirm_challenge(raw,SENDER,NONCE,100,110)
    assert validate_saved(receipt,raw,SENDER,NONCE,111)['nonce']==NONCE
    with pytest.raises(ValueError):validate_saved(receipt,raw+b' ',SENDER,NONCE,111)
    with pytest.raises(ValueError):validate_saved(receipt,raw,SENDER,'badbadbadbad',111)
    with pytest.raises(ValueError):validate_saved(receipt,raw,SENDER,NONCE,221)
