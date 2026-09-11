"""Scoped Google Messages UI readback, including replies absent from SMS storage.

Observation is not cryptographic sender authentication. Read before opening the
camera, then acknowledge ARMED only after live capture passes its health check.
No sends, device setting changes, or detector activation occur in this module.
"""
import hashlib
import re
import time
import xml.etree.ElementTree as ET
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

PACKAGE = 'com.google.android.apps.messaging'


def phone_digits(value):
    digits = re.sub(r'\D', '', value)
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    if len(digits) != 10:
        raise ValueError('Explicit US test sender required')
    return digits


def observe_reply(xml_bytes, sender, expected_body):
    """Match inbound message text AND attributed sender; outgoing echoes refuse."""
    if not isinstance(xml_bytes, bytes) or len(xml_bytes) > 2_000_000:
        raise ValueError('Invalid bounded UI snapshot')
    if not isinstance(expected_body, str) or not 1 <= len(expected_body) <= 80:
        raise ValueError('Invalid exact reply')
    root = ET.fromstring(xml_bytes)
    expected_sender = phone_digits(sender)
    nodes = list(root.iter('node'))
    # A standalone thread header must identify the same number. A number buried
    # inside message text is insufficient to identify the open conversation.
    headers = [n for n in nodes if n.get('package') == PACKAGE
               and re.fullmatch(r'[+()\d -]+', n.get('text', ''))
               and re.sub(r'\D', '', n.get('text', '')).removeprefix('1') == expected_sender
               and n.get('resource-id') != 'message_text']
    if len(headers) != 1:
        raise ValueError('Wrong or ambiguous conversation header')
    matches = []
    for node in nodes:
        if node.get('package') != PACKAGE or node.get('resource-id') != 'message_text':
            continue
        if node.get('text','').strip(' \t\r\n') != expected_body or node.get('enabled') != 'true':
            continue
        desc = node.get('content-desc', '')
        attribution = re.fullmatch(r'([+()\d -]+) said  (.+)', desc)
        if not attribution or phone_digits(attribution[1]) != expected_sender:
            continue
        if not attribution[2].startswith(expected_body + ' '):
            continue
        bounds = re.fullmatch(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', node.get('bounds', ''))
        if not bounds:
            continue
        x1,y1,x2,y2 = map(int,bounds.groups())
        if x2 <= x1 or y2 <= y1:
            continue
        matches.append(node)
    if len(matches) != 1:
        raise ValueError('Reply absent or ambiguous')
    return {'source': 'google_messages_ui', 'snapshot_sha256': hashlib.sha256(xml_bytes).hexdigest(),
            'body': expected_body, 'sender_digits': expected_sender,
            'observation_only': True, 'transport_type': 'not_inferred_from_ui'}


def confirm_challenge(xml_bytes, sender, nonce, issued_at, now=None):
    """Only a fresh exact nonce reply may satisfy a new local test challenge."""
    now = time.time() if now is None else now
    if not isinstance(nonce,str) or not re.fullmatch(r'[a-f0-9]{12}',nonce):
        raise ValueError('Invalid nonce')
    if not isinstance(issued_at,(int,float)) or not 0 <= now-issued_at <= 120:
        raise ValueError('Expired or future challenge')
    receipt = observe_reply(xml_bytes,sender,'START '+nonce)
    receipt.update(nonce=nonce,issued_at=issued_at,readback_at=now,
                   observation_only=False,authority='owner-authorized bounded test only')
    return receipt


def validate_saved(receipt, xml_bytes, sender, nonce, now=None):
    if receipt.get('nonce') != nonce or receipt.get('observation_only') is not False:
        raise ValueError('Wrong challenge receipt')
    if receipt.get('snapshot_sha256') != hashlib.sha256(xml_bytes).hexdigest():
        raise ValueError('Snapshot changed')
    checked = confirm_challenge(xml_bytes,sender,nonce,receipt['issued_at'],now)
    if checked['sender_digits'] != receipt.get('sender_digits'):
        raise ValueError('Sender changed')
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb',required=True)
    parser.add_argument('--serial',required=True)
    parser.add_argument('--sender',required=True)
    parser.add_argument('--nonce',required=True)
    parser.add_argument('--issued-at',required=True,type=float)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--wait-seconds',type=int,choices=range(0,121),default=0)
    args=parser.parse_args()
    # No automatic navigation, old XML reuse, or camera interruption: caller
    # opens the authorized thread BEFORE starting the camera test.
    command=[args.adb,'-s',args.serial,'shell']
    if args.out.exists():raise ValueError('Receipt output already exists; no replay')
    deadline=time.monotonic()+args.wait_seconds
    while True:
        target='/sdcard/reply-'+uuid.uuid4().hex+'.xml'
        result=subprocess.run(command+['uiautomator','dump',target],capture_output=True,timeout=15)
        if result.returncode or b'UI hierchary dumped to:' not in result.stdout:
            raise RuntimeError('Fresh UI dump failed; no old snapshot used')
        result=subprocess.run(command+['cat',target],capture_output=True,timeout=5)
        if result.returncode:raise RuntimeError('Fresh UI read failed')
        xml_bytes=result.stdout
        try:
            receipt=confirm_challenge(xml_bytes,args.sender,args.nonce,args.issued_at)
            break
        except ValueError as exc:
            if str(exc)!='Reply absent or ambiguous' or time.monotonic()>=deadline:raise
            time.sleep(min(3,max(0,deadline-time.monotonic())))
    args.out.mkdir(exist_ok=False)
    snapshot=args.out/'snapshot.xml'
    with snapshot.open('xb') as f:f.write(xml_bytes);f.flush();os.fsync(f.fileno())
    receipt.update(serial=args.serial,snapshot_path=str(snapshot.resolve()))
    with (args.out/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps({'reply_observed':True,'receipt':str(args.out/'receipt.json'),'detector_activated':False}))


if __name__=='__main__':main()
