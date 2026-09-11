"""One bounded authenticated clip read. Private output; no arm/disarm or alerts."""
import argparse, asyncio, hashlib, json
from pathlib import Path
from aiohttp import ClientSession, ClientTimeout
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth

async def run(args):
    credentials=Path(args.credentials).expanduser().resolve(strict=True)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    async with ClientSession(timeout=ClientTimeout(total=45)) as session:
        blink=Blink(session=session)
        blink.auth=Auth(json.loads(credentials.read_text()),no_prompt=True,session=session)
        await blink.start()
        cameras=list(blink.cameras.values())
        report={'camera_count':len(cameras),'clips':[]}
        for i,cam in enumerate(cameras):
            data=cam.video_from_cache
            if data:
                dest=out/f'camera-{i+1}.mp4';dest.write_bytes(data)
                report['clips'].append({'camera_alias':f'camera-{i+1}','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'path':dest.name})
        (out/'probe.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
        if not report['clips']: raise RuntimeError('No cached video available; no fall observation made')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--credentials',default='~/.config/blink/credentials.json');p.add_argument('--out',required=True)
    args=p.parse_args()
    try: asyncio.run(asyncio.wait_for(run(args),timeout=90))
    except Exception as e:
        # Never emit provider exception text which may contain account/session material.
        print(json.dumps({'status':'BLOCKED','exception_type':type(e).__name__,'reason':'Authenticated clip read failed or returned no video; no action or detector ran.'}))
        raise SystemExit(2)
