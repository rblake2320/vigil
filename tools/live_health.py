"""Require sustained recent display observations before a test-ready message."""
import math


def readiness(rows, now):
    if not isinstance(now,(int,float)) or not math.isfinite(now):
        raise ValueError('Invalid health clock')
    recent=[]
    for row in rows:
        stamp=row.get('host_decode_arrival_epoch')
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp):
            raise ValueError('Invalid capture timestamp')
        if stamp>now:raise ValueError('Capture timestamp in future')
        if now-stamp<=3:recent.append(row)
    if not recent:raise ValueError('No recent capture')
    bad=[r for r in recent if r.get('stale_reason') not in (None,'skip_duplicate_frame','skip_equal_arrival_timestamp')]
    if bad:raise ValueError('Capture stale or buffering in readiness window')
    usable=[r for r in recent if r.get('stale_reason') is None and r.get('status')!='analysis_sample_skipped']
    if len(usable)<3 or usable[-1]['host_decode_arrival_epoch']-usable[0]['host_decode_arrival_epoch']<2:
        raise ValueError('Need two seconds of usable observations')
    if now-usable[-1]['host_decode_arrival_epoch']>.5:
        raise ValueError('Latest usable observation too old')
    if len({r.get('frame_sha256') for r in usable})<3:
        raise ValueError('Insufficient changing frames')
    return {'recent_usable':len(usable),'span_seconds':usable[-1]['host_decode_arrival_epoch']-usable[0]['host_decode_arrival_epoch'],
            'display_health_only':True,'camera_timestamp_verified':False}
