import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from live_health import readiness

def rows():return [{'host_decode_arrival_epoch':t,'stale_reason':None,'frame_sha256':str(t)} for t in (10,10.5,11,11.5,12)]
def test_sustained_display():assert readiness(rows(),12.1)['display_health_only']
def test_single_initial_gray_frame_refused():
    with pytest.raises(ValueError):readiness(rows()[:1],10.1)
def test_freeze_in_window_refused():
    r=rows();r[2]['stale_reason']='identical_display_frame'
    with pytest.raises(ValueError):readiness(r,12.1)
def test_only_duplicate_frames_refused():
    r=rows()
    for x in r:x['frame_sha256']='same'
    with pytest.raises(ValueError):readiness(r,12.1)
def test_old_or_future_refused():
    for now in (9,20):
        with pytest.raises(ValueError):readiness(rows(),now)

def test_unanalysed_frames_do_not_prove_readiness():
    r=rows()
    for x in r:x['status']='analysis_sample_skipped'
    with pytest.raises(ValueError):readiness(r,12.1)
