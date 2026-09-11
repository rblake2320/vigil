import math
import pytest
from detectors.pose_descent import PoseDescentDetector,PoseObservation

def pose(s=.15,h=.4,identity='one',x=.5):return PoseObservation(identity,(x,s),(.5,h),(.9,)*4)
def run(samples):
 d=PoseDescentDetector();out=[]
 for t,p,kw in samples:out.extend(d.update(t,p,**kw).candidates)
 return out

def baseline():return [(i/10,pose(),{}) for i in range(11)]
def low():return [(1.1+i/10,pose(.4,.65),{}) for i in range(15)]
def test_rapid_descent_then_seated_torso_and_one_candidate():
 c=run(baseline()+low());assert len(c)==1;assert c[0].low_seconds>=1-1e-9
@pytest.mark.parametrize('kind',['slow_sit','bend','walk'])
def test_nonfall_controls(kind):
 samples=baseline()
 for i in range(1,61):
  amount=i*.005
  p=pose(.15+amount,.4+amount) if kind=='slow_sit' else pose(.15+min(amount,.25),.4) if kind=='bend' else pose(x=.5+math.sin(i)*.03)
  samples.append((1+i/10,p,{}))
 assert run(samples)==[]
@pytest.mark.parametrize('fault',['missing','ambiguous','invalid_frame','new_identity','low_confidence','timestamp_gap'])
def test_interrupted_evidence_never_inherits_baseline(fault):
 samples=baseline();kw={};p=pose(.4,.65);t=1.1
 if fault=='missing':p=None
 if fault=='ambiguous':kw={'identity_ambiguous':True}
 if fault=='invalid_frame':kw={'frame_valid':False} # explicit invalidation, NOT real camera-motion detection
 if fault=='new_identity':p=pose(.4,.65,'other')
 if fault=='low_confidence':p=PoseObservation('one',(.5,.4),(.5,.65),(.1,)*4)
 if fault=='timestamp_gap':t=1.6
 samples.append((t,p,kw));samples += [(t+.1+i/10,pose(.4,.65,'other' if fault=='new_identity' else 'one'),{}) for i in range(15)]
 assert run(samples)==[]
def test_low_position_broken_resets_hold():
 samples=baseline()+low()[:4]+[(1.5,pose(),{})]+[(1.6+i/10,pose(.4,.65),{}) for i in range(15)]
 assert run(samples)==[]
def test_invalid_confidence_not_coerced():
 d=PoseDescentDetector();r=d.update(0,PoseObservation('one',(.5,.15),(.5,.4),(True,.9,.9,.9)))
 assert r.status=='insufficient_observation'
