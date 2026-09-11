import pytest
from detectors.learned_fall import *
B=(.1,.1,.5,.8)
def p(box=B):return HumanPose(box,(.9,)*4)
def c(label='fallen',score=.8,box=B):return PostureBox(box,score,label)
def test_fallen_person_holds_once():
 d=LearnedFallDetector();assert not d.update(0,[p()],[c()]).candidates;assert not d.update(.2,[p()],[c()]).candidates;assert len(d.update(.4,[p()],[c()]).candidates)==1;assert not d.update(.6,[p()],[c()]).candidates
@pytest.mark.parametrize('label',['sitting','standing'])
def test_nonfallen_winner_blocks(label):
 d=LearnedFallDetector()
 for t in [0,.2,.4,.6]:assert not d.update(t,[p()],[c(),c(label,.95)]).candidates
@pytest.mark.parametrize('case',['wall','overlap_people','score_tie','missing','invalid','gap','other_person','low_pose'])
def test_interruptions_clear_hold(case):
 d=LearnedFallDetector();d.update(0,[p()],[c()]);d.update(.2,[p()],[c()]);pp=[p()];cc=[c()];kw={};t=.3
 if case=='wall':cc=[c(box=(.7,.1,.8,.3))]
 if case=='overlap_people':pp=[p(),p((.12,.12,.52,.82))]
 if case=='score_tie':cc=[c(),c('sitting',.75)]
 if case=='missing':pp=[]
 if case=='invalid':kw={'frame_valid':False}
 if case=='gap':t=.8
 if case=='other_person':pp=[p((.6,.1,.9,.8))];cc=[c(box=(.6,.1,.9,.8))]
 if case=='low_pose':pp=[HumanPose(B,(.2,)*4)]
 assert not d.update(t,pp,cc,**kw).candidates
 assert not d.update(t+.1,[p()],[c()]).candidates
 assert not d.update(t+.3,[p()],[c()]).candidates

def test_bad_input_refuses():assert LearnedFallDetector().update(0,[p()],[c(score=True)]).status=='insufficient_observation'
