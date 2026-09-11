"""Experimental learned fallen-posture cue, not a witnessed fall or medical alert.

Same-frame confidence-qualified human pose corroborates a classifier box. A person
already down can qualify. Identity is only spatial continuity, not authentication.
Fixed calibration candidate: three positive of last six observations within
1.5 seconds, current positive required. This is NOT majority voting. Missing
human/ambiguous identity resets; class negatives contribute zero. Literature
motivates temporal posture counts but does not validate our numeric policy:
https://doi.org/10.1177/1550147717707418
"""
from dataclasses import dataclass
import math
from collections import deque

@dataclass(frozen=True)
class HumanPose:
    box: tuple
    torso_confidences: tuple

@dataclass(frozen=True)
class PostureBox:
    box: tuple
    score: float
    label: str

@dataclass(frozen=True)
class LearnedCandidate:
    status: str
    started_at: float
    observed_at: float
    vote_span_seconds: float
    box: tuple
    score: float
    positive_votes: int = 3
    observed_samples: int = 0
    policy: str = 'experimental_person_corroborated_vote3of6_v2'

@dataclass(frozen=True)
class LearnedResult:
    status: str
    candidates: tuple = ()
    reasons: tuple = ()

def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)

def finite(x):return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)
def boxvalid(b):return isinstance(b,(tuple,list)) and len(b)==4 and all(finite(v) and 0<=v<=1 for v in b) and b[2]>b[0] and b[3]>b[1]

class LearnedFallDetector:
    def __init__(self):self.last=None;self.box=None;self.start=None;self.emitted=False;self.votes=deque(maxlen=6)
    def reset(self,why,t=None):
        self.last=t;self.box=None;self.start=None;self.emitted=False;self.votes=deque(maxlen=6)
        return LearnedResult('insufficient_observation',reasons=(why,))
    def update(self,timestamp,poses,classifications,*,frame_valid=True,identity_ambiguous=False):
        if not finite(timestamp) or timestamp<0:return self.reset('invalid_timestamp')
        if frame_valid is not True or identity_ambiguous is not False:return self.reset('invalid_frame_or_identity',timestamp)
        if self.last is not None and (timestamp<=self.last or timestamp-self.last>.5+1e-9):return self.reset('timestamp_or_gap',timestamp)
        if not isinstance(poses,list) or not isinstance(classifications,list) or len(poses)>32 or len(classifications)>128:return self.reset('invalid_input',timestamp)
        for p in poses:
            if not isinstance(p,HumanPose) or not boxvalid(p.box) or not isinstance(p.torso_confidences,(tuple,list)) or len(p.torso_confidences)!=4 or any(not finite(v) or not 0<=v<=1 for v in p.torso_confidences):return self.reset('invalid_pose',timestamp)
        for c in classifications:
            if not isinstance(c,PostureBox) or not boxvalid(c.box) or not finite(c.score) or not 0<=c.score<=1 or c.label not in ('fallen','sitting','standing'):return self.reset('invalid_classification',timestamp)
        humans=[p for p in poses if min(p.torso_confidences)>=.35];matched=[{} for _ in humans]
        for c in classifications:
            if c.score<.25:continue
            indexes=[i for i,p in enumerate(humans) if iou(p.box,c.box)>=.3]
            if len(indexes)>1:return self.reset('ambiguous_human_association',timestamp)
            if indexes:
                scores=matched[indexes[0]];scores[c.label]=max(scores.get(c.label,0),c.score)
        # Class disagreement is not human-identity loss. Preserve the selected
        # physical hypothesis and give every class-negative/uncertain sample zero.
        positives=[]
        for i,scores in enumerate(matched):
            ranked=sorted(scores.items(),key=lambda pair:pair[1],reverse=True)
            positive=bool(ranked and ranked[0][0]=='fallen' and (len(ranked)==1 or ranked[0][1]-ranked[1][1]>=.1-1e-9))
            if positive:positives.append(i)
        if self.box is None:
            if len(positives)!=1:return self.reset('no_unique_fallen_human',timestamp)
            selected=positives[0]
        else:
            indexes=[i for i,p in enumerate(humans) if iou(self.box,p.box)>=.3]
            if len(indexes)!=1:return self.reset('missing_or_ambiguous_tracked_human',timestamp)
            selected=indexes[0]
        self.box=humans[selected].box;self.last=timestamp
        positive=selected in positives
        scores=matched[selected]
        while self.votes and timestamp-self.votes[0][0]>1.5:self.votes.popleft()
        self.votes.append((timestamp,positive))
        votes=[t for t,v in self.votes if v]
        if positive and len(votes)>=3 and not self.emitted:
            self.emitted=True
            return LearnedResult('possible_fallen_posture',(LearnedCandidate('possible_fallen_posture',votes[0],timestamp,timestamp-votes[0],tuple(self.box),scores['fallen'],len(votes),len(self.votes)),))
        return LearnedResult('no_candidate_observed',reasons=(() if positive else ('class_negative_or_uncertain_zero_vote',)))
