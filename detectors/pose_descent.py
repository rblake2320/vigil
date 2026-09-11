"""Experimental rapid descent + low-position candidate, not medical detection.

Thresholds are uncalibrated design choices. This does not distinguish a fall from
fast intentional sitting, nor detect camera motion: callers must invalidate such
frames. Model joint positions and upstream track IDs are untrusted measurements.
"""
from dataclasses import dataclass, field
from collections import deque
import math

@dataclass(frozen=True)
class PoseObservation:
    track_id: str
    shoulder: tuple[float, float]
    hip: tuple[float, float]
    torso_confidences: tuple[float, float, float, float]

@dataclass(frozen=True)
class DescentCandidate:
    track_id: str
    status: str
    baseline_at: float
    descent_at: float
    observed_at: float
    shoulder_drop: float
    hip_drop: float
    low_seconds: float
    policy: str = 'experimental_rapid_descent_low_v1'

@dataclass(frozen=True)
class DescentResult:
    status: str
    candidates: tuple[DescentCandidate, ...] = ()
    reasons: tuple[str, ...] = ()

@dataclass
class _State:
    upright_since: float | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=256))
    onset: float | None = None
    baseline: tuple | None = None
    emitted: bool = False

class PoseDescentDetector:
    """One explicitly selected track; ambiguity or missing observations reset.

    Input coordinates normalized to image width/height; confidence values come
    from actual left/right shoulders and hips. No synthesized confidence allowed.
    """
    def __init__(self):
        self._last=None; self._id=None; self._state=_State()

    def _reset(self, reason, timestamp=None):
        self._last=timestamp; self._id=None; self._state=_State()
        return DescentResult('insufficient_observation',reasons=(reason,))

    def update(self, timestamp, observation=None, *, frame_valid=True,
               identity_ambiguous=False, frame_aspect_ratio=1.0):
        def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
        if not finite(timestamp) or timestamp<0: return self._reset('invalid_timestamp')
        if frame_valid is not True or identity_ambiguous is not False:
            return self._reset('invalid_frame_or_ambiguous_identity',timestamp)
        if not finite(frame_aspect_ratio) or frame_aspect_ratio<=0:
            return self._reset('invalid_aspect_ratio',timestamp)
        if self._last is not None and (timestamp<=self._last or timestamp-self._last>.5+1e-9):
            return self._reset('timestamp_or_observation_gap',timestamp)
        if not isinstance(observation,PoseObservation):return self._reset('missing_pose',timestamp)
        p=observation
        if not isinstance(p.track_id,str) or not p.track_id or len(p.track_id)>128:
            return self._reset('invalid_identity',timestamp)
        if not all(isinstance(v,(tuple,list)) and len(v)==n for v,n in ((p.shoulder,2),(p.hip,2),(p.torso_confidences,4))):
            return self._reset('invalid_pose',timestamp)
        if any(not finite(v) or not 0<=v<=1 for v in (*p.shoulder,*p.hip,*p.torso_confidences)):
            return self._reset('invalid_pose',timestamp)
        if min(p.torso_confidences)<.35:return self._reset('insufficient_joint_confidence',timestamp)
        self._last=timestamp
        if p.track_id!=self._id:
            self._id=p.track_id;self._state=_State()
        s=self._state
        while s.history and timestamp-s.history[0][0]>1.5:s.history.popleft()
        # Maintain an established upright baseline only before onset. A seated
        # upright torso after descent must not erase the low-position episode.
        if s.onset is None and s.history:
            b=s.history[0]
            if p.shoulder[1]-b[1]>=.18 and p.hip[1]-b[2]>=.12:
                s.onset=timestamp;s.baseline=b
        if s.onset is not None:
            b=s.baseline;sd=p.shoulder[1]-b[1];hd=p.hip[1]-b[2]
            if sd<.18 or hd<.12:
                self._state=_State()
                return DescentResult('no_candidate_observed',reasons=('low_position_not_sustained',))
            if not s.emitted and timestamp-s.onset>=1.0-1e-9:
                s.emitted=True
                c=DescentCandidate(p.track_id,'possible_fall',b[0],s.onset,timestamp,sd,hd,timestamp-s.onset)
                return DescentResult('possible_fall',(c,))
            return DescentResult('no_candidate_observed')
        dy=p.hip[1]-p.shoulder[1];dx=abs(p.hip[0]-p.shoulder[0])*frame_aspect_ratio
        upright=dy>=.12 and dx<=dy*.57735026919
        if upright:
            if s.upright_since is None:s.upright_since=timestamp
            if timestamp-s.upright_since>=.5-1e-9:s.history.append((timestamp,p.shoulder[1],p.hip[1]))
        else:s.upright_since=None
        return DescentResult('no_candidate_observed')
