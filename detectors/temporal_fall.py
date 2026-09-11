"""Deterministic candidate heuristic. Not medical detection or calibrated confidence.

The caller supplies media seconds and stable per-person identities with normalized
xyxy boxes. Pose/tracker inference and evidence capture belong to the caller.
Brief missing observations pause horizontal credit; ambiguity resets it.
Horizontal hold is cumulative observed interval time within a two-second episode,
not continuous wall time. Missing intervals receive zero credit; no safety claim.
"""
from dataclasses import dataclass, field
import math
from collections import deque


@dataclass(frozen=True)
class PersonObservation:
    track_id: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class FallCandidate:
    track_id: str
    status: str
    upright_at: float
    horizontal_at: float
    observed_at: float
    center_drop: float
    observed_horizontal_seconds: float = 0.0
    horizontal_span_seconds: float = 0.0
    hold_policy: str = "observed_intervals_within_2s_v1"


@dataclass(frozen=True)
class FrameResult:
    status: str
    candidates: tuple[FallCandidate, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass
class _Track:
    upright_since: float | None = None
    upright_at: float | None = None
    upright_center: float | None = None
    baseline_ready: bool = False
    upright_history: deque = field(default_factory=lambda: deque(maxlen=1024))
    horizontal_at: float | None = None
    horizontal_credit: float = 0.0
    horizontal_last: float | None = None
    last_seen: float | None = None
    drop: float = 0.0
    emitted: bool = False


class TemporalFallDetector:
    def __init__(self, *, upright_hold: float = 0.5, rapid_window: float = 1.5,
                 horizontal_hold: float = 1.0, min_center_drop: float = 0.12,
                 max_frame_gap: float = 0.5, max_tracks: int = 32):
        values=(upright_hold, rapid_window, horizontal_hold, min_center_drop,max_frame_gap)
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in values):
            raise ValueError("Temporal thresholds must be finite and positive")
        if min_center_drop>=1 or not isinstance(max_tracks,int) or isinstance(max_tracks,bool) or max_tracks<1:
            raise ValueError("Invalid normalized drop or track bound")
        self.upright_hold=upright_hold
        self.rapid_window=rapid_window
        self.horizontal_hold=horizontal_hold
        self.min_center_drop=min_center_drop
        self.max_frame_gap=max_frame_gap
        self.max_tracks=max_tracks
        self._last_time=None
        self._tracks: dict[str,_Track]={}

    def _unknown(self, reason: str, timestamp: float | None) -> FrameResult:
        self._tracks.clear()
        self._last_time=timestamp
        return FrameResult("insufficient_observation",reasons=(reason,))

    def pause_observation(self) -> None:
        """Retain established history/episode, never credit a missing interval.

        Do not move _last_time: next real sample must still pass max_frame_gap.
        Unfinished upright evidence is discarded; horizontal interval is interrupted.
        """
        for track in self._tracks.values():
            if not track.baseline_ready:track.upright_since=None
            track.horizontal_last=None

    def update(self, timestamp: float, observations: list[PersonObservation], *,
               frame_valid: bool = True, identity_ambiguous: bool = False,
               frame_aspect_ratio: float = 1.0) -> FrameResult:
        if isinstance(timestamp,bool) or not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or timestamp<0:
            return self._unknown("invalid_timestamp",None)
        if frame_valid is not True or identity_ambiguous is not False:
            return self._unknown("invalid_frame_or_ambiguous_identity",timestamp)
        if isinstance(frame_aspect_ratio,bool) or not isinstance(frame_aspect_ratio,(int,float)) or not math.isfinite(frame_aspect_ratio) or frame_aspect_ratio<=0:
            return self._unknown("invalid_frame_aspect_ratio",timestamp)
        if self._last_time is not None:
            if timestamp<=self._last_time:
                return self._unknown("nonmonotonic_timestamp",timestamp)
            if timestamp-self._last_time>self.max_frame_gap+1e-9:
                return self._unknown("observation_gap",timestamp)
        if not isinstance(observations,list) or len(observations)>self.max_tracks:
            return self._unknown("invalid_or_excessive_tracks",timestamp)
        ids=set()
        for person in observations:
            if not isinstance(person,PersonObservation) or not isinstance(person.track_id,str) or not person.track_id or len(person.track_id)>128 or person.track_id in ids:
                return self._unknown("invalid_or_duplicate_identity",timestamp)
            ids.add(person.track_id)
            box=person.box
            if not isinstance(box,(tuple,list)) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in box):
                return self._unknown("invalid_box",timestamp)
            x1,y1,x2,y2=box
            if x2<=x1 or y2<=y1:
                return self._unknown("empty_or_inverted_box",timestamp)
        self._last_time=timestamp
        missing=set(self._tracks)-ids
        for track_id in missing:
            track=self._tracks[track_id]
            if (ids or track.horizontal_at is None or track.last_seen is None
                    or timestamp-track.last_seen>self.max_frame_gap+1e-9
                    or timestamp-track.horizontal_at>2.0):
                del self._tracks[track_id]
            else:
                track.horizontal_last=None # missing frame adds zero duration
                track.upright_since=None
        if not observations:
            return FrameResult("insufficient_observation",reasons=("no_person_observation",))
        candidates=[]
        for person in observations:
            track=self._tracks.setdefault(person.track_id,_Track())
            if track.last_seen is not None and timestamp-track.last_seen>self.max_frame_gap+1e-9:
                track=_Track();self._tracks[person.track_id]=track
            track.last_seen=timestamp
            if track.horizontal_at is not None and timestamp-track.horizontal_at>2.0:
                track=_Track(last_seen=timestamp);self._tracks[person.track_id]=track
            while track.upright_history and timestamp-track.upright_history[0][0]>self.rapid_window:
                track.upright_history.popleft()
            x1,y1,x2,y2=person.box
            # Box x/y were normalized by different dimensions. Restore the
            # pixel posture ratio before classifying upright/horizontal.
            width,height=(x2-x1)*frame_aspect_ratio,y2-y1
            center=(y1+y2)/2
            if height/width>=1.4:
                if track.upright_since is None: track.upright_since=timestamp
                track.baseline_ready=timestamp-track.upright_since+1e-9>=self.upright_hold
                track.upright_at=timestamp
                track.upright_center=center
                # Only samples backed by the established upright hold qualify.
                # Retain descent history rather than replacing it with the last
                # already-descended upright frame. Bounded storage may discard
                # older valid samples at extreme frame rates, never extend time.
                if track.baseline_ready:
                    track.upright_history.append((timestamp, center))
                else:
                    track.upright_history.clear()
                track.horizontal_at=None
                track.horizontal_credit=0.0
                track.horizontal_last=None
                track.emitted=False
                continue
            track.upright_since=None
            if width/height<1.2:
                track.horizontal_at=None
                track.horizontal_credit=0.0
                track.horizontal_last=None
                continue
            if track.horizontal_at is None:
                if not track.baseline_ready or track.upright_at is None or track.upright_center is None:
                    continue
                if not track.upright_history:
                    continue
                baseline_at, baseline_center = track.upright_history[0]
                drop=center-baseline_center
                if timestamp-baseline_at>self.rapid_window or drop<self.min_center_drop:
                    continue
                track.upright_at=baseline_at
                track.horizontal_at=timestamp
                track.drop=drop
                track.horizontal_credit=0.0
                track.horizontal_last=None
            if track.horizontal_last is not None:
                track.horizontal_credit += timestamp-track.horizontal_last
            track.horizontal_last=timestamp
            if not track.emitted and track.horizontal_credit+1e-9>=self.horizontal_hold:
                candidates.append(FallCandidate(person.track_id,"possible_fall",track.upright_at,track.horizontal_at,timestamp,track.drop,track.horizontal_credit,timestamp-track.horizontal_at))
                track.emitted=True
        reasons=("track_disappeared_continuity_reset",) if missing else ()
        return FrameResult("possible_fall" if candidates else "insufficient_observation" if missing else "no_candidate_observed",tuple(candidates),reasons)
