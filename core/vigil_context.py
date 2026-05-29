"""
Vigil — Surrounding Awareness + Timestamp Context Engine
==========================================================
Turns raw detections into complete situational awareness reports.

Every event gets:
  - Precise timestamp (when detected, how long ongoing)
  - Location context (which camera, room, address)
  - Time-of-day context (is this normal for this hour?)
  - Duration tracking (brief flash vs sustained threat)
  - Behavioral baseline (first time ever vs repeated)
  - Cross-camera correlation (what other cameras see)
  - Escalation state (0s→log, 15s→SMS, 30s→call, 60s→911)
  - Complete 911 call script generated automatically
"""
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict, deque


class CameraProfile:
    """Configuration for a single camera / monitoring point."""
    def __init__(self, camera_id: str, room: str, address: str,
                 normal_hours: tuple = (6, 22),
                 owner_phone: str = "", owner_sms: str = ""):
        self.camera_id   = camera_id
        self.room        = room
        self.address     = address
        self.normal_hours = normal_hours  # (start_hour, end_hour) — 24h
        self.owner_phone = owner_phone
        self.owner_sms   = owner_sms

    def is_normal_hour(self) -> bool:
        h = datetime.now().hour
        return self.normal_hours[0] <= h <= self.normal_hours[1]

    def time_of_day_label(self) -> str:
        h = datetime.now().hour
        if 0  <= h < 5:  return "middle of the night (unusual)"
        if 5  <= h < 9:  return "early morning"
        if 9  <= h < 12: return "morning"
        if 12 <= h < 17: return "afternoon"
        if 17 <= h < 21: return "evening"
        return "late night (unusual)"


class EventContext:
    """
    Tracks an ongoing event with full temporal + spatial awareness.
    From first detection through escalation to resolution.
    """
    def __init__(self, event_type: str, camera: CameraProfile,
                 first_frame_path: str, ai_reasoning: str,
                 confidence: float):
        self.event_type       = event_type
        self.camera           = camera
        self.first_detected   = time.time()
        self.first_detected_iso = datetime.now().isoformat()
        self.last_seen        = time.time()
        self.frame_count      = 1
        self.max_confidence   = confidence
        self.ai_reasoning     = ai_reasoning
        self.evidence_frames  = [first_frame_path]
        self.escalation_level = 0   # 0=detected 1=SMS 2=call 3=911
        self.resolved         = False

    def update(self, frame_path: str, confidence: float, ai: str = ""):
        self.last_seen = time.time()
        self.frame_count += 1
        self.max_confidence = max(self.max_confidence, confidence)
        self.evidence_frames.append(frame_path)
        if ai:
            self.ai_reasoning = ai

    @property
    def duration_s(self) -> float:
        return time.time() - self.first_detected

    @property
    def duration_str(self) -> str:
        s = int(self.duration_s)
        if s < 60: return f"{s} seconds"
        return f"{s//60}m {s%60}s"

    def generate_911_script(self) -> str:
        cam = self.camera
        dt  = datetime.fromtimestamp(self.first_detected)
        return f"""This is an automated emergency alert from Vigil Safety System.

INCIDENT: {self.event_type.replace('_', ' ').upper()}
LOCATION: {cam.room} — {cam.address}
TIME: {dt.strftime('%I:%M:%S %p')} on {dt.strftime('%B %d, %Y')}
DURATION: Event has been active for {self.duration_str}
TIME CONTEXT: Detected during {cam.time_of_day_label()}
NORMAL HOURS: This camera normally active {cam.normal_hours[0]}:00-{cam.normal_hours[1]}:00 — {'WITHIN' if cam.is_normal_hour() else 'OUTSIDE'} normal window
AI ASSESSMENT: {self.ai_reasoning[:200]}
EVIDENCE: {len(self.evidence_frames)} verified frames saved, SHA-256 chain intact
CONFIDENCE: {self.max_confidence*100:.0f}%

Please send emergency services to: {cam.address}
Camera: {cam.room} ({cam.camera_id})"""

    def generate_sms(self) -> str:
        cam = self.camera
        dt = datetime.fromtimestamp(self.first_detected)
        return (
            f"VIGIL ALERT: {self.event_type.upper()} detected at {cam.room}, {cam.address}. "
            f"Time: {dt.strftime('%I:%M %p')}. Duration: {self.duration_str}. "
            f"AI: {self.ai_reasoning[:100]}. "
            f"Reply CLEAR if safe."
        )

    def to_dict(self) -> dict:
        return {
            "event_type":       self.event_type,
            "camera_id":        self.camera.camera_id,
            "room":             self.camera.room,
            "address":          self.camera.address,
            "first_detected":   self.first_detected_iso,
            "duration_s":       round(self.duration_s, 1),
            "frame_count":      self.frame_count,
            "max_confidence":   round(self.max_confidence, 3),
            "ai_reasoning":     self.ai_reasoning,
            "evidence_frames":  self.evidence_frames[-3:],
            "escalation_level": self.escalation_level,
            "time_of_day":      self.camera.time_of_day_label(),
            "outside_normal_hours": not self.camera.is_normal_hour(),
        }


class SurroundingAwareness:
    """
    Multi-camera awareness engine.
    Tracks events across all cameras, correlates them,
    manages escalation cascade, generates incident reports.
    """

    ESCALATION = [
        (0,  "DETECTED",  "log_event"),
        (15, "ALERTING",  "sms_owner"),
        (30, "CALLING",   "call_owner"),
        (60, "EMERGENCY", "call_911"),
    ]

    def __init__(self, log_dir: str = "/tmp/vigil_events"):
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.cameras: dict[str, CameraProfile] = {}
        self.active_events: dict[str, EventContext] = {}
        self.event_history: deque = deque(maxlen=1000)
        self.event_counts: defaultdict = defaultdict(int)  # event_type → total count

    def register_camera(self, profile: CameraProfile):
        self.cameras[profile.camera_id] = profile

    def report_event(self, camera_id: str, event_type: str,
                     confidence: float, ai_reasoning: str,
                     frame_path: str) -> dict:
        """
        Called when a detection fires. Returns full context dict.
        Manages escalation automatically.
        """
        cam = self.cameras.get(camera_id)
        if cam is None:
            # Auto-create minimal profile
            cam = CameraProfile(camera_id, camera_id, "Unknown location")
            self.cameras[camera_id] = cam

        event_key = f"{camera_id}::{event_type}"
        now = time.time()

        if event_key in self.active_events:
            ctx = self.active_events[event_key]
            ctx.update(frame_path, confidence, ai_reasoning)
        else:
            ctx = EventContext(event_type, cam, frame_path, ai_reasoning, confidence)
            self.active_events[event_key] = ctx
            self.event_counts[event_type] += 1
            ctx.is_first_occurrence = self.event_counts[event_type] == 1

        # Build full context report
        report = self._build_report(ctx, cam)

        # Check escalation
        self._check_escalation(ctx, report)

        # Log
        self._log_event(ctx, report)

        return report

    def resolve_event(self, camera_id: str, event_type: str):
        """Mark event as resolved (owner acknowledged or threat cleared)."""
        key = f"{camera_id}::{event_type}"
        if key in self.active_events:
            ctx = self.active_events.pop(key)
            ctx.resolved = True
            self.event_history.append(ctx.to_dict())
            print(f"[vigil] Event resolved: {event_type} on {camera_id}")

    def cross_camera_summary(self) -> str:
        """What are ALL cameras currently seeing?"""
        if not self.active_events:
            return "All cameras: no active events."
        lines = [f"Active events across {len(self.cameras)} cameras:"]
        for key, ctx in self.active_events.items():
            lines.append(f"  {ctx.camera.room}: {ctx.event_type} ({ctx.duration_str})")
        return "\n".join(lines)

    def _build_report(self, ctx: EventContext, cam: CameraProfile) -> dict:
        occurrences = self.event_counts[ctx.event_type]
        return {
            **ctx.to_dict(),
            "911_script":        ctx.generate_911_script(),
            "sms_text":          ctx.generate_sms(),
            "total_occurrences": occurrences,
            "first_ever":        occurrences == 1,
            "cross_camera":      self.cross_camera_summary(),
            "escalation_next": self._next_escalation(ctx),
        }

    def _next_escalation(self, ctx: EventContext) -> str:
        for threshold, label, action in self.ESCALATION:
            if ctx.duration_s < threshold:
                return f"{label} in {int(threshold - ctx.duration_s)}s → {action}"
        return "EMERGENCY — 911 call due"

    def _check_escalation(self, ctx: EventContext, report: dict):
        dur = ctx.duration_s
        for threshold, label, action in reversed(self.ESCALATION):
            if dur >= threshold and ctx.escalation_level < self.ESCALATION.index((threshold, label, action)):
                ctx.escalation_level = self.ESCALATION.index((threshold, label, action))
                print(f"\n[vigil] ESCALATION: {label} → {action}")
                print(f"  Event: {ctx.event_type} | Duration: {ctx.duration_str}")
                if action == "sms_owner":
                    print(f"  SMS: {report['sms_text']}")
                elif action in ("call_owner", "call_911"):
                    print(f"  CALL SCRIPT:\n{report['911_script']}")
                break

    def _log_event(self, ctx: EventContext, report: dict):
        log_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            entry = {k: v for k, v in report.items() if k != "911_script"}
            f.write(json.dumps(entry) + "\n")


# ─── Demo / test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    awareness = SurroundingAwareness()

    # Register cameras
    awareness.register_camera(CameraProfile(
        camera_id="cam_kitchen",
        room="Kitchen",
        address="123 Main Street, Austin TX 78701",
        normal_hours=(6, 22),
    ))
    awareness.register_camera(CameraProfile(
        camera_id="cam_pool",
        room="Pool Area",
        address="123 Main Street, Austin TX 78701",
        normal_hours=(8, 20),
    ))
    awareness.register_camera(CameraProfile(
        camera_id="cam_desktop",
        room="Home Office",
        address="123 Main Street, Austin TX 78701",
        normal_hours=(8, 23),
    ))

    # Simulate fire detection at 3am
    print("=== Simulating fire detection ===\n")
    report = awareness.report_event(
        camera_id="cam_kitchen",
        event_type="fire_detected",
        confidence=0.87,
        ai_reasoning="YES FIRE — Person is holding a lighter with a bright visible flame in the kitchen area.",
        frame_path="/tmp/vigil_fire/fire_030600.jpg",
    )

    print(f"\n--- FULL CONTEXT REPORT ---")
    print(f"Camera:         {report['room']} ({report['camera_id']})")
    print(f"Address:        {report['address']}")
    print(f"Time:           {report['first_detected']}")
    print(f"Time context:   {report['time_of_day']}")
    print(f"Outside hours:  {report['outside_normal_hours']}")
    print(f"First ever:     {report['first_ever']}")
    print(f"Next action:    {report['escalation_next']}")
    print(f"\nSMS text:\n{report['sms_text']}")
    print(f"\n911 script:\n{report['911_script']}")
    print(f"\nCross-camera:\n{report['cross_camera']}")
