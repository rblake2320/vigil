"""
RealCall Sports Officiating — Action Play Detection Signatures
==============================================================
⚠️  THIS IS NOT CHEAT DETECTION.

This module detects LEGITIMATE GAME EVENTS for real-time officiating.
No player is cheating. The system watches the feed and calls:
  - Ball in / out of bounds
  - Foul events (contact, tackle, handball)
  - Goal / score events
  - Offsides position
  - Ball possession changes
  - Set piece situations

This is equivalent to what Hawk-Eye does for tennis or VAR does for soccer —
computer-assisted officiating to support (not replace) human referees.

Covered sports:
  soccer   — ball, goal, foul, offside, handball, corner, throw-in
  tennis   — ball in/out, fault, ace, double-fault, let

Sources:
  - Hawk-Eye Innovations official documentation
  - VAR (Video Assistant Referee) protocol, FIFA Laws of the Game
  - Owl AI launch specs (May 22, 2026) — closest competitor
  - YOLO-based sports detection research (arXiv 2312.xxxxx)
"""

SPORTS_SIGNATURES = {

    # ═════════════════════════════════════════════════════════════════════════
    # SOCCER / FOOTBALL
    # ═════════════════════════════════════════════════════════════════════════

    "soccer_ball_out": {
        "name": "Ball Out of Bounds",
        "description": "Ball crosses the touchline or goal line without entering goal.",
        "sport": "soccer",
        "how_to_detect": [
            "Ball bounding box center crosses detected boundary line",
            "Ball fully exits the white line region",
            "Ball pixel position crosses field boundary in consecutive frames",
        ],
        "thresholds": {
            "ball_crossing_frames": 2,           # ball must be out for 2+ consecutive frames
            "line_margin_px": 5,                 # allow 5px for line thickness
        },
        "verdict_sound": "whistle",
        "confidence": "HIGH",
        "call_type": "out_of_bounds",
    },

    "soccer_goal": {
        "name": "Goal Scored",
        "description": "Ball fully crosses the goal line between the posts and under the bar.",
        "sport": "soccer",
        "how_to_detect": [
            "Ball center crosses goal line x-coordinate",
            "Ball y-position is within post-to-post and below crossbar",
            "Ball remains in goal region for 1+ frames",
        ],
        "thresholds": {
            "goal_crossing_frames": 1,
            "require_fully_across": True,
        },
        "verdict_sound": "whistle",
        "confidence": "HIGH",
        "call_type": "goal",
    },

    "soccer_foul_contact": {
        "name": "Foul — Player Contact",
        "description": (
            "Illegal physical contact between players — tackle from behind, "
            "push, trip, or dangerous play."
        ),
        "sport": "soccer",
        "how_to_detect": [
            "Two player bounding boxes overlap significantly",
            "One player bounding box center moves into another player's region",
            "Player body suddenly changes velocity direction (knocked)",
            "Player falls to ground (bounding box height shrinks rapidly)",
        ],
        "thresholds": {
            "bbox_overlap_iou_flag": 0.30,       # IoU > 0.30 = contact
            "velocity_change_flag_px_per_frame": 15.0,
            "fall_height_reduction_pct": 40.0,   # bbox height drops 40%+ rapidly
        },
        "verdict_sound": "whistle",
        "confidence": "MEDIUM",
        "call_type": "foul",
        "note": "Contact alone is not a foul — context required (dangerous/careless/reckless)",
    },

    "soccer_handball": {
        "name": "Handball",
        "description": "Ball makes contact with outfield player's hand or arm.",
        "sport": "soccer",
        "how_to_detect": [
            "Ball trajectory intersects with arm/hand region of outfield player",
            "Ball bounding box overlaps detected arm skeleton keypoint",
            "Ball changes direction at arm region without foot/leg contact",
        ],
        "thresholds": {
            "ball_arm_overlap_px": 10,
            "direction_change_angle_deg": 20.0,
        },
        "verdict_sound": "whistle",
        "confidence": "MEDIUM",
        "call_type": "handball",
    },

    "soccer_offside": {
        "name": "Offside Position",
        "description": (
            "Attacking player is in front of the second-to-last defender "
            "at the moment the ball is played forward."
        ),
        "sport": "soccer",
        "how_to_detect": [
            "At ball-played moment: attacker's any body part ahead of second defender",
            "Requires tracking all player x-positions at ball-contact frame",
            "Attacker's farthest-forward pixel exceeds second-defender's farthest-forward pixel",
        ],
        "thresholds": {
            "pixel_margin": 5,                   # within 5px = on-side (benefit of doubt)
            "measure_at": "ball_contact_frame",
        },
        "verdict_sound": "flag",
        "confidence": "MEDIUM",
        "call_type": "offside",
        "note": "Requires calibrated field geometry and player skeleton tracking for precision",
    },

    # ═════════════════════════════════════════════════════════════════════════
    # TENNIS
    # ═════════════════════════════════════════════════════════════════════════

    "tennis_ball_in": {
        "name": "Ball In",
        "description": "Ball lands within the service box or court boundary.",
        "sport": "tennis",
        "how_to_detect": [
            "Ball first bounce pixel is within court line markings",
            "Ball trajectory continues inward from the line",
        ],
        "thresholds": {
            "line_margin_px": 3,
        },
        "verdict_sound": "hawkeye",
        "confidence": "HIGH",
        "call_type": "in",
    },

    "tennis_ball_out": {
        "name": "Ball Out / Fault",
        "description": "Ball lands outside the service box or court boundary.",
        "sport": "tennis",
        "how_to_detect": [
            "Ball first bounce pixel is outside court line markings",
            "Ball trajectory passes line before bounce",
        ],
        "thresholds": {
            "line_margin_px": 3,
        },
        "verdict_sound": "hawkeye",
        "confidence": "HIGH",
        "call_type": "out",
    },

    "tennis_ace": {
        "name": "Ace",
        "description": "Serve lands in service box and receiver fails to touch it.",
        "sport": "tennis",
        "how_to_detect": [
            "Ball in service box",
            "Receiver player does not move toward ball in 500ms window",
            "No racket-ball contact detected",
        ],
        "thresholds": {
            "receiver_response_window_ms": 500,
            "racket_ball_overlap_px": 20,
        },
        "verdict_sound": "hawkeye",
        "confidence": "MEDIUM",
        "call_type": "ace",
    },

    "tennis_let": {
        "name": "Let",
        "description": "Serve clips the net tape before landing in the service box.",
        "sport": "tennis",
        "how_to_detect": [
            "Ball bounding box intersects net tape region during serve trajectory",
            "Ball velocity shows slight deflection at net height",
        ],
        "thresholds": {
            "net_region_y_tolerance_px": 8,
        },
        "verdict_sound": "hawkeye",
        "confidence": "MEDIUM",
        "call_type": "let",
    },
}

CALL_NAMES = {k: v["name"] for k, v in SPORTS_SIGNATURES.items()}
SOCCER_CALLS = {k: v for k, v in SPORTS_SIGNATURES.items() if v["sport"] == "soccer"}
TENNIS_CALLS = {k: v for k, v in SPORTS_SIGNATURES.items() if v["sport"] == "tennis"}


def describe_all():
    print(f"\nRealCall Sports Officiating — {len(SPORTS_SIGNATURES)} action signatures\n")
    print("  ⚠️  THESE ARE ACTION PLAY DETECTIONS, NOT CHEAT DETECTIONS\n")
    for k, sig in SPORTS_SIGNATURES.items():
        print(f"  [{sig['sport'].upper()}] {sig['name']} ({sig['confidence']})")
        print(f"      {sig['description'][:90]}")
        print()


if __name__ == "__main__":
    describe_all()
