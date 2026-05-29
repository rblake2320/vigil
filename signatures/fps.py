"""
FPS Game Cheats — Detection Signatures
=======================================
Covers: Call of Duty, CS2 / CS:GO, Valorant

Cheat types:
  1. Aimbot (color-based, memory-read, DMA/PCIe hardware)
  2. Silent Aimbot (fires from off-crosshair, invisible to spectator without frame analysis)
  3. Triggerbot (fires the moment crosshair overlaps target)
  4. ESP / Wallhack (bounding boxes, health bars, names through walls)
  5. No-Recoil / Recoil Script (macro or driver-level compensation)
  6. Trigger + Recoil combo (most common in competitive play)

Sources:
  - BotScreen (USENIX Security '23) — labeled aimbot vs human CS:GO crosshair dataset
  - XGuardian (arXiv 2601.18068) — GRU-CNN on pitch/yaw trajectories
  - NVIDIA VADNet (arXiv 2103.10031) — visual ESP detection in FPS
  - DMA For Dummies — PCIe hardware cheat mechanical breakdown
  - panda-community.com — spectator aimbot identification guide
"""

FPS_SIGNATURES = {

    # ─────────────────────────────────────────────────────────────────────────
    # 1. AIMBOT — Crosshair snaps to target with superhuman speed
    # ─────────────────────────────────────────────────────────────────────────
    "aimbot": {
        "name": "Aimbot",
        "description": (
            "Auto-aims crosshair to nearest valid target. Raw aimbot corrects in 0ms. "
            "Smoothed aimbot interpolates over 5-50ms to evade detection."
        ),
        "how_to_detect": [
            "Crosshair velocity > 1500 deg/s — human maximum ~400 deg/s",
            "Acquisition path linearity > 0.97 — human paths curve and wobble",
            "Acquisition time < 15ms from target visibility — human minimum ~150ms",
            "On-target fraction (TAR) > 0.85 — human average ~0.35",
            "Z-pattern flick: three rapid consecutive corrections in a zigzag",
        ],
        "thresholds": {
            "snap_velocity_flag_deg_s": 1500.0,
            "path_linearity_flag": 0.97,
            "acquisition_time_flag_ms": 15.0,
            "target_acquisition_rate_flag": 0.85,
            "human_snap_velocity_mean": 320.0,
            "human_snap_velocity_std": 110.0,
            "human_path_linearity_mean": 0.78,
            "human_tar_mean": 0.32,
        },
        "visual_roi": "crosshair_region",
        "confidence": "HIGH",
        "collect_label": "aimbot",
        "priority": 1,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 2. SILENT AIMBOT — Fires from off-screen without visible snap
    # ─────────────────────────────────────────────────────────────────────────
    "silent_aimbot": {
        "name": "Silent Aimbot",
        "description": (
            "Fires a shot that registers as a hit even when the crosshair is not "
            "visually on target. The aimbot snaps, fires, and returns to original "
            "position within 1-3 frames — invisible without frame-level analysis."
        ),
        "how_to_detect": [
            "Blood hit marker / damage event fires with crosshair visually off target",
            "Subtle crosshair snap toward target and back to origin in same frame window",
            "Hit registration without corresponding crosshair-on-target frame",
            "Player scores hits without their crosshair ever visibly touching the enemy",
        ],
        "thresholds": {
            "hit_without_crosshair_flag": True,
            "snap_return_window_frames": 3,
            "hit_marker_crosshair_distance_flag_px": 30,
        },
        "visual_roi": "full_frame",
        "confidence": "MEDIUM_HIGH",
        "collect_label": "silent_aimbot",
        "priority": 2,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TRIGGERBOT — Fires instantly on target contact
    # ─────────────────────────────────────────────────────────────────────────
    "triggerbot": {
        "name": "Triggerbot",
        "description": (
            "Fires the moment the crosshair overlaps a valid target pixel. "
            "Reaction time is 1-10ms — the human biological floor is 140ms."
        ),
        "how_to_detect": [
            "Shot fires in < 60ms after crosshair reaches target",
            "Reaction time distribution has no variance — all shots fire at same delay",
            "Fires on brief contacts (<40ms) that humans would miss",
            "No pre-fire anticipation — fires only when exactly on target",
        ],
        "thresholds": {
            "human_reaction_floor_ms": 140.0,
            "flag_reaction_below_ms": 60.0,
            "brief_contact_fire_rate_flag": 0.80,
            "reaction_std_flag": 5.0,
        },
        "visual_roi": "crosshair_region",
        "confidence": "HIGH",
        "collect_label": "triggerbot",
        "priority": 3,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ESP / WALLHACK — Enemy positions revealed through walls
    # ─────────────────────────────────────────────────────────────────────────
    "esp_wallhack": {
        "name": "ESP / Wallhack",
        "description": (
            "Enemy unit positions, health, and locations are rendered through walls "
            "and terrain. Typically appears as colored bounding boxes or player outlines."
        ),
        "how_to_detect": [
            "Colored bounding boxes (red/green/yellow) visible over wall/terrain pixels",
            "Health bar or name tag floating over a solid surface region",
            "Player pre-aims at enemy position 2-3 seconds before enemy becomes visible",
            "Uniform-border rectangles rendered on non-player screen regions",
            "Wireframe/outline overlays on terrain where no player should be",
        ],
        "thresholds": {
            "overlay_box_confidence": 0.70,
            "pre_aim_lead_time_s": 2.0,
            "wall_tracking_index_flag": 0.70,
        },
        "visual_roi": "full_frame",
        "confidence": "HIGH",
        "collect_label": "wallhack",
        "priority": 4,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 5. NO-RECOIL / RECOIL SCRIPT
    # ─────────────────────────────────────────────────────────────────────────
    "no_recoil": {
        "name": "No-Recoil / Recoil Script",
        "description": (
            "Automatically compensates for weapon recoil, moving the crosshair "
            "downward at the exact inverse of the recoil pattern. Result: "
            "perfectly straight sprays at any range."
        ),
        "how_to_detect": [
            "Sustained automatic fire with crosshair showing near-zero vertical drift",
            "Spray standard deviation < 0.15 degrees during full-auto",
            "Micro-corrections each frame exactly matching recoil pattern",
            "Full-auto accuracy at long range that no human achieves",
        ],
        "thresholds": {
            "spray_std_flag_deg": 0.15,
            "correction_lag_flag_ms": 8.0,
            "recoil_pattern_correlation_flag": 0.90,
            "human_spray_std_normal": 2.5,
        },
        "visual_roi": "crosshair_region",
        "confidence": "MEDIUM_HIGH",
        "collect_label": "no_recoil",
        "priority": 5,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 6. SPEED HACK (FPS variant)
    # ─────────────────────────────────────────────────────────────────────────
    "speed_hack": {
        "name": "Speed Hack",
        "description": (
            "Player movement speed exceeds game physics maximum. "
            "Appears as teleportation or 'ice skating' where feet animate at normal "
            "speed but body moves impossibly fast."
        ),
        "how_to_detect": [
            "Player pixel displacement > 2x expected maximum per frame",
            "Feet animation cycle at normal speed while body moves 3x faster",
            "Position teleports — skips movement frames entirely",
            "Traverses known map distance in fraction of expected time",
        ],
        "thresholds": {
            "max_px_per_frame_normal": 12.0,
            "flag_px_per_frame": 25.0,
            "teleport_gap_px": 80.0,
            "animation_position_ratio_flag": 2.0,
        },
        "visual_roi": "player_region",
        "confidence": "MEDIUM_HIGH",
        "collect_label": "speed_hack",
        "priority": 6,
    },
}

CHEAT_NAMES = {k: v["name"] for k, v in FPS_SIGNATURES.items()}
PRIORITY_ORDER = sorted(FPS_SIGNATURES.keys(),
                        key=lambda k: FPS_SIGNATURES[k]["priority"])


def describe_all():
    print(f"\nFPS Games (CoD/CS2/Valorant) — {len(FPS_SIGNATURES)} cheat signatures\n")
    for k in PRIORITY_ORDER:
        sig = FPS_SIGNATURES[k]
        print(f"  [{sig['priority']}] {sig['name']} ({sig['confidence']} confidence)")
        print(f"      {sig['description'][:90]}...")
        print()


if __name__ == "__main__":
    describe_all()
