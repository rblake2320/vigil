"""
Dark War Survival — Cheat Detection Signatures
===============================================
Dark War Survival is a MOBILE RTS / SURVIVAL STRATEGY game.
Cheats here are fundamentally different from FPS cheats.
No crosshair. No aimbot. Detection is based on:
  - HP bar temporal tracking (God Mode)
  - Time-to-kill measurement (High Damage)
  - Attack animation rate (High Speed Attack)
  - Ability cooldown UI state (No Skill Cooldown)
  - Game simulation pacing (Global Speed Multiplier)
  - Resource counter behavior (Unlimited Resources)

Sources:
  - platinmods.com v1.250.641–643 confirmed mod menu feature list
  - liteapks.com, modyolo.com corroboration
  - NVIDIA arXiv 2103.10031 — visual cheat detection methodology
  - BotScreen USENIX '23 — behavioral baseline methodology

Each signature has:
  name          — cheat type identifier
  description   — what the cheat does in-game
  how_to_detect — what a camera watching the HDMI feed can see
  thresholds    — numeric detection cutoffs
  confidence    — HIGH / MEDIUM / LOW based on detectability from HDMI feed
  collect_label — label to use when collecting training data for this cheat
"""

DARKWAR_SIGNATURES = {

    # ─────────────────────────────────────────────────────────────────────────
    # 1. GOD MODE — Easiest to detect, highest priority
    # ─────────────────────────────────────────────────────────────────────────
    "god_mode": {
        "name": "God Mode / Invincible",
        "description": (
            "HP value is locked at maximum. Incoming damage is processed by the engine "
            "and hit animations fire normally, but the health bar never decrements."
        ),
        "how_to_detect": [
            "Health bar stays at 100% throughout an entire combat sequence",
            "Hit/flinch animations fire on the unit while HP bar does not move",
            "Unit survives sustained fire from multiple enemies with zero HP loss",
            "In PvP arena: player absorbs full assault without HP dropping below a threshold",
        ],
        "thresholds": {
            "hp_delta_over_combat_window": 0.0,   # flag if HP delta = 0 for >10s of active combat
            "min_combat_duration_s": 10.0,        # must be in combat this long to count
            "max_hp_variance_pct": 2.0,           # allow 2% HP variance (render rounding)
            "flag_if_hits_absorbed": 3,           # flag if 3+ hit animations with zero HP change
        },
        "visual_roi": "hp_bar_region",            # camera region to track
        "confidence": "HIGH",
        "collect_label": "god_mode",
        "priority": 1,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 2. HIGH DAMAGE — One-hit kills, instant enemy deletion
    # ─────────────────────────────────────────────────────────────────────────
    "high_damage": {
        "name": "High Damage / One-Hit Kill",
        "description": (
            "Player damage output is multiplied many times. Full-health enemy units "
            "die in 1-2 attacks instead of the normal 5-20+ hit count per unit tier."
        ),
        "how_to_detect": [
            "Enemy health bar drops from 100% to 0% in a single attack animation cycle",
            "High-tier enemies (bosses, fortifications) eliminated on first contact",
            "Combat that should take 10-30 seconds resolves in 1-2 seconds",
            "Squad kills a force many times its size in seconds with no attrition",
        ],
        "thresholds": {
            "min_ttk_s_normal": 3.0,            # normal minimum time-to-kill any unit
            "flag_ttk_below_s": 0.5,            # flag if enemy dies in <0.5s from full HP
            "one_hit_flag": True,               # flag any full-HP → 0 in single attack
            "kill_rate_multiplier_flag": 3.0,   # flag if kills/sec > 3x baseline for troop count
        },
        "visual_roi": "enemy_hp_bar_region",
        "confidence": "HIGH",
        "collect_label": "high_damage",
        "priority": 2,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 3. NO SKILL COOLDOWN — Ability spam with zero recharge
    # ─────────────────────────────────────────────────────────────────────────
    "no_skill_cooldown": {
        "name": "No Skill Cooldown",
        "description": (
            "Hero/unit special abilities are usable every frame. The cooldown timer "
            "that should prevent re-use for 10-60 seconds never activates."
        ),
        "how_to_detect": [
            "Skill effect (explosion, heal pulse, AoE) fires back-to-back with zero gap",
            "Ability icon UI element never grays out or shows countdown timer",
            "Hero uses 4+ special abilities in a row within 10 seconds",
            "Cooldown ring/fill animation on ability icon is never observed",
        ],
        "thresholds": {
            "min_ability_cooldown_s": 8.0,      # documented minimum cooldown in game
            "flag_if_reuse_within_s": 3.0,      # flag if same ability fires within 3s
            "ability_spam_count": 4,            # flag if 4+ ability uses in 10s window
            "cooldown_ui_absent_window_s": 60.0,# flag if no cooldown UI seen in 60s of combat
        },
        "visual_roi": "ability_bar_region",
        "confidence": "HIGH",
        "collect_label": "no_cooldown",
        "priority": 3,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 4. HIGH SPEED ATTACK — Attack animations faster than game physics allows
    # ─────────────────────────────────────────────────────────────────────────
    "high_speed_attack": {
        "name": "High Speed Attack",
        "description": (
            "Attack animation rate is multiplied far beyond the unit's documented "
            "attacks-per-second cap. Animations may stutter, blur, or become incoherent."
        ),
        "how_to_detect": [
            "Attack animations stutter or blur — frame interpolation can't keep up",
            "Attacks per second visibly exceeds the maximum for any unit type",
            "Audio-visual desync: attack sound effects overlap or clip",
            "Units appear to vibrate or oscillate due to rapid attack cycling",
        ],
        "thresholds": {
            "max_attacks_per_sec_normal": 2.5,  # documented maximum attack rate
            "flag_attacks_per_sec": 5.0,        # flag if >5 attacks/sec observed
            "animation_blur_threshold": 0.15,   # motion blur score above this = flag
            "audio_visual_desync_ms": 80,       # flag if audio leads visual by >80ms
        },
        "visual_roi": "combat_unit_region",
        "confidence": "MEDIUM_HIGH",
        "collect_label": "speed_attack",
        "priority": 4,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 5. GLOBAL SPEED MULTIPLIER — Entire game simulation runs faster
    # ─────────────────────────────────────────────────────────────────────────
    "speed_multiplier": {
        "name": "Global Speed Multiplier",
        "description": (
            "The entire game simulation is accelerated. All animations, timers, "
            "movement speeds, and build/research queues run faster than real-time."
        ),
        "how_to_detect": [
            "All animations visibly faster — walking, attacking, dying are sped up",
            "Resource counters increment faster than documented per-hour generation rate",
            "Build/research queue completes in a fraction of documented time",
            "Unit walking speed exceeds documented maximum",
            "In-game timer (if visible) desyncs from wall-clock time",
        ],
        "thresholds": {
            "speed_multiplier_flag": 1.5,       # flag if any animation > 1.5x calibrated speed
            "walk_speed_max_px_per_frame": 8.0, # calibrated at 30fps; flag if exceeded
            "resource_tick_rate_multiplier": 2.0,# flag if resource ticks > 2x documented rate
            "build_time_fraction_flag": 0.3,    # flag if build completes in <30% of expected time
        },
        "visual_roi": "full_frame",
        "confidence": "MEDIUM",
        "collect_label": "speed_multiplier",
        "priority": 5,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 6. UNLIMITED RESOURCES — Resource counters never decrease
    # ─────────────────────────────────────────────────────────────────────────
    "unlimited_resources": {
        "name": "Unlimited Resources",
        "description": (
            "Client-side resource values (Food, Water, Metal, Fuel) are modified "
            "to prevent depletion. Counter stays at max or resets instantly after spending."
        ),
        "how_to_detect": [
            "Resource counter stays at maximum during active base operations",
            "Counter resets to maximum immediately after a spend event",
            "Player trains an impossible number of troops given base production rate",
            "All four resources static at max for 5+ minutes of active play",
        ],
        "thresholds": {
            "resource_static_window_s": 300.0,  # flag if any resource doesn't change in 5 min of active play
            "instant_refill_threshold_s": 2.0,  # flag if resource refills within 2s of spending
            "impossible_spend_ratio": 10.0,     # flag if spending rate > 10x base production
        },
        "visual_roi": "resource_counter_region",
        "confidence": "MEDIUM",
        "collect_label": "unlimited_resources",
        "priority": 6,
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 7b. BUILDING INSTANT UPGRADE — Dorm / structures upgrade in seconds
    # ─────────────────────────────────────────────────────────────────────────
    "instant_upgrade": {
        "name": "Instant Building Upgrade",
        "description": (
            "Building upgrade (Dorm, Watchtower, etc.) completes in under 30 seconds. "
            "Minimum legitimate time for Dorm 3 is ~2-5 minutes even with max alliance help. "
            "Also detects resource counter dropping then instantly refilling (unlimited resources mod)."
        ),
        "how_to_detect": [
            "Orange/yellow construction shimmer appears on building then vanishes in <30s",
            "Build timer countdown completes impossibly fast",
            "Resource counter (Wood/Stone/Iron) drops on upgrade start then immediately refills",
            "Multiple buildings upgrading simultaneously when only 1-2 queues normally available",
        ],
        "thresholds": {
            "min_legit_upgrade_s": 30.0,       # absolute floor — physically impossible below
            "suspicious_upgrade_s": 60.0,       # flag as suspicious below this
            "resource_refill_frames": 3,         # refill within 3 frames = instant
            "construction_color_hsv": {
                "h_low": 15, "h_high": 35,
                "s_min": 120, "v_min": 160,
            },
        },
        "visual_roi": "base_area_region",
        "confidence": "HIGH",
        "collect_label": "instant_upgrade",
        "priority": 2,
        "game_note": "Dorm 3 normally takes 2-5 min. Resources: Wood/Stone/Iron (NOT Food/Water/Metal/Fuel)",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # 7. ESP / FOG-OF-WAR HACK — Enemies revealed through fog
    # ─────────────────────────────────────────────────────────────────────────
    "esp_fog_hack": {
        "name": "ESP / Fog-of-War Hack",
        "description": (
            "Enemy unit positions, routes, or base layouts are revealed through "
            "the game's fog of war. May appear as colored overlays or cause the "
            "player to act on intelligence they shouldn't have."
        ),
        "how_to_detect": [
            "Colored bounding boxes or markers visible in fogged map regions",
            "Player sends scouts/attacks toward hidden enemy positions with no prior intel",
            "Player retreats from a flank before any visible scout has revealed it",
            "Enemy unit outlines visible through terrain or darkness",
        ],
        "thresholds": {
            "pre_knowledge_response_s": 5.0,    # flag if player reacts to hidden enemy >5s before revealed
            "overlay_box_in_fog_confidence": 0.7, # CNN confidence threshold for box-in-fog detection
        },
        "visual_roi": "minimap_and_fog_region",
        "confidence": "MEDIUM",
        "collect_label": "esp_fog",
        "priority": 7,
    },
}

# ─── Cheat type index for fast lookup ────────────────────────────────────────
CHEAT_NAMES = {k: v["name"] for k, v in DARKWAR_SIGNATURES.items()}
PRIORITY_ORDER = sorted(DARKWAR_SIGNATURES.keys(),
                        key=lambda k: DARKWAR_SIGNATURES[k]["priority"])

# ─── Detection ROI regions (relative, 0.0-1.0 of frame) ─────────────────────
# These are approximate — a calibration step should refine per screen resolution.
ROI_MAP = {
    "hp_bar_region":         {"x": 0.01, "y": 0.88, "w": 0.25, "h": 0.06},
    "enemy_hp_bar_region":   {"x": 0.35, "y": 0.05, "w": 0.30, "h": 0.05},
    "ability_bar_region":    {"x": 0.25, "y": 0.88, "w": 0.50, "h": 0.10},
    "resource_counter_region":{"x": 0.00, "y": 0.00, "w": 1.00, "h": 0.08},
    "combat_unit_region":    {"x": 0.10, "y": 0.10, "w": 0.80, "h": 0.80},
    "minimap_and_fog_region":{"x": 0.00, "y": 0.60, "w": 0.25, "h": 0.40},
    "full_frame":            {"x": 0.00, "y": 0.00, "w": 1.00, "h": 1.00},
}


def describe_all():
    """Print all signatures in human-readable format."""
    print(f"\nDark War Survival — {len(DARKWAR_SIGNATURES)} cheat signatures\n")
    for k in PRIORITY_ORDER:
        sig = DARKWAR_SIGNATURES[k]
        print(f"  [{sig['priority']}] {sig['name']} ({sig['confidence']} confidence)")
        print(f"      {sig['description'][:90]}...")
        print()


if __name__ == "__main__":
    describe_all()
