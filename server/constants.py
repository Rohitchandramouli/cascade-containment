# constants.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for all numeric configuration in the environment.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Task Configuration ────────────────────────────────────────────────────────

TASK_CONFIG = {
    "easy": {
        "num_districts":    2,
        "max_steps":        10,
        "resource_pool":    10,
        "data_lag_days":    0,
    },
    "medium": {
        "num_districts":    4,
        "max_steps":        15,
        "resource_pool":    8,
        "data_lag_days":    0,
    },
    "hard": {
        "num_districts":    6,
        "max_steps":        15,
        "resource_pool":    7,
        "data_lag_days":    3,
    },
}


# ── Infection Thresholds ──────────────────────────────────────────────────────

INFECTION_THRESHOLD   = 0.40    # Above this → district is in danger
SAFE_THRESHOLD        = 0.20    # Below this → district is contained
LOW_THRESHOLD         = 0.20    # Below this → restriction is unnecessary

# Real health systems reach operational collapse before zero capacity.
# ICU overflow, staff exhaustion and triage failure begin at ~10% remaining.
HOSPITAL_BREACH_POINT = 0.10    # ≤ 10% capacity remaining = collapse


# ── Spread Mechanics ──────────────────────────────────────────────────────────

SPREAD_RATE_MIN       = 0.03
SPREAD_RATE_MAX       = 0.08
GROWTH_HINT_NOISE     = 0.03

# Natural recovery: ~2% of active cases resolve per day without intervention.
# Reflects a realistic R-effective slightly above 1 in an uncontrolled outbreak.
# Infection grows without intervention but not explosively — it needs active
# management to be pushed below safe threshold.
NATURAL_RECOVERY_RATE = 0.02

TREATMENT_REDUCTION   = 0.10    # Allocating reduces existing infection this step
ALLOCATE_REDUCTION    = 0.10    # Allocating reduces future spread rate this step
RESTRICT_REDUCTION    = 0.05    # Restricting reduces spread per step
SPILLOVER_RATE        = 0.01    # Infection fraction spilling to adjacent districts

RESOURCE_REPLENISH    = 1       # Resource units restored each step (capped at pool)


# ── Reward Weights ────────────────────────────────────────────────────────────

REWARD_INFECTION_PENALTY       = -0.50
REWARD_HOSPITAL_BREACH         = -1.00
REWARD_EARLY_CONTAINMENT       = +0.50
REWARD_UNNECESSARY_RESTRICTION = -0.20
REWARD_CORRECT_PRIORITISATION  = +0.30


# ── Episode Terminal Conditions ───────────────────────────────────────────────

# Success:  ALL districts below SAFE_THRESHOLD → speed bonus fires
# Failure:  ANY hospital at or below HOSPITAL_BREACH_POINT → episode ends
# Natural:  max_steps reached