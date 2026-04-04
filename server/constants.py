# constants.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for all numeric configuration in the environment.
# Nothing in this file is computed — these are fixed values only.
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

INFECTION_THRESHOLD   = 0.40
SAFE_THRESHOLD        = 0.20
LOW_THRESHOLD         = 0.20
HOSPITAL_BREACH_POINT = 0.00


# ── Spread Mechanics ──────────────────────────────────────────────────────────

SPREAD_RATE_MIN       = 0.05    # Slowest possible spread rate per day
SPREAD_RATE_MAX       = 0.13    # Fastest possible spread rate per day
GROWTH_HINT_NOISE     = 0.03    # Noise on growth_rate_hint (± value)

TREATMENT_REDUCTION   = 0.08    # Allocating reduces existing infection this step
ALLOCATE_REDUCTION    = 0.10    # Allocating reduces future spread rate this step
RESTRICT_REDUCTION    = 0.05    # Restricting reduces spread per step
SPILLOVER_RATE        = 0.01    # Infection fraction that spills to adjacent districts

RESOURCE_REPLENISH    = 0       # No replenishment — pool is total budget for episode


# ── Reward Weights ────────────────────────────────────────────────────────────

REWARD_INFECTION_PENALTY       = -0.50
REWARD_HOSPITAL_BREACH         = -1.00
REWARD_EARLY_CONTAINMENT       = +0.50
REWARD_UNNECESSARY_RESTRICTION = -0.20
REWARD_CORRECT_PRIORITISATION  = +0.30


# ── Episode Terminal Conditions ───────────────────────────────────────────────

# Success: ALL districts below SAFE_THRESHOLD → speed bonus fires
# Failure: ANY hospital at HOSPITAL_BREACH_POINT → episode ends immediately
# Natural: max_steps reached