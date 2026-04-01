# constants.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for all numeric configuration in the environment.
# Nothing in this file is computed — these are fixed values only.
# Adjust reward weights here during tuning without touching environment.py.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Task Configuration ────────────────────────────────────────────────────────

TASK_CONFIG = {
    "easy": {
        "num_districts":    2,
        "max_steps":        10,
        "resource_pool":    10,     # Resources available per episode
        "data_lag_days":    0,      # Agent sees real-time infection data
    },
    "medium": {
        "num_districts":    4,
        "max_steps":        15,
        "resource_pool":    8,      # Tighter budget forces real tradeoffs
        "data_lag_days":    0,
    },
    "hard": {
        "num_districts":    6,
        "max_steps":        20,
        "resource_pool":    7,      # Scarce resources + delayed data
        "data_lag_days":    3,      # Agent sees infection rates from 3 days ago
    },
}


# ── Infection Thresholds ──────────────────────────────────────────────────────

INFECTION_THRESHOLD   = 0.40    # Above this → district is in danger (penalty fires)
SAFE_THRESHOLD        = 0.20    # Below this → district is contained (bonus fires)
LOW_THRESHOLD         = 0.20    # Below this → restriction is deemed unnecessary
HOSPITAL_BREACH_POINT = 0.00    # At or below this → hospital has collapsed


# ── Spread Mechanics ──────────────────────────────────────────────────────────

SPREAD_RATE_MIN       = 0.05    # Slowest possible true spread rate per day
SPREAD_RATE_MAX       = 0.20    # Fastest possible true spread rate per day
GROWTH_HINT_NOISE     = 0.03    # Random noise added to growth_rate_hint (± value)

ALLOCATE_REDUCTION    = 0.10    # How much one 'allocate' reduces spread this step
RESTRICT_REDUCTION    = 0.05    # How much one 'restrict' reduces spread per step
SPILLOVER_RATE        = 0.02    # Fraction of infection that spreads to adjacent districts per day

RESOURCE_REPLENISH    = 3       # Resource units restored at the start of each new day


# ── Reward Weights ────────────────────────────────────────────────────────────

REWARD_INFECTION_PENALTY      = -0.50   # Per district above INFECTION_THRESHOLD each step
REWARD_HOSPITAL_BREACH        = -1.00   # Per district with breached hospital capacity
REWARD_EARLY_CONTAINMENT      = +0.50   # Base value; scaled by (1 - step/max_steps)
REWARD_UNNECESSARY_RESTRICTION = -0.20  # Restricting a district below LOW_THRESHOLD
REWARD_CORRECT_PRIORITISATION = +0.30   # Allocating to the highest-infected district


# ── Episode Terminal Conditions ───────────────────────────────────────────────

# Episode ends early (success) if ALL districts drop below SAFE_THRESHOLD.
# Episode ends early (failure) if ANY district's hospital capacity hits HOSPITAL_BREACH_POINT.
# Otherwise episode runs until max_steps is reached.