# constants.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

TASK_CONFIG = {
    "easy":   {"num_districts": 2,  "max_steps": 10, "resource_pool": 10, "data_lag_days": 0},
    "medium": {"num_districts": 4,  "max_steps": 15, "resource_pool": 8,  "data_lag_days": 0},
    "hard":   {"num_districts": 6,  "max_steps": 15, "resource_pool": 7,  "data_lag_days": 3},
}

INFECTION_THRESHOLD   = 0.40
SAFE_THRESHOLD        = 0.20
LOW_THRESHOLD         = 0.20
# Real health systems reach operational collapse before zero capacity.
# ICU overflow and triage failure begin at ~10% remaining capacity.
HOSPITAL_BREACH_POINT = 0.10

SPREAD_RATE_MIN       = 0.03
SPREAD_RATE_MAX       = 0.08
GROWTH_HINT_NOISE     = 0.03

# Natural recovery: ~1% of active cases resolve per day without intervention.
# Reflects a realistic epidemic where spread dominates unless actively managed.
# Infection grows without intervention, but sustained allocation can drive it below threshold.
NATURAL_RECOVERY_RATE = 0.01

# Treatment reduces existing infection by 5% per allocation action.
# Medical deployment (antivirals, PPE, rapid response teams) realistic at this scale.
TREATMENT_REDUCTION   = 0.05
ALLOCATE_REDUCTION    = 0.10    # Reduces future spread rate this step
RESTRICT_REDUCTION    = 0.05    # Restricting reduces spread per step
SPILLOVER_RATE        = 0.01    # Infection spilling to adjacent districts

RESOURCE_REPLENISH    = 1       # Resource units restored each step (capped at pool)

REWARD_INFECTION_PENALTY       = -0.50
REWARD_HOSPITAL_BREACH         = -1.00
REWARD_EARLY_CONTAINMENT       = +0.50
REWARD_UNNECESSARY_RESTRICTION = -0.20
REWARD_CORRECT_PRIORITISATION  = +0.30

# Success:  ALL districts below SAFE_THRESHOLD → speed bonus fires
# Failure:  ANY hospital at or below HOSPITAL_BREACH_POINT → episode ends
# Natural:  max_steps reached