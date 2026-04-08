import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

TASK_CONFIG = {
    "easy":   {"num_districts": 2,  "max_steps": 10, "resource_pool": 10, "data_lag_days": 0},
    "medium": {"num_districts": 4,  "max_steps": 15, "resource_pool": 8,  "data_lag_days": 0},
    "hard":   {"num_districts": 6,  "max_steps": 15, "resource_pool": 7,  "data_lag_days": 3},
}

INFECTION_THRESHOLD   = 0.40   # above this → district is in the danger zone
SAFE_THRESHOLD        = 0.20   # below this → district is considered contained (grader + terminal check)
# LOW_THRESHOLD and SAFE_THRESHOLD are intentionally separate even though both equal 0.20.
# SAFE_THRESHOLD is a containment concept; LOW_THRESHOLD is a restriction-penalty concept.
# Keeping them separate means they can diverge independently if the design changes.
LOW_THRESHOLD         = 0.20   # restricting below this threshold earns a penalty
# real ICU overflow and triage failure kick in well before zero capacity
HOSPITAL_BREACH_POINT = 0.10

SPREAD_RATE_MIN   = 0.03
SPREAD_RATE_MAX   = 0.08
GROWTH_HINT_NOISE = 0.03   # noise added to spread rate before showing agent

# ~1% of active cases resolve per day without intervention; spread still dominates
NATURAL_RECOVERY_RATE = 0.01

TREATMENT_REDUCTION   = 0.05   # allocate reduces existing infection by this amount
ALLOCATE_REDUCTION    = 0.10   # allocate also suppresses future spread rate this step
RESTRICT_REDUCTION    = 0.05   # restrict reduces spread rate while active
SPILLOVER_RATE        = 0.01   # infection that bleeds into adjacent districts each step

RESOURCE_REPLENISH = 1   # units added each step, capped at the task's resource_pool

REWARD_INFECTION_PENALTY       = -0.50
REWARD_HOSPITAL_BREACH         = -1.00
REWARD_EARLY_CONTAINMENT       = +0.50
REWARD_UNNECESSARY_RESTRICTION = -0.20
REWARD_CORRECT_PRIORITISATION  = +0.30
