from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ── District-level view (visible to agent) ────────────────────────────────────

@dataclass
class DistrictObservation:
    district_id:                  int
    reported_infection_rate:      float   # Lagged in hard task; real-time otherwise
    growth_rate_hint:             float   # Noisy signal of true spread rate
    hospital_capacity_remaining:  float   # 0.0 = overwhelmed, 1.0 = fully available
    population_density:           float   # Fraction of city population in this district
    tested_recently:              bool    # True if tested within last 2 days
    restriction_active:           bool    # True if movement restriction is in place


# ── District-level ground truth (hidden from agent) ───────────────────────────

@dataclass
class DistrictTruth:
    district_id:                  int
    true_infection_rate:          float   # Actual infection rate used by grader
    true_spread_rate:             float   # Fixed per episode; agent never sees this
    hospital_capacity_remaining:  float
    population_density:           float
    days_since_tested:            int
    restriction_active:           bool
    deployed_resources:           int     # Resource units currently active here


# ── City state (internal world truth; never sent to agent) ────────────────────
# Not a subclass of State — stored internally in environment.py alongside
# a plain State(episode_id=..., step_count=...) for OpenEnv tracking.

@dataclass
class CityState:
    day:                  int                  = 0
    available_resources:  int                  = 0
    task_name:            str                  = "easy"
    data_lag_days:        int                  = 0
    max_steps:            int                  = 10
    districts:            List[DistrictTruth]  = field(default_factory=list)
    infection_history:    List[List[float]]    = field(default_factory=list)


# ── Action (sent by agent each step) ─────────────────────────────────────────

class ContainmentAction(Action):
    """
    One action per step. action_type must be one of:
      'test'      — Spend 1 resource for accurate district infection data
      'restrict'  — Impose movement restriction (penalised if infection is low)
      'allocate'  — Deploy 1 resource unit to reduce spread rate this step
    """
    action_type:  str = Field(..., description="One of: 'test', 'restrict', 'allocate'")
    district_id:  int = Field(..., description="Target district (0-indexed)")


# ── Observation (received by agent each step) ─────────────────────────────────
# done and reward are inherited from Observation — do not redeclare them.

class CityObservation(Observation):
    districts:            List[DistrictObservation] = Field(...,  description="Per-district state visible to agent")
    available_resources:  int                       = Field(...,  description="Resource units remaining this turn")
    current_step:         int                       = Field(...,  description="Current step in the episode")
    max_steps:            int                       = Field(...,  description="Total steps allowed this episode")
    message:              Optional[str]             = Field(None, description="Human-readable feedback for debugging")