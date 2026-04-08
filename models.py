from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


@dataclass
class DistrictObservation:
    district_id:                  int
    reported_infection_rate:      float   # lagged in hard task, real-time otherwise
    growth_rate_hint:             float   # noisy estimate of true spread rate
    hospital_capacity_remaining:  float   # 0.0 = overwhelmed, 1.0 = full capacity
    population_density:           float   # this district's share of total city population
    tested_recently:              bool    # true if tested within the last 2 days
    restriction_active:           bool    # true if movement restrictions are active


@dataclass
class DistrictTruth:
    district_id:                  int
    true_infection_rate:          float   # actual rate used by the grader, never sent to agent
    true_spread_rate:             float   # fixed for the episode, agent never observes this directly
    hospital_capacity_remaining:  float
    population_density:           float
    days_since_tested:            int
    restriction_active:           bool
    deployed_resources:           int     # resource units allocated this step


# CityState is the hidden simulation truth.
# It is NOT a subclass of State — environment.py maintains a separate
# State(episode_id, step_count) for OpenEnv tracking alongside this.
@dataclass
class CityState:
    day:                  int                  = 0
    available_resources:  int                  = 0
    task_name:            str                  = "easy"
    data_lag_days:        int                  = 0
    max_steps:            int                  = 10
    districts:            List[DistrictTruth]  = field(default_factory=list)
    infection_history:    List[List[float]]    = field(default_factory=list)


class ContainmentAction(Action):
    """
    One action per step. action_type must be one of:
      'test'     — spend 1 resource to get accurate district data
      'restrict' — impose movement restrictions (penalised if infection is already low)
      'allocate' — deploy 1 resource to reduce existing infection and slow spread
    """
    action_type:  str = Field(..., description="One of: 'test', 'restrict', 'allocate'")
    district_id:  int = Field(..., description="Target district (0-indexed)")


# done and reward come from the Observation base class — do not redeclare them here.
class CityObservation(Observation):
    districts:            List[DistrictObservation] = Field(...,  description="Per-district state visible to agent")
    available_resources:  int                       = Field(...,  description="Resource units remaining this turn")
    current_step:         int                       = Field(...,  description="Current step number")
    max_steps:            int                       = Field(...,  description="Total steps allowed this episode")
    message:              Optional[str]             = Field(None, description="Feedback string for debugging")
