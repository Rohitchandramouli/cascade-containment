# server/grader.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List
from dataclasses import dataclass
from models import CityState, ContainmentAction
from server.constants import (
    INFECTION_THRESHOLD,
    SAFE_THRESHOLD,
    HOSPITAL_BREACH_POINT,
    TREATMENT_REDUCTION,
    TASK_CONFIG,
)


@dataclass
class TrajectoryStep:
    step:          int
    city_state:    CityState
    action:        ContainmentAction
    reward:        float
    done:          bool


@dataclass
class GradeResult:
    final_score:          float
    containment_score:    float
    hospital_score:       float
    efficiency_score:     float
    speed_score:          float
    hospital_breached:    bool
    districts_contained:  int
    total_steps:          int


def grade_trajectory(trajectory: List[TrajectoryStep], task_name: str) -> GradeResult:
    if not trajectory:
        return GradeResult(0.0, 0.0, 0.0, 0.0, 0.0, False, 0, 0)

    config        = TASK_CONFIG[task_name]
    num_districts = config["num_districts"]
    max_steps     = config["max_steps"]
    total_steps   = len(trajectory)

    # ── Component 1: Containment Score ───────────────────────────────────────
    # Fraction of district-days that stayed below infection threshold.
    # Skips first 2 steps (initial conditions outside agent's control).
    safe_district_days  = 0
    for step in trajectory[2:]:
        for district in step.city_state.districts:
            if district.true_infection_rate <= INFECTION_THRESHOLD:
                safe_district_days += 1
    total_district_days = max(len(trajectory) - 2, 1) * num_districts
    containment_score   = safe_district_days / total_district_days

    # ── Component 2: Hospital Score ───────────────────────────────────────────
    # Average capacity preserved. Breach multiplier of 0.6 if any district collapsed.
    hospital_breached       = False
    total_capacity_preserved = 0.0
    for step in trajectory:
        for district in step.city_state.districts:
            if district.hospital_capacity_remaining <= HOSPITAL_BREACH_POINT:
                hospital_breached = True
            total_capacity_preserved += district.hospital_capacity_remaining
    hospital_district_days = total_steps * num_districts
    avg_capacity   = total_capacity_preserved / hospital_district_days
    hospital_score = round(min(1.0, max(0.0, avg_capacity * (0.6 if hospital_breached else 1.0))), 4)

    # ── Component 3: Efficiency Score ────────────────────────────────────────
    # Fraction of resource actions that targeted the right district.
    # Uses the PREVIOUS step's infection rates (pre-action state) so that
    # successful treatments are not penalised retroactively.
    correct_actions = 0
    total_resource  = 0
    for idx, step in enumerate(trajectory):
        if step.action.action_type not in {"allocate", "test"}:
            continue
        total_resource += 1
        # Determine pre-action infection state
        if idx > 0:
            prev_districts  = trajectory[idx - 1].city_state.districts
            pre_action_rate = prev_districts[step.action.district_id].true_infection_rate
            highest_before  = max(prev_districts, key=lambda d: d.true_infection_rate).district_id
        else:
            # First step: estimate pre-action rate from post-action + treatment
            curr_d          = step.city_state.districts[step.action.district_id]
            pre_action_rate = curr_d.true_infection_rate + TREATMENT_REDUCTION
            highest_before  = max(step.city_state.districts, key=lambda d: d.true_infection_rate).district_id
        # Correct if the targeted district was above threshold before action,
        # or if it was the most infected district at the time
        if pre_action_rate > INFECTION_THRESHOLD or step.action.district_id == highest_before:
            correct_actions += 1
    efficiency_score = correct_actions / max(total_resource, 1)

    # ── Component 4: Speed Score ──────────────────────────────────────────────
    # Reward early containment. Only fires if episode ended before max_steps.
    last_step   = trajectory[-1]
    speed_score = round(max(0.0, 1.0 - total_steps / max_steps), 4) \
                  if last_step.done and total_steps < max_steps else 0.0

    # ── Final Weighted Score ──────────────────────────────────────────────────
    # Hospital (45%) is the primary constraint — system collapse is catastrophic.
    # Containment (30%) — keeping infection below dangerous levels.
    # Efficiency (15%) — quality of resource allocation decisions.
    # Speed (10%) — tiebreaker rewarding proactive early containment.
    final_score = round(min(1.0, max(0.0,
        containment_score * 0.30 +
        hospital_score    * 0.45 +
        efficiency_score  * 0.15 +
        speed_score       * 0.10
    )), 4)

    districts_contained = sum(
        1 for d in trajectory[-1].city_state.districts
        if d.true_infection_rate < SAFE_THRESHOLD
    )

    return GradeResult(
        final_score         = final_score,
        containment_score   = round(containment_score, 4),
        hospital_score      = hospital_score,
        efficiency_score    = round(efficiency_score, 4),
        speed_score         = speed_score,
        hospital_breached   = hospital_breached,
        districts_contained = districts_contained,
        total_steps         = total_steps,
    )


def grade_task(trajectory: List[TrajectoryStep], task_name: str) -> float:
    return grade_trajectory(trajectory, task_name).final_score