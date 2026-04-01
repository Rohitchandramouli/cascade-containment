# server/grader.py
# ─────────────────────────────────────────────────────────────────────────────
# Deterministic scorer for completed Cascade Containment episodes.
# Called by baseline/evaluator.py after each full episode.
# Always returns a float in [0.0, 1.0].
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Tuple
from dataclasses import dataclass

from models import CityState, ContainmentAction
from server.constants import (
    INFECTION_THRESHOLD,
    SAFE_THRESHOLD,
    HOSPITAL_BREACH_POINT,
    TASK_CONFIG,
)


# ── Trajectory Record ─────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """
    A single recorded step in an episode.
    Stored by environment.py and passed to the grader after episode ends.
    """
    step:          int
    city_state:    CityState           # Hidden ground truth at this step
    action:        ContainmentAction   # What the agent did
    reward:        float               # Reward received
    done:          bool                # Was this the final step


# ── Grader Score Breakdown ────────────────────────────────────────────────────

@dataclass
class GradeResult:
    """
    Full scoring breakdown for one episode.
    The final_score is what the evaluator reports.
    """
    final_score:          float   # Weighted composite: 0.0 to 1.0
    containment_score:    float   # How well infection was kept below threshold
    hospital_score:       float   # How well hospital capacity was preserved
    efficiency_score:     float   # How well resources were directed
    speed_score:          float   # How quickly the episode was resolved
    hospital_breached:    bool    # Whether any hospital collapse occurred
    districts_contained:  int     # How many districts ended below safe threshold
    total_steps:          int     # Steps taken before episode ended


# ── Main Grader ───────────────────────────────────────────────────────────────

def grade_trajectory(
    trajectory: List[TrajectoryStep],
    task_name:  str,
) -> GradeResult:
    """
    Score a completed episode trajectory.

    Args:
        trajectory: Ordered list of TrajectoryStep from one full episode.
        task_name:  "easy", "medium", or "hard" — affects scoring strictness.

    Returns:
        GradeResult with final_score in [0.0, 1.0] and full breakdown.
    """
    if not trajectory:
        return GradeResult(
            final_score         = 0.0,
            containment_score   = 0.0,
            hospital_score      = 0.0,
            efficiency_score    = 0.0,
            speed_score         = 0.0,
            hospital_breached   = False,
            districts_contained = 0,
            total_steps         = 0,
        )

    config       = TASK_CONFIG[task_name]
    num_districts = config["num_districts"]
    max_steps     = config["max_steps"]
    total_steps   = len(trajectory)

    # ── Component 1: Containment Score ───────────────────────────────────────
    # Fraction of district-days that stayed below infection threshold.
    # Perfect agent = 1.0 (no district ever exceeded threshold).

    total_district_days   = total_steps * num_districts
    safe_district_days    = 0

    for step in trajectory:
        for district in step.city_state.districts:
            if district.true_infection_rate <= INFECTION_THRESHOLD:
                safe_district_days += 1

    containment_score = safe_district_days / total_district_days

    # ── Component 2: Hospital Score ───────────────────────────────────────────
    # Measures how well hospital capacity was preserved across the episode.
    # Any breach = heavy penalty. Near-breach is also penalised proportionally.

    hospital_breached       = False
    total_capacity_preserved = 0.0

    for step in trajectory:
        for district in step.city_state.districts:
            if district.hospital_capacity_remaining <= HOSPITAL_BREACH_POINT:
                hospital_breached = True
            total_capacity_preserved += district.hospital_capacity_remaining

    avg_capacity     = total_capacity_preserved / total_district_days
    hospital_score   = avg_capacity * (0.3 if hospital_breached else 1.0)
    hospital_score   = round(min(1.0, max(0.0, hospital_score)), 4)

    # ── Component 3: Efficiency Score ────────────────────────────────────────
    # Fraction of allocate/test actions that targeted districts above threshold.
    # Rewards directing resources where they're actually needed.

    resource_actions  = [
        s for s in trajectory
        if s.action.action_type in {"allocate", "test"}
    ]

    if resource_actions:
        correct_actions = 0
        for step in resource_actions:
            target = step.city_state.districts[step.action.district_id]
            if target.true_infection_rate > INFECTION_THRESHOLD:
                correct_actions += 1
        efficiency_score = correct_actions / len(resource_actions)
    else:
        efficiency_score = 0.5  # Neutral if no resource actions taken

    # ── Component 4: Speed Score ──────────────────────────────────────────────
    # Rewards finishing faster than max_steps.
    # If episode ran to max_steps, speed_score = 0.0.
    # If contained in half the steps, speed_score = 0.5. Etc.

    last_step = trajectory[-1]
    if last_step.done and not hospital_breached:
        speed_score = round(1.0 - (total_steps / max_steps), 4)
        speed_score = max(0.0, speed_score)
    else:
        speed_score = 0.0   # No speed bonus for failed or incomplete episodes

    # ── Final Weighted Score ──────────────────────────────────────────────────
    # Weights reflect judging priorities:
    #   containment = primary signal
    #   hospital    = safety constraint
    #   efficiency  = quality differentiator
    #   speed       = tiebreaker

    final_score = (
        containment_score  * 0.45 +
        hospital_score     * 0.30 +
        efficiency_score   * 0.15 +
        speed_score        * 0.10
    )
    final_score = round(min(1.0, max(0.0, final_score)), 4)

    # ── Final district count ──────────────────────────────────────────────────
    final_step          = trajectory[-1]
    districts_contained = sum(
        1 for d in final_step.city_state.districts
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


# ── Convenience: Grade a Single Score to 0.0–1.0 ─────────────────────────────

def grade_task(
    trajectory: List[TrajectoryStep],
    task_name:  str,
) -> float:
    """
    Thin wrapper that returns just the final_score float.
    Used by baseline/evaluator.py for clean score reporting.
    """
    result = grade_trajectory(trajectory, task_name)
    return result.final_score