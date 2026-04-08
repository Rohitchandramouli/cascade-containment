import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import copy
from uuid import uuid4
from typing import Optional, Tuple

from server.grader import TrajectoryStep
from openenv.core.env_server.types import State
from openenv.core.env_server.interfaces import Environment

from models import (
    CityState,
    CityObservation,
    ContainmentAction,
)
from server.constants import (
    HOSPITAL_BREACH_POINT,
    TASK_CONFIG,
    INFECTION_THRESHOLD,
    SAFE_THRESHOLD,
    LOW_THRESHOLD,
    ALLOCATE_REDUCTION,
    RESTRICT_REDUCTION,
    TREATMENT_REDUCTION,
    RESOURCE_REPLENISH,
    REWARD_INFECTION_PENALTY,
    REWARD_HOSPITAL_BREACH,
    REWARD_EARLY_CONTAINMENT,
    REWARD_UNNECESSARY_RESTRICTION,
    REWARD_CORRECT_PRIORITISATION,
)
from server.utils import (
    build_observation,
    compute_spread,
    get_highest_infected_district,
    all_districts_contained,
    any_hospital_breached,
    districts_above_threshold,
    snapshot_infection_rates,
    generate_episode_id,
)
from server.tasks.registry import get_task
from server.grader import grade_trajectory

_last_grade: dict = {}


class EpidemicContainmentEnv(Environment):
    """
    Cascade Containment — RL environment for epidemic response.

    The agent acts as a city health authority making sequential resource
    allocation decisions under uncertainty and potentially delayed data.

    Interface:
        reset(task_name)  → CityObservation
        step(action)      → CityObservation
        state             → State (property)
    """

    def __init__(self):
        self._city:       CityState = CityState()
        self._state:      State     = State(episode_id=str(uuid4()), step_count=0)
        self._task_name:  str       = "easy"
        self._trajectory: list      = []

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self, task_name: str = "easy") -> CityObservation:
        self._task_name = task_name
        task            = get_task(task_name)
        self._city      = task.build_initial_state()
        self._state     = State(episode_id=generate_episode_id(), step_count=0)
        self._trajectory = []

        return build_observation(
            state      = self._city,
            step_count = self._state.step_count,
            reward     = None,
            message    = (
                f"Episode started. Task: {task_name}. "
                f"Districts: {len(self._city.districts)}, "
                f"Steps: {self._city.max_steps} available."
            ),
            done       = False,
        )

    def step(self, action: ContainmentAction) -> CityObservation:
        assert self._city  is not None, "Call reset() before step()."
        assert self._state is not None, "Call reset() before step()."

        action, message = self._validate_action(action)

        self._city.infection_history.append(
            snapshot_infection_rates(self._city.districts)
        )

        self._apply_action(action)

        new_rates = compute_spread(self._city.districts)
        for i, district in enumerate(self._city.districts):
            district.true_infection_rate = new_rates[i]

        self._update_hospital_capacity()

        self._city.available_resources = min(
            self._city.available_resources + RESOURCE_REPLENISH,
            TASK_CONFIG[self._task_name]["resource_pool"],
        )

        for district in self._city.districts:
            district.deployed_resources = 0

        self._city.day         += 1
        self._state.step_count += 1

        reward = self._compute_reward(action)

        self._trajectory.append(TrajectoryStep(
            step       = self._state.step_count,
            city_state = copy.deepcopy(self._city),
            action     = action,
            reward     = reward,
            done       = False,
        ))

        done, terminal_message = self._check_terminal()

        if done and self._trajectory:
            self._trajectory[-1].done = True
            import server.environment as _self_module
            result = grade_trajectory(self._trajectory, self._task_name)
            _self_module._last_grade = {
                "final_score":         result.final_score,
                "containment_score":   result.containment_score,
                "hospital_score":      result.hospital_score,
                "efficiency_score":    result.efficiency_score,
                "speed_score":         result.speed_score,
                "hospital_breached":   result.hospital_breached,
                "districts_contained": result.districts_contained,
                "total_steps":         result.total_steps,
                "task_name":           self._task_name,
            }

        return build_observation(
            state      = self._city,
            step_count = self._state.step_count,
            reward     = reward,
            message    = terminal_message if terminal_message else message,
            done       = done,
        )

    @property
    def state(self) -> State:
        return self._state

    def get_trajectory(self) -> list:
        return self._trajectory

    # ── Action handling ───────────────────────────────────────────────────────

    def _validate_action(
        self, action: ContainmentAction
    ) -> Tuple[ContainmentAction, str]:
        """
        Invalid actions are replaced with a safe default rather than raising —
        the episode must continue even when the LLM returns malformed output.
        """
        valid_types   = {"test", "restrict", "allocate"}
        num_districts = len(self._city.districts)

        if action.action_type not in valid_types:
            return (
                ContainmentAction(action_type="allocate", district_id=0),
                f"Invalid action_type '{action.action_type}'. Defaulted to allocate on district 0.",
            )

        if not (0 <= action.district_id < num_districts):
            safe_id = max(0, min(action.district_id, num_districts - 1))
            return (
                ContainmentAction(action_type=action.action_type, district_id=safe_id),
                f"district_id {action.district_id} out of range. Clamped to {safe_id}.",
            )

        if action.action_type in {"test", "allocate"} and self._city.available_resources <= 0:
            return (
                ContainmentAction(action_type="restrict", district_id=action.district_id),
                f"No resources left. Switched to restrict on district {action.district_id}.",
            )

        return action, f"{action.action_type.capitalize()} on district {action.district_id}."

    def _apply_action(self, action: ContainmentAction) -> None:
        district = self._city.districts[action.district_id]

        if action.action_type == "test":
            district.days_since_tested      = 0
            self._city.available_resources -= 1

        elif action.action_type == "restrict":
            district.restriction_active = True
            district.days_since_tested += 1

        elif action.action_type == "allocate":
            district.deployed_resources    += 1
            district.true_infection_rate    = max(0.0, district.true_infection_rate - TREATMENT_REDUCTION)
            self._city.available_resources -= 1
            district.days_since_tested     += 1

        for d in self._city.districts:
            if d.district_id != action.district_id:
                d.days_since_tested += 1

    # ── Simulation mechanics ──────────────────────────────────────────────────

    def _update_hospital_capacity(self) -> None:
        for district in self._city.districts:
            if district.true_infection_rate > INFECTION_THRESHOLD:
                excess = district.true_infection_rate - INFECTION_THRESHOLD
                drain  = round(excess * 0.25, 4)
                district.hospital_capacity_remaining = max(
                    0.0,
                    district.hospital_capacity_remaining - drain
                )
            else:
                district.hospital_capacity_remaining = min(
                    1.0,
                    district.hospital_capacity_remaining + 0.02
                )
                if district.true_infection_rate < SAFE_THRESHOLD:
                    district.restriction_active = False

    def _compute_reward(self, action: ContainmentAction) -> float:
        reward = 0.0

        for district in districts_above_threshold(self._city.districts):
            density_weight = max(0.5, district.population_density * len(self._city.districts))
            reward += REWARD_INFECTION_PENALTY * min(2.0, density_weight)

        for district in self._city.districts:
            if district.hospital_capacity_remaining <= HOSPITAL_BREACH_POINT:
                reward += REWARD_HOSPITAL_BREACH

        for district in self._city.districts:
            if district.true_infection_rate < SAFE_THRESHOLD:
                time_factor = 1 - (self._state.step_count / self._city.max_steps)
                reward += REWARD_EARLY_CONTAINMENT * time_factor

        if action.action_type == "restrict":
            target = self._city.districts[action.district_id]
            if target.true_infection_rate < LOW_THRESHOLD:
                reward += REWARD_UNNECESSARY_RESTRICTION

        if action.action_type == "allocate":
            if action.district_id == get_highest_infected_district(self._city.districts):
                reward += REWARD_CORRECT_PRIORITISATION

        return round(reward, 4)

    def _check_terminal(self) -> Tuple[bool, Optional[str]]:
        if all_districts_contained(self._city.districts):
            return True, "✓ Outbreak contained. All districts below safe threshold."

        if any_hospital_breached(self._city.districts):
            return True, "✗ Hospital capacity breached. Episode failed."

        if self._state.step_count >= self._city.max_steps:
            return True, f"Episode complete. {self._city.max_steps} steps reached."

        return False, None
