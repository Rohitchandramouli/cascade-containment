# server/environment.py
# ─────────────────────────────────────────────────────────────────────────────
# Core RL environment for Cascade Containment.
# Implements the three-method OpenEnv interface: reset(), step(), state().
# Maintains two objects: OpenEnv State (episode tracking) and CityState
# (city simulation). The agent only ever sees CityObservation.
# ─────────────────────────────────────────────────────────────────────────────


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from uuid import uuid4
from typing import Optional, Tuple

import copy
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
    Cascade Containment — an RL environment for epidemic response policy.

    The agent plays a city health authority making sequential resource
    allocation decisions under uncertainty and delayed feedback.

    Interface:
        reset(task_name)  → CityObservation
        step(action)      → CityObservation
        state()           → State
    """

    def __init__(self):
        self._city:      CityState = CityState()
        self._state:     State      = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str       = "easy"
        self._trajectory: list       = []

    # ── Public Interface ──────────────────────────────────────────────────────

    def reset(self, task_name: str = "easy") -> CityObservation:
        """
        Start a new episode. Initialises city state from the chosen task
        and returns the first observation. Agent sees no reward on reset.
        """
        self._task_name = task_name
        task            = get_task(task_name)

        # Build fresh city state from task definition
        self._city = task.build_initial_state()

        # Initialise OpenEnv State for episode tracking
        self._state = State(
            episode_id  = generate_episode_id(),
            step_count  = 0,
        )

        self._trajectory = []

        return build_observation(
            state       = self._city,
            step_count  = self._state.step_count,
            reward      = None,
            message = (f"Episode started. Task: {task_name}. "
                       f"Districts: {len(self._city.districts)}, "
                       f"Steps: {self._city.max_steps} available."),
            done        = False,
        )

    def step(self, action: ContainmentAction) -> CityObservation:
        """
        Apply the agent's action, advance the simulation by one day,
        and return the resulting observation with reward signal.
        """
        assert self._city  is not None, "Call reset() before step()."
        assert self._state is not None, "Call reset() before step()."

        # ── 1. Validate action ────────────────────────────────────────────────
        action, message = self._validate_action(action)

        # ── 2. Snapshot infection rates into history (before updating) ────────
        self._city.infection_history.append(
            snapshot_infection_rates(self._city.districts)
        )

        # ── 3. Apply action effect to city state ──────────────────────────────
        self._apply_action(action)

        # ── 4. Advance spread dynamics by one day ─────────────────────────────
        new_rates = compute_spread(self._city.districts)
        for i, district in enumerate(self._city.districts):
            district.true_infection_rate = new_rates[i]

        # ── 5. Update hospital capacity based on infection levels ─────────────
        self._update_hospital_capacity()

        # ── 6. Replenish resources at start of each new day ───────────────────
        self._city.available_resources = min(
            self._city.available_resources + RESOURCE_REPLENISH,
            TASK_CONFIG[self._task_name]["resource_pool"],  # Cap at task pool size
        )

        # ── 7. Reset deployed resources (allocate effect lasts one step) ──────
        for district in self._city.districts:
            district.deployed_resources = 0

        # ── 8. Increment counters ─────────────────────────────────────────────
        self._city.day        += 1
        self._state.step_count += 1

        # ── 9. Compute reward ─────────────────────────────────────────────────
        reward = self._compute_reward(action)

        # Record step for grader
        self._trajectory.append(TrajectoryStep(
            step       = self._state.step_count,
            city_state = copy.deepcopy(self._city),
            action     = action,
            reward     = reward,
            done       = False,   
        ))

        # ── 10. Check terminal conditions ─────────────────────────────────────
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
        # ── 11. Build and return observation ──────────────────────────────────
        final_message = terminal_message if terminal_message else message

        return build_observation(
            state      = self._city,
            step_count = self._state.step_count,
            reward     = reward,
            message    = final_message,
            done       = done,
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Private: Action Handling ──────────────────────────────────────────────

    def _validate_action(
        self, action: ContainmentAction
    ) -> Tuple[ContainmentAction, str]:
        """
        Validate the action and handle edge cases gracefully.
        Invalid actions are replaced with a safe default rather than crashing —
        this ensures the episode continues even if the LLM produces bad output.
        """
        valid_types     = {"test", "restrict", "allocate"}
        num_districts   = len(self._city.districts)

        # Fix invalid action_type
        if action.action_type not in valid_types:
            return ContainmentAction(action_type="allocate", district_id=0), \
                   f"Invalid action_type '{action.action_type}'. Defaulted to allocate on district 0."

        # Fix out-of-range district_id
        if not (0 <= action.district_id < num_districts):
            safe_id = max(0, min(action.district_id, num_districts - 1))
            return ContainmentAction(action_type=action.action_type, district_id=safe_id), \
                   f"district_id {action.district_id} out of range. Clamped to {safe_id}."

        # Handle resource exhaustion — fall back to restrict (free action)
        if action.action_type in {"test", "allocate"} and self._city.available_resources <= 0:
            return ContainmentAction(action_type="restrict", district_id=action.district_id), \
                   f"No resources left. Action changed to restrict on district {action.district_id}."

        return action, f"{action.action_type.capitalize()} on district {action.district_id}."

    def _apply_action(self, action: ContainmentAction) -> None:
        """Apply the validated action's effect to the city state."""
        district = self._city.districts[action.district_id]

        if action.action_type == "test":
            # Reveal accurate data (handled in build_observation via days_since_tested)
            district.days_since_tested      = 0
            self._city.available_resources -= 1

        elif action.action_type == "restrict":
            # Toggle restriction state
            district.restriction_active = True
            district.days_since_tested += 1

        elif action.action_type == "allocate":
            district.deployed_resources    += 1
            district.true_infection_rate    = max(0.0, district.true_infection_rate - TREATMENT_REDUCTION)
            self._city.available_resources -= 1
            district.days_since_tested     += 1

        # Increment days_since_tested for all non-targeted districts
        for d in self._city.districts:
            if d.district_id != action.district_id:
                d.days_since_tested += 1

    # ── Private: Simulation Mechanics ────────────────────────────────────────

    def _update_hospital_capacity(self) -> None:
        """
        Reduce hospital capacity in districts above the infection threshold.
        High infection consumes capacity faster. Recovery is slow.
        """
        for district in self._city.districts:
            if district.true_infection_rate > INFECTION_THRESHOLD:
                # Capacity drains proportional to how far above threshold
                excess   = district.true_infection_rate - INFECTION_THRESHOLD
                drain = round(excess * 0.25, 4)
                district.hospital_capacity_remaining = max(
                    0.0,
                    district.hospital_capacity_remaining - drain
                )
            else:
                # Slow recovery when infection is below threshold
                district.hospital_capacity_remaining = min(
                    1.0,
                    district.hospital_capacity_remaining + 0.02
                )

    # ── Private: Reward Computation ───────────────────────────────────────────

    def _compute_reward(self, action: ContainmentAction) -> float:
        """
        Compute the shaped reward signal for the current step.
        All five reward terms fire independently each step.
        """
        reward = 0.0

        # Term 1: Penalty for each district above danger threshold
        for district in districts_above_threshold(self._city.districts):
            reward += REWARD_INFECTION_PENALTY

        # Term 2: Heavy penalty for hospital capacity breach
        for district in self._city.districts:
            if district.hospital_capacity_remaining <= HOSPITAL_BREACH_POINT:
                reward += REWARD_HOSPITAL_BREACH

        # Term 3: Early containment bonus (decays over time)
        for district in self._city.districts:
            if district.true_infection_rate < SAFE_THRESHOLD:
                time_factor = 1 - (self._state.step_count / self._city.max_steps)
                reward += REWARD_EARLY_CONTAINMENT * time_factor

        # Term 4: Penalty for unnecessary restriction
        if action.action_type == "restrict":
            target = self._city.districts[action.district_id]
            if target.true_infection_rate < LOW_THRESHOLD:
                reward += REWARD_UNNECESSARY_RESTRICTION

        # Term 5: Bonus for correctly prioritising the most infected district
        if action.action_type == "allocate":
            if action.district_id == get_highest_infected_district(self._city.districts):
                reward += REWARD_CORRECT_PRIORITISATION

        return round(reward, 4)

    # ── Private: Terminal Conditions ──────────────────────────────────────────

    def _check_terminal(self) -> Tuple[bool, Optional[str]]:
        """
        Check if the episode should end.
        Returns (done, message) — message is None if episode continues.
        """
        # Success: all districts contained
        if all_districts_contained(self._city.districts):
            return True, "✓ Outbreak contained. All districts below safe threshold."

        # Failure: hospital collapse
        if any_hospital_breached(self._city.districts):
            return True, "✗ Hospital capacity breached. Episode failed."

        # Natural end: max steps reached
        if self._state.step_count >= self._city.max_steps:
            return True, f"Episode complete. {self._city.max_steps} steps reached."

        return False, None
    
    def get_trajectory(self) -> list:
        """Return the recorded trajectory for the current episode."""
        return self._trajectory