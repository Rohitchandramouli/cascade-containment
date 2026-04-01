# utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Helper functions used by environment.py, grader.py, and task files.
# No game logic lives here — only pure utility functions.
# ─────────────────────────────────────────────────────────────────────────────

import random
import uuid
from typing import List, Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import (
    DistrictObservation,
    DistrictTruth,
    CityObservation,
    CityState,
)
from server.constants import (
    SPREAD_RATE_MIN,
    SPREAD_RATE_MAX,
    GROWTH_HINT_NOISE,
    INFECTION_THRESHOLD,
    SAFE_THRESHOLD,
    HOSPITAL_BREACH_POINT,
)


# ── City Generation ───────────────────────────────────────────────────────────

def generate_districts(
    num_districts: int,
    seed_infections: List[float],
) -> List[DistrictTruth]:
    """
    Create a fresh list of DistrictTruth objects for a new episode.

    seed_infections is a list of starting infection rates, one per district.
    The task files control this — easy seeds one district, medium seeds two,
    hard seeds multiple with small values that grow over time.

    Population densities are assigned so they sum to exactly 1.0 across all
    districts, reflecting real cities where denser districts carry more risk.
    """
    assert len(seed_infections) == num_districts, (
        f"Expected {num_districts} seed values, got {len(seed_infections)}"
    )

    # Generate random population densities that sum to 1.0
    raw_densities  = [random.uniform(0.5, 1.5) for _ in range(num_districts)]
    total          = sum(raw_densities)
    densities      = [round(d / total, 4) for d in raw_densities]

    districts = []
    for i in range(num_districts):
        districts.append(DistrictTruth(
            district_id                 = i,
            true_infection_rate         = seed_infections[i],
            true_spread_rate            = round(random.uniform(SPREAD_RATE_MIN, SPREAD_RATE_MAX), 4),
            hospital_capacity_remaining = 1.0,
            population_density          = densities[i],
            days_since_tested           = 99,   # Large value → not recently tested
            restriction_active          = False,
            deployed_resources          = 0,
        ))

    return districts


def generate_episode_id() -> str:
    """Return a unique identifier for a new episode."""
    return str(uuid.uuid4())[:8]


# ── Observation Builder ───────────────────────────────────────────────────────

def build_observation(
    state:        CityState,
    step_count:   int,
    reward:       Optional[float] = None,
    message:      Optional[str]   = None,
    done:         bool             = False,
) -> CityObservation:
    """
    Convert the hidden CityState into the CityObservation the agent receives.

    This is where partial observability is enforced. When data_lag_days > 0,
    infection rates are pulled from infection_history rather than the current
    true values. Everything else (hospital capacity, growth hint, flags) is
    always reported in real time.
    """
    district_observations = []

    for i, district in enumerate(state.districts):

        # Apply data lag for hard task
        if state.data_lag_days > 0 and len(state.infection_history) >= state.data_lag_days:
            reported_rate = state.infection_history[-state.data_lag_days][i]
        else:
            reported_rate = district.true_infection_rate

        # Add noise to spread rate hint — agent sees a signal, not the truth
        noise        = random.uniform(-GROWTH_HINT_NOISE, GROWTH_HINT_NOISE)
        growth_hint  = round(max(0.0, min(1.0, district.true_spread_rate + noise)), 4)

        district_observations.append(DistrictObservation(
            district_id                 = district.district_id,
            reported_infection_rate     = round(reported_rate, 4),
            growth_rate_hint            = growth_hint,
            hospital_capacity_remaining = round(district.hospital_capacity_remaining, 4),
            population_density          = district.population_density,
            tested_recently             = district.days_since_tested <= 2,
            restriction_active          = district.restriction_active,
        ))

    return CityObservation(
        districts           = district_observations,
        available_resources = state.available_resources,
        current_step        = step_count,
        max_steps           = state.max_steps,
        done                = done,
        reward              = reward,
        message             = message,
    )


# ── Spread Computation ────────────────────────────────────────────────────────

def compute_spread(districts: List[DistrictTruth]) -> List[float]:
    """
    Calculate new infection rates for all districts after one day passes.

    Each district's infection grows by its spread rate, reduced by any
    active restriction or deployed resources. Adjacent districts (by index)
    receive a small spillover from their neighbours, simulating geographic
    spread without requiring a full spatial model.

    Returns a list of new infection rates (not yet applied to state).
    """
    from server.constants import ALLOCATE_REDUCTION, RESTRICT_REDUCTION, SPILLOVER_RATE

    n           = len(districts)
    new_rates   = []

    for i, district in enumerate(districts):

        # Base growth this day
        effective_spread = district.true_spread_rate

        # Reduce spread based on active interventions
        if district.restriction_active:
            effective_spread = max(0.0, effective_spread - RESTRICT_REDUCTION)

        if district.deployed_resources > 0:
            effective_spread = max(
                0.0,
                effective_spread - (ALLOCATE_REDUCTION * district.deployed_resources)
            )

        # Grow infection by effective spread rate
        new_rate = district.true_infection_rate + effective_spread

        # Add spillover from adjacent districts (wrap-around neighbours)
        left_neighbour  = districts[(i - 1) % n]
        right_neighbour = districts[(i + 1) % n]

        new_rate += left_neighbour.true_infection_rate  * SPILLOVER_RATE
        new_rate += right_neighbour.true_infection_rate * SPILLOVER_RATE

        # Clamp to valid range
        new_rates.append(round(min(1.0, max(0.0, new_rate)), 4))

    return new_rates


# ── Query Helpers ─────────────────────────────────────────────────────────────

def get_highest_infected_district(districts: List[DistrictTruth]) -> int:
    """
    Return the district_id of the district with the highest true infection rate.
    Used by the reward function to determine correct prioritisation.
    """
    return max(districts, key=lambda d: d.true_infection_rate).district_id


def all_districts_contained(districts: List[DistrictTruth]) -> bool:
    """
    Return True if every district's true infection rate is below SAFE_THRESHOLD.
    This triggers early episode termination with a success condition.
    """
    return all(d.true_infection_rate < SAFE_THRESHOLD for d in districts)


def any_hospital_breached(districts: List[DistrictTruth]) -> bool:
    """
    Return True if any district's hospital capacity has hit zero.
    This triggers early episode termination with a failure condition.
    """
    return any(d.hospital_capacity_remaining <= HOSPITAL_BREACH_POINT for d in districts)


def districts_above_threshold(districts: List[DistrictTruth]) -> List[DistrictTruth]:
    """
    Return all districts currently above INFECTION_THRESHOLD.
    Used by the reward function to compute per-step infection penalties.
    """
    return [d for d in districts if d.true_infection_rate > INFECTION_THRESHOLD]


def snapshot_infection_rates(districts: List[DistrictTruth]) -> List[float]:
    """
    Return current true infection rates as a plain list, ordered by district_id.
    Used by environment.py to append to infection_history each step.
    """
    return [d.true_infection_rate for d in sorted(districts, key=lambda d: d.district_id)]