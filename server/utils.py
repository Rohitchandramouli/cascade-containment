# utils.py
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
    NATURAL_RECOVERY_RATE,
)


def generate_districts(
    num_districts: int,
    seed_infections: List[float],
) -> List[DistrictTruth]:
    assert len(seed_infections) == num_districts
    raw_densities = [random.uniform(0.5, 1.5) for _ in range(num_districts)]
    total         = sum(raw_densities)
    densities     = [round(d / total, 4) for d in raw_densities]
    districts = []
    for i in range(num_districts):
        districts.append(DistrictTruth(
            district_id                 = i,
            true_infection_rate         = seed_infections[i],
            true_spread_rate            = round(random.uniform(SPREAD_RATE_MIN, SPREAD_RATE_MAX), 4),
            hospital_capacity_remaining = 1.0,
            population_density          = densities[i],
            days_since_tested           = 99,
            restriction_active          = False,
            deployed_resources          = 0,
        ))
    return districts


def generate_episode_id() -> str:
    return str(uuid.uuid4())[:8]


def build_observation(
    state:        CityState,
    step_count:   int,
    reward:       Optional[float] = None,
    message:      Optional[str]   = None,
    done:         bool             = False,
) -> CityObservation:
    """
    Convert hidden CityState into the agent-visible CityObservation.
    Hard task enforces 3-day data lag on infection rates.
    Hospital capacity and growth hints are always real-time.
    """
    district_observations = []
    for i, district in enumerate(state.districts):
        if state.data_lag_days > 0 and len(state.infection_history) >= state.data_lag_days:
            reported_rate = state.infection_history[-state.data_lag_days][i]
        else:
            reported_rate = district.true_infection_rate

        noise       = random.uniform(-GROWTH_HINT_NOISE, GROWTH_HINT_NOISE)
        growth_hint = round(max(0.0, min(1.0, district.true_spread_rate + noise)), 4)

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


def compute_spread(districts: List[DistrictTruth]) -> List[float]:
    """
    Compute new infection rates after one day.

    Epidemiological model:
      net_change = spread_rate - natural_recovery - intervention_reductions
      new_rate   = current + net_change + geographic_spillover

    Natural recovery (NATURAL_RECOVERY_RATE = 0.02/day) reflects infected
    individuals recovering without medical intervention. This means infection
    naturally decays slightly each day, but spread rate still dominates
    without active response — districts grow unless the agent acts.

    Spillover is LINEAR (no wrap-around): district 0 and district N-1 are
    not adjacent, reflecting a realistic city corridor or ring layout where
    geographically distant districts do not directly infect each other.
    """
    from server.constants import (
        ALLOCATE_REDUCTION,
        RESTRICT_REDUCTION,
        SPILLOVER_RATE,
        NATURAL_RECOVERY_RATE,
    )

    n         = len(districts)
    new_rates = []

    for i, district in enumerate(districts):
        effective_spread = district.true_spread_rate

        if district.restriction_active:
            effective_spread = max(0.0, effective_spread - RESTRICT_REDUCTION)

        if district.deployed_resources > 0:
            effective_spread = max(
                0.0,
                effective_spread - (ALLOCATE_REDUCTION * district.deployed_resources)
            )

        # Net change: growth minus natural recovery
        net_change = effective_spread - NATURAL_RECOVERY_RATE
        new_rate   = district.true_infection_rate + net_change

        # Linear spillover — no wrap-around
        if i > 0:
            new_rate += districts[i - 1].true_infection_rate * SPILLOVER_RATE
        if i < n - 1:
            new_rate += districts[i + 1].true_infection_rate * SPILLOVER_RATE

        new_rates.append(round(min(1.0, max(0.0, new_rate)), 4))

    return new_rates


def get_highest_infected_district(districts: List[DistrictTruth]) -> int:
    return max(districts, key=lambda d: d.true_infection_rate).district_id


def all_districts_contained(districts: List[DistrictTruth]) -> bool:
    return all(d.true_infection_rate < SAFE_THRESHOLD for d in districts)


def any_hospital_breached(districts: List[DistrictTruth]) -> bool:
    return any(d.hospital_capacity_remaining <= HOSPITAL_BREACH_POINT for d in districts)


def districts_above_threshold(districts: List[DistrictTruth]) -> List[DistrictTruth]:
    return [d for d in districts if d.true_infection_rate > INFECTION_THRESHOLD]


def snapshot_infection_rates(districts: List[DistrictTruth]) -> List[float]:
    return [d.true_infection_rate for d in sorted(districts, key=lambda d: d.district_id)]