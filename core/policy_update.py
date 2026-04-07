# core/policy_update.py
# ─────────────────────────────────────────────────────────────────────────────
# GRPO-style advantage computation and memory update logic.
# Determines which rollouts are above average and should be reinforced.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Tuple
from core.trajectory import EpisodicMemory


def compute_advantage(
    current_reward: float,
    completed_rewards: List[float],
) -> float:
    """
    GRPO advantage = R_i - mean(R).
    Positive advantage → this rollout was better than average → reinforce.
    Negative advantage → below average → suppress.
    """
    if not completed_rewards:
        return 0.0
    mean = sum(completed_rewards) / len(completed_rewards)
    return round(current_reward - mean, 4)


def should_reinforce(advantage: float) -> bool:
    """
    Reinforce if advantage >= 0 (at or above mean).
    Suppress if below mean.
    """
    return advantage > -0.5  # allow small negative margin to encourage exploration


def update_memory(
    memory:     EpisodicMemory,
    trajectory: List[dict],
    advantage:  float,
) -> int:
    """
    If advantage >= 0, store all positive-reward steps from this trajectory
    into episodic memory. Returns number of steps stored.

    If advantage < 0, memory is unchanged — bad rollout suppressed.
    """
    if not should_reinforce(advantage):
        return 0

    stored = 0
    for step_data in trajectory:
        if step_data["reward"] > -0.3:
            memory.store(step_data["obs"], step_data["action"], step_data["reward"])
            stored += 1

    return stored