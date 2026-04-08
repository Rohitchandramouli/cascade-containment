import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List
from core.trajectory import EpisodicMemory


def compute_advantage(
    current_reward:    float,
    completed_rewards: List[float],
) -> float:
    """
    GRPO advantage = R_i - mean(R_completed).
    Positive means this rollout was better than average; negative means worse.
    Returns 0.0 on the first rollout where there's nothing to compare against.
    """
    if not completed_rewards:
        return 0.0
    mean = sum(completed_rewards) / len(completed_rewards)
    return round(current_reward - mean, 4)


def should_reinforce(advantage: float) -> bool:
    # Small negative margin is allowed to encourage exploration on borderline rollouts.
    return advantage > -0.5


def update_memory(
    memory:     EpisodicMemory,
    trajectory: List[dict],
    advantage:  float,
) -> int:
    """
    Store positive-reward steps from this trajectory into episodic memory
    if the rollout was at or above average (advantage > threshold).
    Returns the number of steps stored. Bad rollouts leave memory unchanged.
    """
    if not should_reinforce(advantage):
        return 0

    stored = 0
    for step_data in trajectory:
        if step_data["reward"] > -0.3:
            memory.store(step_data["obs"], step_data["action"], step_data["reward"])
            stored += 1

    return stored
