# core/trajectory.py
# ─────────────────────────────────────────────────────────────────────────────
# Episodic memory for GRPO-style simulated learning.
# Stores high-reward (observation, action, reward) tuples from past rollouts.
# Retrieved at each step to provide instance-level guidance to the policy.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List
from models import ContainmentAction, CityObservation


class EpisodicMemory:
    """
    Stores high-reward steps from past rollouts.
    Retrieved by similarity to current observation to guide next rollout.

    This is the contextual bandit component — the agent gets specific examples:
    "last time infection was [0.45, 0.12] with 7 resources,
     allocating to district 0 earned +0.3"
    """

    def __init__(self, max_size: int = 20):
        self.memories: List[dict] = []
        self.max_size             = max_size

    def store(self, obs: CityObservation, action: ContainmentAction, reward: float):
        """Store a step only if it earned positive reward."""
        if reward < -0.3:
            return

        self.memories.append({
            "infection_profile": [round(d.reported_infection_rate, 2) for d in obs.districts],
            "resources":         obs.available_resources,
            "action_type":       action.action_type,
            "district_id":       action.district_id,
            "reward":            round(reward, 4),
        })

        # Keep only the highest-reward memories
        self.memories.sort(key=lambda m: m["reward"], reverse=True)
        self.memories = self.memories[:self.max_size]

    def retrieve(self, obs: CityObservation, top_k: int = 3) -> str:
        """
        Find stored memories most similar to the current observation.
        Similarity = L1 distance between infection profiles.
        Returns a formatted string for prompt injection.
        """
        if not self.memories:
            return ""

        current = [round(d.reported_infection_rate, 2) for d in obs.districts]

        def l1_distance(memory: dict) -> float:
            profile = memory["infection_profile"]
            if len(profile) != len(current):
                return float("inf")
            return sum(abs(a - b) for a, b in zip(profile, current))

        ranked = sorted(self.memories, key=l1_distance)
        top    = ranked[:top_k]

        lines  = ["Relevant past decisions (from successful rollouts):"]
        for m in top:
            lines.append(
                f"  - Profile {m['infection_profile']} | resources={m['resources']}: "
                f"'{m['action_type']}' on district {m['district_id']} "
                f"→ reward {m['reward']:+.4f}"
            )
        return "\n".join(lines)

    def clear(self):
        """Clear memory between tasks — memories are task-specific."""
        self.memories = []

    def __len__(self) -> int:
        return len(self.memories)