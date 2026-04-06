# core/trajectory.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from typing import List
from models import ContainmentAction, CityObservation


class EpisodicMemory:
    """
    Stores high-reward steps from past rollouts.
    Retrieves by similarity to guide the next rollout.
    
    Improvement: stores resource level and episode phase alongside infection
    profile, and retrieves top_k=5 instead of 3 for richer context.
    """

    def __init__(self, max_size: int = 20):
        self.memories: List[dict] = []
        self.max_size             = max_size

    def store(self, obs: CityObservation, action: ContainmentAction, reward: float):
        """Store a step only if it earned meaningful positive reward."""
        if reward < -0.3:   # stricter threshold — only store clearly positive steps
            return

        # Phase: early/mid/late episode
        phase = "early" if obs.current_step <= obs.max_steps // 3 else \
                "mid"   if obs.current_step <= 2 * obs.max_steps // 3 else "late"

        self.memories.append({
            "infection_profile": [round(d.reported_infection_rate, 2) for d in obs.districts],
            "resources":         obs.available_resources,
            "phase":             phase,
            "action_type":       action.action_type,
            "district_id":       action.district_id,
            "reward":            round(reward, 4),
            # Store which district was highest at this step (useful for pattern learning)
            "highest_district":  max(range(len(obs.districts)),
                                     key=lambda i: obs.districts[i].reported_infection_rate),
        })

        # Keep only the highest-reward memories
        self.memories.sort(key=lambda m: m["reward"], reverse=True)
        self.memories = self.memories[:self.max_size]

    def retrieve(self, obs: CityObservation, top_k: int = 5) -> str:
        """
        Find stored memories most similar to the current observation.
        Similarity = L1 distance between infection profiles.
        Returns a formatted string for prompt injection.
        """
        if not self.memories:
            return ""

        current = [round(d.reported_infection_rate, 2) for d in obs.districts]
        phase   = "early" if obs.current_step <= obs.max_steps // 3 else \
                  "mid"   if obs.current_step <= 2 * obs.max_steps // 3 else "late"

        def score(memory: dict) -> float:
            profile = memory["infection_profile"]
            if len(profile) != len(current):
                return float("inf")
            l1 = sum(abs(a - b) for a, b in zip(profile, current))
            # Slight preference for matching episode phase
            phase_bonus = 0.0 if memory.get("phase") == phase else 0.1
            return l1 + phase_bonus

        ranked = sorted(self.memories, key=score)
        top    = ranked[:top_k]

        lines = ["Past decisions that earned positive reward (use as guidance):"]
        for m in top:
            lines.append(
                f"  Phase={m.get('phase','?')} Profile={m['infection_profile']} resources={m['resources']}: "
                f"'{m['action_type']}' D{m['district_id']} → reward {m['reward']:+.2f}"
            )
        return "\n".join(lines)

    def clear(self):
        self.memories = []

    def __len__(self) -> int:
        return len(self.memories)