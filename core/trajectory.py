import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from typing import List
from models import ContainmentAction, CityObservation


class EpisodicMemory:
    """
    Stores high-reward steps from past rollouts and retrieves similar
    past decisions to guide the next rollout via prompt injection.

    Similarity is measured by L1 distance on infection profiles,
    with a small bonus for matching the episode phase (early/mid/late).
    """

    def __init__(self, max_size: int = 20):
        self.memories: List[dict] = []
        self.max_size             = max_size

    def store(self, obs: CityObservation, action: ContainmentAction, reward: float):
        if reward < -0.3:
            return

        phase = "early" if obs.current_step <= obs.max_steps // 3 else \
                "mid"   if obs.current_step <= 2 * obs.max_steps // 3 else "late"

        self.memories.append({
            "infection_profile": [round(d.reported_infection_rate, 2) for d in obs.districts],
            "resources":         obs.available_resources,
            "phase":             phase,
            "action_type":       action.action_type,
            "district_id":       action.district_id,
            "reward":            round(reward, 4),
            "highest_district":  max(
                                     range(len(obs.districts)),
                                     key=lambda i: obs.districts[i].reported_infection_rate
                                 ),
        })

        self.memories.sort(key=lambda m: m["reward"], reverse=True)
        self.memories = self.memories[:self.max_size]

    def retrieve(self, obs: CityObservation, top_k: int = 5) -> str:
        if not self.memories:
            return ""

        current = [round(d.reported_infection_rate, 2) for d in obs.districts]
        phase   = "early" if obs.current_step <= obs.max_steps // 3 else \
                  "mid"   if obs.current_step <= 2 * obs.max_steps // 3 else "late"

        def score(memory: dict) -> float:
            profile = memory["infection_profile"]
            if len(profile) != len(current):
                return float("inf")
            l1          = sum(abs(a - b) for a, b in zip(profile, current))
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
