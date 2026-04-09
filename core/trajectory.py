import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from typing import List
from models import ContainmentAction, CityObservation


class EpisodicMemory:
    """
    Stores high-reward steps from past rollouts and retrieves similar
    past decisions to guide the next rollout via prompt injection.

    Key design choice: actions are stored as RELATIVE ranks rather than
    absolute district IDs. Rank 0 = highest-infected district at the time
    of the action, rank 1 = second-highest, and so on.

    This matters because each episode randomises spread rates and densities,
    so "allocate D1" from rollout 1 may refer to a completely different
    epidemiological situation in rollout 2. Storing rank instead means
    memory encodes the strategy ("target the worst district") rather than
    an accident of episode initialisation ("target district 1").

    On retrieval, ranks are resolved back to actual current district IDs
    so the injected prompt text is immediately actionable.
    """

    def __init__(self, max_size: int = 20):
        self.memories: List[dict] = []
        self.max_size             = max_size

    def store(self, obs: CityObservation, action: ContainmentAction, reward: float):
        if reward < -0.3:
            return

        phase = "early" if obs.current_step <= obs.max_steps // 3 else \
                "mid"   if obs.current_step <= 2 * obs.max_steps // 3 else "late"

        # Sort districts by infection rate descending to get current rankings
        sorted_by_infection = sorted(
            obs.districts,
            key=lambda d: d.reported_infection_rate,
            reverse=True
        )
        id_to_rank = {d.district_id: rank for rank, d in enumerate(sorted_by_infection)}

        # Store rank rather than absolute ID
        district_rank = id_to_rank.get(action.district_id, 0)

        target_infection = next(
            (d.reported_infection_rate for d in obs.districts if d.district_id == action.district_id),
            0.0
        )
        highest_infection = sorted_by_infection[0].reported_infection_rate if sorted_by_infection else 0.0

        self.memories.append({
            "infection_profile":  [round(d.reported_infection_rate, 2) for d in obs.districts],
            "resources":          obs.available_resources,
            "phase":              phase,
            "action_type":        action.action_type,
            "district_rank":      district_rank,       # 0 = highest infected
            "target_infection":   round(target_infection, 2),
            "highest_infection":  round(highest_infection, 2),
            "reward":             round(reward, 4),
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

        # Resolve stored ranks back to current district IDs for this episode
        current_sorted = sorted(
            obs.districts,
            key=lambda d: d.reported_infection_rate,
            reverse=True
        )

        lines = ["Past decisions that earned positive reward (use as guidance):"]
        for m in top:
            rank = m["district_rank"]
            if rank < len(current_sorted):
                resolved_id = current_sorted[rank].district_id
                rank_label  = f"rank-{rank} district (currently D{resolved_id})"
            else:
                resolved_id = current_sorted[0].district_id if current_sorted else 0
                rank_label  = f"rank-0 district (currently D{resolved_id})"

            lines.append(
                f"  Phase={m.get('phase','?')} resources={m['resources']} "
                f"highest={m['highest_infection']:.2f} target={m['target_infection']:.2f}: "
                f"'{m['action_type']}' {rank_label} → reward {m['reward']:+.2f}"
            )
        return "\n".join(lines)

    def clear(self):
        self.memories = []

    def __len__(self) -> int:
        return len(self.memories)
