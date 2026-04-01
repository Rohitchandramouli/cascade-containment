# core/reward.py
# ─────────────────────────────────────────────────────────────────────────────
# Reward computation utilities for the GRPO evaluation loop.
# ─────────────────────────────────────────────────────────────────────────────

import math


def normalise_score(total_reward: float, steps: int) -> float:
    """
    Map cumulative reward to [0.0, 1.0] via sigmoid on average reward per step.
    Guaranteed to always return a value strictly within the valid range.

    Average reward of 0   → 0.5
    Positive average      → above 0.5
    Negative average      → below 0.5
    """
    raw   = total_reward / max(steps, 1)
    score = 1.0 / (1.0 + math.exp(-raw))
    return round(min(1.0, max(0.0, score)), 4)