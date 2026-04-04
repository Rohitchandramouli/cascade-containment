# core/reward.py
# ─────────────────────────────────────────────────────────────────────────────
# Reward computation utilities for the GRPO evaluation loop.
# ─────────────────────────────────────────────────────────────────────────────

import math

def normalise_score(total_reward: float, steps: int, num_districts: int = 2) -> float:
    """
    Linear normalization with task-aware worst case.
    Worst case per step = num_districts × (-0.5 infection) + num_districts × (-1.0 breach)
    Best case per step = num_districts × (+0.5 containment) + 0.30 prioritisation
    """
    avg   = total_reward / max(steps, 1)
    worst = num_districts * (-1.5)   # -0.5 infection + -1.0 breach per district
    best  = num_districts * (0.5) + 0.3
    score = (avg - worst) / (best - worst)
    return round(min(1.0, max(0.0, score)), 4)