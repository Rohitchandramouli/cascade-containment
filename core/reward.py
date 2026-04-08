def normalise_score(total_reward: float, steps: int, num_districts: int = 2) -> float:
    """
    Fallback score when the /grade endpoint is unreachable.
    Maps cumulative reward linearly into [0, 1] using task-aware best/worst bounds.
    """
    avg   = total_reward / max(steps, 1)
    worst = num_districts * (-1.5)      # -0.5 infection + -1.0 breach per district
    best  = num_districts * (0.5) + 0.3
    score = (avg - worst) / (best - worst)
    return round(min(1.0, max(0.0, score)), 4)
