# baseline/evaluator.py
# ─────────────────────────────────────────────────────────────────────────────
# GRPO-style evaluation loop for Cascade Containment.
# Imports core components — stays focused on orchestration only.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
from typing import List, Tuple
from openai import OpenAI
from typing import Any

from client import CascadeContainmentEnv
from models import ContainmentAction, CityObservation
from baseline.policy import get_client, build_prompt, call_llm, parse_action
from core.trajectory import EpisodicMemory
from core.reward import normalise_score
from core.policy_update import compute_advantage, update_memory

import requests as http_requests

N_ROLLOUTS = 5


# ── Prompt Builder With Memory ────────────────────────────────────────────────

def build_prompt_with_memory(obs: CityObservation, memory: EpisodicMemory) -> str:
    """Extend base prompt with retrieved memories from similar past situations."""
    from baseline.policy import build_prompt
    base         = build_prompt(obs)
    memory_block = memory.retrieve(obs)

    if not memory_block:
        return base

    injection = (
        "\n"
        + memory_block
        + "\nUse these past experiences to make a better decision.\n"
    )
    return base.replace("Your decision:", injection + "Your decision:")


# ── Single Rollout ────────────────────────────────────────────────────────────

def run_rollout(
    env:       Any,
    task_name: str,
    client:    OpenAI,
    memory:    EpisodicMemory,
    verbose:   bool = True,
) -> Tuple[float, int, List[dict]]:
    """Run one complete episode using memory-augmented prompts."""
    result = env.reset(task_name=task_name)
    obs          = result.observation
    done         = result.done
    total_reward = 0.0
    step         = 0
    trajectory   = []

    while not done:
        prompt   = build_prompt_with_memory(obs, memory)
        response = call_llm(prompt, client)
        action   = parse_action(response, len(obs.districts))

        try:
            result   = env.step(action)
        except Exception as e:
            if "close frame" in str(e).lower() or "websocket" in str(e).lower():
                if verbose:
                    print(f"      ⚠ WebSocket dropped at step {step+1}, ending rollout early")
                break
            raise

        next_obs     = result.observation
        reward       = result.reward or 0.0
        done         = result.done
        total_reward += reward
        step         += 1

        trajectory.append({
            "obs":    obs,
            "action": action,
            "reward": reward,
        })

        if verbose:
            print(
                f"      step {step:2d}: {action.action_type:8} "
                f"→ district {action.district_id} "
                f"| reward: {reward:+.4f}"
            )

        obs = next_obs
        if done:
            break

    return total_reward, step, trajectory


# ── GRPO Task Runner ──────────────────────────────────────────────────────────

def run_task_grpo(
    env:       Any,
    task_name: str,
    client:    OpenAI,
    base_url:  str,
    verbose:   bool = True,
) -> float:
    """GRPO-style simulated learning loop for one task."""
    if verbose:
        print(f"\n  Task: {task_name.upper()} | {N_ROLLOUTS} rollouts")
        print(f"  {'─'*44}")

    memory   = EpisodicMemory(max_size=20)
    rollouts = []

    for i in range(N_ROLLOUTS):
        if verbose:
            label = "base prompt" if len(memory) == 0 else f"memory: {len(memory)} entries"
            print(f"\n    Rollout {i+1}/{N_ROLLOUTS} [{label}]")

        total_reward, steps, trajectory = run_rollout(env, task_name, client, memory, verbose)

        # Get proper grader score from server
        num_districts = {"easy": 2, "medium": 4, "hard": 6}.get(task_name, 2)
        try:
            grade_resp = http_requests.get(
                base_url.rstrip('/') + '/grade', timeout=10
            )
            if grade_resp.status_code == 200:
                data  = grade_resp.json()
                score = data["final_score"]
                if verbose:
                    print(
                        f"    → Grader: containment={data['containment_score']:.3f} "
                        f"hospital={data['hospital_score']:.3f} "
                        f"efficiency={data['efficiency_score']:.3f} "
                        f"speed={data['speed_score']:.3f}"
                    )
            else:
                score = normalise_score(total_reward, steps, num_districts)
        except Exception:
            score = normalise_score(total_reward, steps, num_districts)

        # Append BEFORE advantage computation
        rollouts.append((total_reward, steps, score))

        if verbose:
            print(f"    → Reward: {total_reward:+.4f} | Score: {score:.4f}")

        # ── GRPO advantage computation and memory update ──────────────────────
        completed_rewards = [r[0] for r in rollouts]
        advantage         = compute_advantage(total_reward, completed_rewards[:-1])
        stored            = update_memory(memory, trajectory, advantage)

        if verbose:
            mean = sum(completed_rewards[:-1]) / max(len(completed_rewards) - 1, 1) \
                if len(completed_rewards) > 1 else total_reward
            print(f"    → Advantage: {advantage:+.4f} | "
                + (f"↑ Stored {stored} steps" if stored > 0 else "↓ Suppressed"))

    all_rewards = [r[0] for r in rollouts]
    mean_reward = sum(all_rewards) / len(all_rewards)
    best_score = max(rollouts, key=lambda x: x[2])[2]

    if verbose:
        print(f"\n  Rewards:    {[round(r, 4) for r in all_rewards]}")
        print(f"  Mean:       {mean_reward:+.4f}")
        print(f"  Advantages: {[round(r - mean_reward, 4) for r in all_rewards]}")
        print(f"  Best score: {best_score:.4f}")

    return best_score


# ── Full Evaluator ────────────────────────────────────────────────────────────

def run_evaluation(
    base_url: str  = "http://localhost:7860",
    verbose:  bool = True,
) -> dict:
    """Run all three tasks with GRPO episodic memory learning."""
    if verbose:
        print("\n" + "="*52)
        print("  CASCADE CONTAINMENT — GRPO EVALUATION")
        print("="*52)
        print(f"  Rollouts per task:  {N_ROLLOUTS}")
        print(f"  Learning:           Episodic memory + advantage gating")

    client = get_client()
    scores = {}
    start  = time.time()

    with CascadeContainmentEnv(base_url=base_url).sync() as env:
        for task_name in ["easy", "medium", "hard"]:
            try:
                score = run_task_grpo(env, task_name, client, base_url, verbose)
                scores[task_name] = score
                if verbose:
                    print(f"\n  ✓ {task_name.upper()} final score: {score:.4f}")
            except Exception as e:
                scores[task_name] = 0.0
                if verbose:
                    print(f"  ✗ {task_name.upper()} failed: {e}")
                    import traceback
                    traceback.print_exc()

    scores["average"] = round(
        sum(v for k, v in scores.items() if k != "average") / 3,
        4
    )

    elapsed = round(time.time() - start, 1)

    if verbose:
        print("\n" + "="*52)
        print("  FINAL SCORES")
        print("="*52)
        print(f"  Easy:    {scores.get('easy',    0.0):.4f}")
        print(f"  Medium:  {scores.get('medium',  0.0):.4f}")
        print(f"  Hard:    {scores.get('hard',    0.0):.4f}")
        print(f"  {'─'*32}")
        print(f"  Average: {scores.get('average', 0.0):.4f}")
        print(f"  Time:    {elapsed}s")
        print("="*52 + "\n")

    return scores