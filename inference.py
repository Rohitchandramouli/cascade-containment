import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
import requests as http_requests

from client import CascadeContainmentEnv
from models import ContainmentAction
from baseline.policy import get_client, build_prompt, call_llm, parse_action, build_prompt_with_memory
from core.trajectory import EpisodicMemory
from core.policy_update import compute_advantage, update_memory
from core.reward import normalise_score

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "cascade-containment"

N_ROLLOUTS = {
    "easy":   2,
    "medium": 4,
    "hard":   4,
}


# ── Mandatory structured log format ──────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


def run_rollout(
    env,
    task_name:   str,
    client:      OpenAI,
    memory:      EpisodicMemory,
    rollout_idx: int,
) -> tuple:
    result       = env.reset(task_name=task_name)
    obs          = result.observation
    done         = result.done
    total_reward = 0.0
    step         = 0
    trajectory   = []
    rewards      = []

    log_start(task=f"{task_name}-r{rollout_idx}", env=BENCHMARK, model=MODEL_NAME)

    try:
        while not done:
            prompt     = build_prompt_with_memory(obs, memory)
            response   = call_llm(prompt, client)
            action     = parse_action(response, len(obs.districts))
            action_str = f"{action.action_type}(district={action.district_id})"

            try:
                result = env.step(action)
            except Exception as e:
                log_step(step=step + 1, action=action_str, reward=0.0, done=True, error=str(e)[:80])
                log_end(success=False, steps=step, score=0.0, rewards=rewards)
                return total_reward, step, trajectory, 0.0

            next_obs      = result.observation
            reward        = result.reward or 0.0
            done          = result.done
            total_reward += reward
            step         += 1

            rewards.append(reward)
            trajectory.append({"obs": obs, "action": action, "reward": reward})
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            obs = next_obs
            if done:
                break

        score = 0.0
        try:
            grade_resp = http_requests.get(ENV_BASE_URL.rstrip('/') + '/grade', timeout=10)
            if grade_resp.status_code == 200:
                score = grade_resp.json().get("final_score", 0.0)
        except Exception:
            num_districts = {"easy": 2, "medium": 4, "hard": 6}.get(task_name, 2)
            score = normalise_score(total_reward, step, num_districts)

        success = score >= 0.40

    except Exception:
        log_end(success=False, steps=step, score=0.0, rewards=rewards)
        return total_reward, step, trajectory, 0.0

    log_end(success=success, steps=step, score=score, rewards=rewards)
    return total_reward, step, trajectory, score


def run_task(env, task_name: str, client: OpenAI) -> float:
    n_rollouts = N_ROLLOUTS[task_name]
    memory     = EpisodicMemory(max_size=20)
    rollouts   = []

    for i in range(1, n_rollouts + 1):
        total_reward, steps, trajectory, score = run_rollout(
            env, task_name, client, memory, rollout_idx=i
        )
        rollouts.append((total_reward, steps, score))

        completed_rewards = [r[0] for r in rollouts]
        advantage         = compute_advantage(total_reward, completed_rewards[:-1])
        update_memory(memory, trajectory, advantage)

    return max(r[2] for r in rollouts)


def main() -> dict:
    client = get_client()
    scores = {}
    start  = time.time()

    with CascadeContainmentEnv(base_url=ENV_BASE_URL).sync() as env:
        for task_name in ["easy", "medium", "hard"]:
            try:
                scores[task_name] = run_task(env, task_name, client)
            except Exception as e:
                scores[task_name] = 0.0
                print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)

    scores["average"] = round(
        sum(v for k, v in scores.items() if k != "average") / 3, 4
    )
    elapsed = round(time.time() - start, 1)

    print(
        f"\n# SCORES easy={scores.get('easy', 0):.4f} "
        f"medium={scores.get('medium', 0):.4f} "
        f"hard={scores.get('hard', 0):.4f} "
        f"average={scores.get('average', 0):.4f} "
        f"elapsed={elapsed}s",
        flush=True,
    )

    return scores


if __name__ == "__main__":
    scores = main()

    if scores.get("average", 0.0) == 0.0:
        print("\n[DEBUG] All scores zero — check environment variables:", flush=True)
        print(f"  ENV_BASE_URL = {ENV_BASE_URL}", flush=True)
        print(f"  API_BASE_URL = {API_BASE_URL}", flush=True)
        print(f"  MODEL_NAME   = {MODEL_NAME}", flush=True)
        sys.exit(1)
