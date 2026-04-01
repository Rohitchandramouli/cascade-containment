# scripts/test_local.py
# Quick sanity check for everything built so far.
# Run this from the project root: python scripts/test_local.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../server'))

from server.environment import EpidemicContainmentEnv
from models import ContainmentAction
from server.grader import grade_trajectory, grade_task

def test_grader(task_name: str):
    print(f"\n--- Grader test: {task_name} ---")
    env = EpidemicContainmentEnv()
    obs = env.reset(task_name)

    while not obs.done:
        action = ContainmentAction(action_type="allocate", district_id=0)
        obs = env.step(action)

    trajectory = env.get_trajectory()
    result     = grade_trajectory(trajectory, task_name)

    print(f"  Final score:       {result.final_score:.4f}")
    print(f"  Containment:       {result.containment_score:.4f}")
    print(f"  Hospital:          {result.hospital_score:.4f}")
    print(f"  Efficiency:        {result.efficiency_score:.4f}")
    print(f"  Speed:             {result.speed_score:.4f}")
    print(f"  Hospital breached: {result.hospital_breached}")
    print(f"  Districts safe:    {result.districts_contained}")
    print(f"  Steps taken:       {result.total_steps}")
    assert 0.0 <= result.final_score <= 1.0, "Score out of range!"
    print(f"✓ Score in valid range [0.0, 1.0]")


def test_task(task_name: str):
    print(f"\n{'='*50}")
    print(f"Testing task: {task_name.upper()}")
    print(f"{'='*50}")

    env = EpidemicContainmentEnv()

    # Test reset()
    obs = env.reset(task_name)
    print(f"✓ reset() OK")
    print(f"  Districts:  {len(obs.districts)}")
    print(f"  Resources:  {obs.available_resources}")
    print(f"  Max steps:  {obs.max_steps}")
    print(f"  Message:    {obs.message}")

    # Test state()
    state = env.state
    print(f"✓ state() OK")
    print(f"  Episode ID: {state.episode_id}")
    print(f"  Step count: {state.step_count}")

    # Run a few steps with different action types
    actions = [
        ContainmentAction(action_type="test",     district_id=0),
        ContainmentAction(action_type="allocate", district_id=0),
        ContainmentAction(action_type="restrict", district_id=1),
        ContainmentAction(action_type="allocate", district_id=0),
        ContainmentAction(action_type="test",     district_id=1),
    ]

    total_reward = 0.0
    for i, action in enumerate(actions):
        obs = env.step(action)
        total_reward += obs.reward or 0.0
        print(f"  Step {i+1}: {action.action_type:8} → district {action.district_id} "
              f"| reward: {obs.reward:+.4f} | done: {obs.done}")
        if obs.done:
            print(f"  Episode ended early: {obs.message}")
            break

    print(f"✓ step() OK — total reward so far: {total_reward:+.4f}")

    # Test invalid action handling
    obs = env.reset(task_name)
    bad_action = ContainmentAction(action_type="invalid_type", district_id=99)
    obs = env.step(bad_action)
    print(f"✓ Invalid action handled gracefully: {obs.message}")


def run_full_episode(task_name: str):
    """Run a complete episode to verify terminal conditions work."""
    print(f"\n--- Full episode: {task_name} ---")
    env = EpidemicContainmentEnv()
    obs = env.reset(task_name)

    total_reward = 0.0
    step = 0

    while not obs.done:
        # Simple greedy policy: always allocate to district 0
        action = ContainmentAction(action_type="allocate", district_id=0)
        obs = env.step(action)
        total_reward += obs.reward or 0.0
        step += 1

    print(f"  Ended at step {step}: {obs.message}")
    print(f"  Total reward: {total_reward:+.4f}")
    print(f"✓ Full episode completed cleanly")


if __name__ == "__main__":
    print("Running Cascade Containment environment tests...\n")

    try:
        test_task("easy")
        test_task("medium")
        test_task("hard")

        test_grader("easy")
        test_grader("medium")
        test_grader("hard")

        run_full_episode("easy")
        run_full_episode("medium")
        run_full_episode("hard")

        print(f"\n{'='*50}")
        print("✓ ALL TESTS PASSED")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()