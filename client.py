from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import ContainmentAction, CityObservation


class CascadeContainmentEnv(EnvClient[ContainmentAction, CityObservation, State]):
    """
    Client for the Cascade Containment environment.

    Async:
        async with CascadeContainmentEnv(base_url="http://localhost:7860") as env:
            obs = await env.reset("easy")
            result = await env.step(ContainmentAction(action_type="allocate", district_id=0))

    Sync:
        with CascadeContainmentEnv(base_url="http://localhost:7860").sync() as env:
            obs = env.reset("easy")
            result = env.step(ContainmentAction(action_type="allocate", district_id=0))
    """

    def _step_payload(self, action: ContainmentAction) -> dict:
        return {
            "action_type": action.action_type,
            "district_id": action.district_id,
        }

    def _parse_result(self, result: dict) -> StepResult:
        observation = CityObservation(**result["observation"])
        return StepResult(
            observation = observation,
            reward      = result.get("reward", 0.0),
            done        = result.get("done", False),
        )

    def _parse_state(self, result: dict) -> State:
        return State(
            episode_id = result.get("episode_id", ""),
            step_count = result.get("step_count", 0),
        )


if __name__ == "__main__":
    # Quick smoke test — run directly to verify the client connects and steps correctly.
    with CascadeContainmentEnv(base_url="http://localhost:7860").sync() as env:
        obs = env.reset()
        print(f"✓ Connected")
        print(f"  Districts:   {len(obs.observation.districts)}")
        print(f"  Resources:   {obs.observation.available_resources}")
        print(f"  Max steps:   {obs.observation.max_steps}")

        result = env.step(ContainmentAction(action_type="allocate", district_id=0))
        print(f"  Step reward: {result.reward}")
        print(f"✓ End-to-end OK")
