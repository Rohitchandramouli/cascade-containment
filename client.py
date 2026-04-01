# client.py
# ─────────────────────────────────────────────────────────────────────────────
# Client-side interface for the Cascade Containment environment.
# Implements the two required abstract methods from EnvClient:
#   _step_payload  — serialises ContainmentAction to dict for WebSocket
#   _parse_result  — deserialises server response to CityObservation
# ─────────────────────────────────────────────────────────────────────────────

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import ContainmentAction, CityObservation


class CascadeContainmentEnv(EnvClient[ContainmentAction, CityObservation, State]):
    """
    Client for the Cascade Containment OpenEnv environment.

    Async usage:
        async with CascadeContainmentEnv(base_url="http://localhost:7860") as env:
            obs = await env.reset("easy")
            result = await env.step(ContainmentAction(action_type="allocate", district_id=0))

    Sync usage:
        with CascadeContainmentEnv(base_url="http://localhost:7860").sync() as env:
            obs = env.reset("easy")
            result = env.step(ContainmentAction(action_type="allocate", district_id=0))
    """

    def _step_payload(self, action: ContainmentAction) -> dict:
        """Serialise ContainmentAction to dict for WebSocket transmission."""
        return {
            "action_type": action.action_type,
            "district_id": action.district_id,
        }

    def _parse_result(self, result: dict) -> StepResult:
        """Deserialise server response into a typed StepResult."""
        observation = CityObservation(**result["observation"])
        return StepResult(
            observation = observation,
            reward      = result.get("reward", 0.0),
            done        = result.get("done", False),
        )
    
    def _parse_state(self, result: dict) -> State:
        """Deserialise server response into a typed State."""
        return State(
            episode_id  = result.get("episode_id", ""),
            step_count  = result.get("step_count", 0),
        )


# ── Connection test (run directly to verify client works) ─────────────────────

if __name__ == "__main__":
    with CascadeContainmentEnv(base_url="http://localhost:7860").sync() as env:
        obs = env.reset()
        print(f"✓ Connected successfully")
        print(f"  Districts:  {len(obs.observation.districts)}")
        print(f"  Resources:  {obs.observation.available_resources}")
        print(f"  Max steps:  {obs.observation.max_steps}")

        result = env.step(ContainmentAction(action_type="allocate", district_id=0))
        print(f"  Step reward: {result.reward}")
        print(f"✓ Client working end-to-end")