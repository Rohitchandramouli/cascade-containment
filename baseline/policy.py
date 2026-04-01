# baseline/policy.py
# ─────────────────────────────────────────────────────────────────────────────
# LLM-based policy for Cascade Containment.
# Reads CityObservation, calls LLM via OpenAI client, returns ContainmentAction.
# Uses environment variables for API configuration as required by hackathon rules.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import re
from openai import OpenAI
from models import CityObservation, ContainmentAction


# ── Client Setup ──────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """
    Initialise OpenAI client from environment variables.
    Required by hackathon rules — never hardcode API keys.
    """
    return OpenAI(
        api_key  = os.environ.get("HF_TOKEN", ""),
        base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    )


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(obs: CityObservation) -> str:
    """
    Convert a CityObservation into a clear, structured prompt.
    The prompt gives the LLM everything it needs to make an informed decision.
    """
    lines = [
        "You are a public health authority managing an epidemic outbreak.",
        "Your goal is to contain infection across all districts before hospitals collapse.",
        "",
        f"Current situation (Step {obs.current_step}/{obs.max_steps}):",
        f"Available resources: {obs.available_resources}",
        "",
        "District status:",
    ]

    for d in obs.districts:
        status = "DANGER" if d.reported_infection_rate > 0.4 else \
                 "WARNING" if d.reported_infection_rate > 0.2 else "SAFE"
        lines.append(
            f"  District {d.district_id}: "
            f"infection={d.reported_infection_rate:.2f} [{status}], "
            f"growth_hint={d.growth_rate_hint:.2f}, "
            f"hospital={d.hospital_capacity_remaining:.2f}, "
            f"restricted={'yes' if d.restriction_active else 'no'}, "
            f"tested_recently={'yes' if d.tested_recently else 'no'}"
        )

    lines += [
        "",
        "Available actions:",
        "  - 'test'     : Get accurate infection data for a district (costs 1 resource)",
        "  - 'restrict' : Impose movement restriction in a district (free, but penalised if infection is low)",
        "  - 'allocate' : Deploy medical resources to a district (costs 1 resource)",
        "",
        "Strategy hints:",
        "  - Prioritise districts in DANGER or with high growth_hint",
        "  - Use 'test' on high growth_hint districts to reveal true infection",
        "  - Use 'allocate' on the most infected district",
        "  - Only 'restrict' districts above 0.2 infection rate",
        "  - If resources = 0, you can only use 'restrict'",
        "",
        "Respond with ONLY a JSON object in this exact format:",
        '{"action_type": "allocate", "district_id": 2}',
        "",
        "Your decision:",
    ]

    return "\n".join(lines)


# ── LLM Call ──────────────────────────────────────────────────────────────────

def call_llm(prompt: str, client: OpenAI) -> str:
    """Call the LLM and return the raw response string."""
    response = client.chat.completions.create(
        model    = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
        messages = [
            {
                "role":    "system",
                "content": "You are an epidemic response AI. Always respond with valid JSON only. No explanation."
            },
            {
                "role":    "user",
                "content": prompt
            }
        ],
        max_tokens  = 50,
        temperature = 0.2,   # Low temperature for consistent, reliable decisions
    )
    return (response.choices[0].message.content or "").strip()


# ── Response Parser ───────────────────────────────────────────────────────────

def parse_action(response: str, num_districts: int) -> ContainmentAction:
    """
    Parse LLM response into a ContainmentAction.
    Handles common LLM formatting issues defensively.
    Falls back to a safe default if parsing fails entirely.
    """
    valid_types = {"test", "restrict", "allocate"}

    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", response).strip()

        # Extract JSON object if surrounded by other text
        match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group()

        data        = json.loads(cleaned)
        action_type = str(data.get("action_type", "allocate")).lower().strip()
        district_id = int(data.get("district_id", 0))

        # Validate and clamp
        if action_type not in valid_types:
            action_type = "allocate"
        district_id = max(0, min(district_id, num_districts - 1))

        return ContainmentAction(
            action_type = action_type,
            district_id = district_id,
        )

    except Exception:
        # Safe fallback — allocate to district 0
        return ContainmentAction(action_type="allocate", district_id=0)


# ── Main Policy Function ──────────────────────────────────────────────────────

def get_action(obs: CityObservation, client: OpenAI) -> ContainmentAction:
    """
    Main entry point for the policy.
    Takes an observation, returns a ContainmentAction.
    Called by evaluator.py on every step.
    """
    prompt      = build_prompt(obs)
    response    = call_llm(prompt, client)
    action      = parse_action(response, len(obs.districts))
    return action