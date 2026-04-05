# baseline/policy.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import re
from openai import OpenAI
from models import CityObservation, ContainmentAction


def get_client() -> OpenAI:
    return OpenAI(
        api_key  = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", ""),
        base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    )


def build_prompt(obs: CityObservation) -> str:
    num_districts = len(obs.districts)
    has_data_lag  = num_districts == 6

    sorted_districts = sorted(
        obs.districts,
        key=lambda d: d.reported_infection_rate,
        reverse=True
    )

    lines = [
        "You are an epidemic response coordinator.",
        "Your goal: reduce infection in all districts and prevent hospital collapse.",
        "",
        f"Step {obs.current_step}/{obs.max_steps} | Resources remaining: {obs.available_resources}",
        "ℹ️  Resources replenish by 1 each step but cannot exceed your starting pool.",
        "   Spend wisely — you can never have more resources than the starting amount.",
        "",
        "Districts (sorted by infection rate, highest first):",
    ]

    for d in sorted_districts:
        if d.reported_infection_rate > 0.4:
            status = "🔴 CRITICAL"
        elif d.reported_infection_rate > 0.2:
            status = "🟡 WARNING"
        else:
            status = "🟢 SAFE"

        if d.hospital_capacity_remaining < 0.3:
            hosp_status = "⚠️ HOSPITAL DANGER"
        elif d.hospital_capacity_remaining < 0.6:
            hosp_status = "hospital LOW"
        else:
            hosp_status = "hospital OK"

        lag_note = " [DATA IS 3 DAYS OLD]" if has_data_lag else ""
        lines.append(
            f"  D{d.district_id}: {status} infection={d.reported_infection_rate:.2f}{lag_note} "
            f"growth={d.growth_rate_hint:.2f} {hosp_status}({d.hospital_capacity_remaining:.2f})"
        )

    lines += [""]

    if not has_data_lag:
        lines += [
            "HOW ACTIONS WORK:",
            "  - 'allocate': costs 1 resource. REDUCES existing infection AND slows spread.",
            "    Infections naturally recover slightly each day, but spread dominates",
            "    without intervention. Sustained allocation is needed to drive below safe.",
            "  - 'restrict': FREE. Slows future spread. Does NOT reduce existing infection.",
            "    Use when resources are low or for WARNING districts.",
            "  - 'test': costs 1 resource. Reveals accurate data. (Not needed — data is real-time.)",
            "",
            "STRATEGY — follow in order:",
            "1. If ANY hospital is below 0.3 capacity: 'allocate' on that district IMMEDIATELY.",
            "2. Find the district with HIGHEST infection rate.",
            "3. If it is above 0.2 and you have resources: 'allocate' on it.",
            "4. Keep allocating to the SAME district next step too.",
            "   Only switch when that district drops below 0.2 (safe).",
            "5. If resources = 0: 'restrict' on the highest infected district.",
            "6. NEVER use 'test' — data is real-time and accurate.",
            "7. NEVER restrict a district below 0.2 — you will be penalised.",
        ]
    else:
        lines += [
            "HOW ACTIONS WORK:",
            "  - 'allocate': costs 1 resource. Reduces infection AND slows spread.",
            "    Data is 3 days old — act on growth_hint to anticipate true state.",
            "  - 'restrict': FREE. Slows future spread only.",
            "  - 'test': costs 1 resource. (NOT recommended — lag is unavoidable.)",
            "",
            "STRATEGY — follow in order:",
            "1. If ANY hospital is below 0.3 capacity: 'allocate' on that district IMMEDIATELY.",
            "2. If resources > 0: 'allocate' on the district with HIGHEST growth_hint.",
            "   High growth_hint means true infection is accelerating — act early.",
            "3. If resources = 0: 'restrict' on the district with highest growth_hint.",
            "4. NEVER use 'test' — spending resources on information wastes your budget.",
        ]

    lines += [
        "",
        "Respond with ONLY valid JSON. No explanation. Example:",
        '{"action_type": "allocate", "district_id": 0}',
        "",
        "Your decision:",
    ]

    return "\n".join(lines)


def call_llm(prompt: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model    = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
        messages = [
            {
                "role":    "system",
                "content": "You are an epidemic response AI. Always respond with valid JSON only. No explanation, no markdown."
            },
            {
                "role":    "user",
                "content": prompt
            }
        ],
        max_tokens  = 50,
        temperature = 0.1,
    )
    return (response.choices[0].message.content or "").strip()


def parse_action(response: str, num_districts: int) -> ContainmentAction:
    valid_types = {"test", "restrict", "allocate"}
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
        match   = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group()
        data        = json.loads(cleaned)
        action_type = str(data.get("action_type", "allocate")).lower().strip()
        district_id = int(data.get("district_id", 0))
        if action_type not in valid_types:
            action_type = "allocate"
        district_id = max(0, min(district_id, num_districts - 1))
        return ContainmentAction(action_type=action_type, district_id=district_id)
    except Exception:
        return ContainmentAction(action_type="allocate", district_id=0)


def get_action(obs: CityObservation, client: OpenAI) -> ContainmentAction:
    prompt   = build_prompt(obs)
    response = call_llm(prompt, client)
    return parse_action(response, len(obs.districts))