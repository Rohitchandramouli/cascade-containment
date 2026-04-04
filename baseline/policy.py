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
    has_data_lag  = num_districts == 6  # only hard task has lagged data

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
        "⚠️  Resources do NOT replenish. Every resource spent is gone permanently.",
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
        # Easy and medium: data is accurate, allocate reduces existing infection
        lines += [
            "HOW ACTIONS WORK:",
            "  - 'allocate': costs 1 resource, REDUCES existing infection AND slows spread",
            "  - 'restrict': FREE, only slows future spread, does NOT reduce infection",
            "  - 'test': costs 1 resource, gives accurate data (NOT needed here, data is real-time)",
            "",
            "DECISION RULES (follow in order):",
            "1. If ANY hospital is below 0.3 capacity: 'allocate' on that district IMMEDIATELY.",
            "2. If resources > 0: ALWAYS 'allocate' on the district with the SINGLE HIGHEST infection rate.",
            "3. Do NOT spread resources across multiple districts in the same turn — focus all pressure on the worst district.",
            "4. If resources = 0: 'restrict' on the highest infected district.",
            "5. NEVER use 'test' — data is already accurate.",
            "6. NEVER restrict a SAFE district (below 0.2) — you will be penalised.",
        ]
    else:
        # Hard task: data is 3 days old, act on growth_hint
        lines += [
            "HOW ACTIONS WORK:",
            "  - 'allocate': costs 1 resource, reduces infection AND slows spread",
            "  - 'restrict': FREE, only slows future spread",
            "  - 'test': costs 1 resource (NOT recommended — data lag is unavoidable)",
            "",
            "DECISION RULES (follow in order):",
            "1. If ANY hospital is below 0.3 capacity: 'allocate' on that district IMMEDIATELY.",
            "2. If resources > 0: 'allocate' on the district with HIGHEST growth_hint.",
            "3. Sustain pressure — keep allocating to the same district until growth_hint drops below 0.08.",
            "4. If resources = 0: 'restrict' on highest growth_hint district.",
            "5. NEVER use 'test'.",
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
        temperature = 0.1,  # Lower temperature for more consistent decisions
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