# baseline/policy.py
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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
        "You are an epidemic response coordinator making life-or-death resource decisions.",
        "Your goal: keep infection BELOW 0.40 in all districts and prevent hospital collapse.",
        "",
        f"Step {obs.current_step}/{obs.max_steps} | Resources remaining: {obs.available_resources}",
        "ℹ️  Resources replenish by 1 each step, capped at starting pool. Spend carefully.",
        "",
        "Districts (sorted by CURRENT infection, highest threat first):",
    ]

    for d in sorted_districts:
        if d.reported_infection_rate > 0.4:
            status = "🔴 CRITICAL"
        elif d.reported_infection_rate > 0.2:
            if d.growth_rate_hint > 0.06:
                status = "🟡 WARNING→CRITICAL SOON"
            else:
                status = "🟡 WARNING"
        else:
            status = "🟢 SAFE"

        if d.hospital_capacity_remaining < 0.3:
            hosp_status = "⚠️ HOSPITAL DANGER"
        elif d.hospital_capacity_remaining < 0.6:
            hosp_status = "hospital LOW"
        else:
            hosp_status = "hospital OK"

        if has_data_lag:
            # Pre-compute estimated current infection — don't ask LLM to do math
            estimated = round(min(1.0, d.reported_infection_rate + 3 * d.growth_rate_hint), 2)
            if estimated > 0.4:
                est_status = "🔴 EST.CRITICAL"
            elif estimated > 0.2:
                est_status = "🟡 EST.WARNING"
            else:
                est_status = "🟢 EST.SAFE"
            lines.append(
                f"  D{d.district_id}: reported={d.reported_infection_rate:.2f} [3 days old] "
                f"growth={d.growth_rate_hint:.2f} → ESTIMATED NOW={estimated:.2f} {est_status} "
                f"{hosp_status}({d.hospital_capacity_remaining:.2f})"
            )
        else:
            lines.append(
                f"  D{d.district_id}: {status} infection={d.reported_infection_rate:.2f} "
                f"growth={d.growth_rate_hint:.2f} {hosp_status}({d.hospital_capacity_remaining:.2f})"
            )

    lines += [""]

    if not has_data_lag:
        lines += [
            "HOW ACTIONS WORK:",
            "  - 'allocate': costs 1 resource. REDUCES existing infection by 5% AND slows spread.",
            "    Infections naturally recover 1%/day but spread (3-8%/day) dominates without action.",
            "    You need SUSTAINED allocation (multiple steps) to drive a district below safe level.",
            "  - 'restrict': FREE. Slows future spread only. Does NOT reduce existing infection.",
            "    Use only when you have no resources OR for districts already below 0.20.",
            "  - 'test': wastes 1 resource. Data is already real-time. NEVER use this.",
            "",
            "DECISION RULES — follow this priority order every step:",
            "1. HOSPITAL EMERGENCY: If ANY hospital < 0.30 capacity → allocate on that district NOW.",
            "2. TRIAGE: Look at ALL districts. Find the one with the HIGHEST infection rate right now.",
            "   That is your target this step. Do not stick to the same district if another is worse.",
            "3. CRITICAL DISTRICT (above 0.40 and have resources): allocate on the highest.",
            "4. WARNING DISTRICT (0.20-0.40) with growth > 0.06 AND resources available:",
            "   allocate on it NOW to prevent it from becoming CRITICAL next step.",
            "5. If resources = 0: restrict on the highest infected district.",
            "6. NEVER restrict a district below 0.20 — you will be penalised.",
            "7. NEVER use 'test' — it wastes a resource you cannot afford.",
            "",
            "KEY INSIGHT: Infection spreads 3-8% per day. A WARNING district at 0.38 with",
            "growth=0.07 will be CRITICAL next step. Act before it escalates, not after.",
        ]
    else:
        lines += [
            "HOW ACTIONS WORK:",
            "  - 'allocate': costs 1 resource. Reduces infection AND slows spread.",
            "  - 'restrict': FREE. Slows future spread only.",
            "  - Data is 3 DAYS OLD — you cannot see current true infection rates.",
            "    Use growth_hint to estimate which districts are getting worse fastest.",
            "",
            "DECISION RULES for delayed-information scenario:",
            "1. HOSPITAL EMERGENCY: If ANY hospital < 0.30 → allocate on that district NOW.",
            "2. Look at ESTIMATED NOW column — this is your best estimate of current infection.",
            "   The estimation accounts for 3 days of growth automatically.",
            "3. Allocate on the district with HIGHEST 'ESTIMATED NOW' value.",
            "4. If resources = 0: restrict on the district with highest ESTIMATED NOW.",
            "5. NEVER use 'test' — data lag is structural, testing does not help.",
        ]

    lines += [
        "",
        "Think briefly (1 sentence): Which district is most dangerous RIGHT NOW and why?",
        "Then give your JSON decision.",
        "",
        "Example response:",
        'District 0 is critical at 0.65 and growing fastest.',
        '{"action_type": "allocate", "district_id": 0}',
        "",
        "Your response:",
    ]

    return "\n".join(lines)


def call_llm(prompt: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model    = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
        messages = [
            {
                "role":    "system",
                "content": (
                    "You are an epidemic response AI. "
                    "First write one sentence of reasoning, then a JSON action on the next line. "
                    "JSON must be valid and contain action_type and district_id."
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens  = 60,   # increased to allow brief reasoning + JSON
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