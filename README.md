---
title: Cascade Containment
emoji: 🦠
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🦠 Cascade Containment

> An RL benchmark for sequential resource allocation under spreading cascade dynamics.

A city health authority must allocate limited resources across districts to contain a spreading outbreak — with delayed data, resource scarcity, and cascading hospital stress. Designed as a **generalizable benchmark**: the same environment mechanics model wildfire resource deployment, cyberattack isolation, and misinformation containment.

Built for the **Meta PyTorch OpenEnv Hackathon x SST 2026**.

---

## The Problem

Sequential resource allocation under uncertainty is one of the most common and consequential decision problems in the real world. Whether containing an epidemic, deploying firefighting crews, or isolating a cyberattack — the agent faces the same fundamental challenge:

- Resources are scarce and cannot cover every district simultaneously
- Data is delayed — by the time a crisis is visible, it has already grown
- Interventions have cascading effects across adjacent areas
- Acting too late is catastrophic; acting too early wastes resources

No existing OpenEnv benchmark formalizes this problem class. Cascade Containment does.

---

## Environment Design

### Action Space
One decision per step — kept deliberately simple to maximize strategic depth:

| Field | Type | Values |
|-------|------|--------|
| `action_type` | string | `"test"` · `"restrict"` · `"allocate"` |
| `district_id` | int | 0-indexed district target |

- **test** — spend 1 resource to get accurate infection data for a district
- **restrict** — impose movement restriction (free, but penalised if infection is low)
- **allocate** — deploy 1 resource unit to reduce spread rate this step

### Observation Space
The agent receives a filtered, potentially lagged view of the world — never the full ground truth:
```python
CityObservation:
  districts: List[DistrictObservation]  # per-district visible state
  available_resources: int               # budget remaining this turn
  current_step: int
  max_steps: int
  done: bool
  reward: float | None
  message: str | None                    # human-readable feedback
```

Each `DistrictObservation` contains:
- `reported_infection_rate` — real-time (easy/medium) or **3 days lagged** (hard)
- `growth_rate_hint` — noisy signal of true spread rate
- `hospital_capacity_remaining` — always accurate (hospitals report in real time)
- `tested_recently`, `restriction_active`

### The Key Design Decision: Partial Observability
The hard task exposes infection rates from **3 days ago**. The agent must learn to act on noisy forward signals (`growth_rate_hint`) rather than react to confirmed data — exactly the challenge real public health officials face. This single mechanic is what separates a thoughtful agent from a reactive one.

---

## Three Tasks

| Task | Districts | Steps | Resources | Data Lag | Challenge |
|------|-----------|-------|-----------|----------|-----------|
| `easy` | 2 | 10 | 10 | None | Single outbreak, clear signal |
| `medium` | 4 | 15 | 8 | None | Two simultaneous outbreaks, forced triage |
| `hard` | 6 | 20 | 7 | 3 days | Scarce resources, invisible acceleration |

---

## Reward Function

Five shaped reward terms fire independently each step, providing dense feedback throughout the episode:

| Term | Value | Purpose |
|------|-------|---------|
| Infection penalty | `-0.50` per district above threshold | Primary containment signal |
| Hospital breach | `-1.00` per collapsed hospital | Catastrophic failure deterrent |
| Early containment | `+0.50 × (1 - step/max_steps)` | Teaches proactive behaviour |
| Unnecessary restriction | `-0.20` | Prevents lazy blanket lockdowns |
| Correct prioritisation | `+0.30` | Rewards triage intelligence |

The early containment bonus decays over time — containing an outbreak on day 3 is worth more than on day 8. This single design decision is what teaches the agent to act before crises emerge rather than after.

---

## Grader

The grader is fully deterministic — no randomness, no LLM calls — producing a weighted composite score in `[0.0, 1.0]`:

| Component | Weight | Measures |
|-----------|--------|---------|
| Containment score | 45% | District-days below infection threshold |
| Hospital score | 30% | Capacity preserved across episode |
| Efficiency score | 15% | Resources directed to high-need districts |
| Speed score | 10% | Containment achieved faster than max steps |

---

## Baseline Agent — GRPO-Style Episodic Memory

The baseline uses a **simulated GRPO learning loop** with episodic memory — no weight updates required:

1. **Rollout 1** runs with base prompt, no prior knowledge
2. After each rollout, compute advantage = `R_i - mean(completed rollouts)`
3. If above average → store positive-reward steps into `EpisodicMemory`
4. If below average → suppress, memory unchanged
5. Next rollout retrieves the 3 most similar past situations by L1 distance on infection profiles and injects them as concrete examples into the prompt

The prompt is the policy. Memory updates are the policy improvement. This produces measurably better decisions across rollouts without any gradient computation.

---

## Baseline Scores

Dumb greedy policy (always allocates to district 0):

| Task | Score | Hospital Breached |
|------|-------|-------------------|
| Easy | ~0.50 | No |
| Medium | ~0.23 | Yes |
| Hard | ~0.21 | Yes |

A smart LLM agent using the episodic memory baseline consistently scores 0.65–0.80 on easy and shows meaningful improvement on medium across rollouts.

---

## Usage
```python
from client import CascadeContainmentEnv
from models import ContainmentAction

with CascadeContainmentEnv(
    base_url="https://RohitChandramouli6618-cascade-containment.hf.space"
).sync() as env:
    # Run easy task
    obs = env.reset(task_name="easy")
    
    while not obs.done:
        action = ContainmentAction(
            action_type="allocate",
            district_id=0
        )
        result = env.step(action)
        obs = result.observation
        print(f"Reward: {result.reward:.4f}")
```

### Running the Full Evaluation
```bash
# Set required environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_hf_token"
export ENV_BASE_URL="https://RohitChandramouli6618-cascade-containment.hf.space"

# Run inference
python inference.py
```

---

## Generalisation

This environment is not epidemic-specific. The core mechanics — spreading cascade, delayed data, resource scarcity, spatial spillover — are identical to:

- **Wildfire deployment** — pre-position crews before fire reaches populated areas
- **Cyberattack isolation** — quarantine systems before lateral movement completes
- **Misinformation containment** — deploy corrections before false narratives entrench
- **Poverty intervention** — allocate aid where need is growing, not just where it's visible

The environment is designed to be a lasting benchmark for this general problem class, not a pandemic novelty.

---

## Project Structure
```
epidemic_containment_env/
├── models.py              # Data contracts (Action, Observation, State)
├── constants.py           # All numeric configuration
├── client.py              # Client-side interface
├── openenv.yaml           # Environment manifest
├── inference.py           # Evaluation entry point
├── server/
│   ├── environment.py     # Core RL loop
│   ├── grader.py          # Deterministic scorer
│   ├── utils.py           # Spread computation, observation builder
│   ├── app.py             # FastAPI server
│   ├── Dockerfile
│   └── tasks/             # Easy / Medium / Hard task definitions
├── baseline/
│   ├── policy.py          # LLM agent with prompt engineering
│   ├── evaluator.py       # GRPO episodic memory loop
│   └── run.py             # CLI entry point
└── core/
    ├── trajectory.py      # EpisodicMemory class
    ├── reward.py          # Score normalisation
    └── policy_update.py   # Advantage computation
```

---

## Tags
`reinforcement-learning` · `resource-allocation` · `sequential-decision-making` · `partial-observability` · `cascade-dynamics` · `openenv` · `llm-agent`