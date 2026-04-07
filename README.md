---
title: Cascade Containment
emoji: 🦠
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

## 🦠 An RL Benchmark for Sequential Resource Allocation Under Spreading Cascade Dynamics

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square)](https://hub.docker.com)
[![HF Space](https://img.shields.io/badge/HF%20Space-Live-green?style=flat-square)](https://huggingface.co/spaces/TheRubberDuckDebuggers/cascade-containment)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Meta PyTorch OpenEnv Hackathon × SST 2026 — [Live Demo](https://therubberduckdebuggers-cascade-containment.hf.space) · [GitHub](https://github.com/Rohitchandramouli/cascade-containment)

---

## The Problem

Sequential resource allocation under uncertainty is one of the most consequential decision problems in the real world. Whether containing an epidemic, deploying firefighting crews, isolating a cyberattack, or routing aid — the agent faces the same fundamental challenge:

- **Resources are scarce** — you cannot cover every district simultaneously
- **Data is delayed** — by the time a crisis is visible, it has already grown
- **Interventions cascade** — actions in one district affect adjacent ones
- **Acting too late is catastrophic** — hospital collapse ends the episode; proactive containment is rewarded exponentially more than reactive response

No existing OpenEnv benchmark formalises this problem class. Cascade Containment does.

---

## Environment Overview

A city health authority must allocate limited medical resources across districts to contain a spreading outbreak. Each step, the agent observes district infection rates (possibly lagged), hospital capacity levels, and growth signals — then decides where to deploy resources, impose restrictions, or gather data.

The environment is **not epidemic-specific**. The underlying mechanics — spreading cascade, delayed observation, resource scarcity, geographic spillover — are structurally identical across multiple real-world domains.

---

## Quick Start

```python
from client import CascadeContainmentEnv
from models import ContainmentAction

with CascadeContainmentEnv(
    base_url="https://therubberduckdebuggers-cascade-containment.hf.space"
).sync() as env:
    result = env.reset(task_name="medium")
    obs    = result.observation

    while not result.done:
        most_infected = max(obs.districts, key=lambda d: d.reported_infection_rate)
        action = ContainmentAction(
            action_type="allocate",
            district_id=most_infected.district_id
        )
        result = env.step(action)
        obs    = result.observation
        print(f"Step {obs.current_step}: reward={result.reward:.3f}")
```

## Running the Full Baseline Evaluation

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="https://therubberduckdebuggers-cascade-containment.hf.space"

python inference.py
```

---

## Action Space

One decision per step, deliberately minimal to maximise strategic depth:

| Field | Type | Values |
| --- | --- | --- |
| `action_type` | `string` | `"test"` · `"restrict"` · `"allocate"` |
| `district_id` | `int` | 0-indexed district target |

| Action | Cost | Effect |
| --- | --- | --- |
| **test** | 1 resource | Reveals accurate current infection data for district |
| **restrict** | Free | Imposes movement restrictions; reduces spread rate; penalised if infection < 0.20 |
| **allocate** | 1 resource | Deploys medical resources; reduces existing infection by 5% and slows future spread |

Movement restrictions lift automatically once a district's infection drops below the safe threshold — reflecting real policy: restrictions are lifted when the outbreak is controlled.

---

## Observation Space

The agent receives a filtered, potentially lagged view of the world — **never the full ground truth**:

```python
CityObservation:
  districts:            List[DistrictObservation]  # per-district visible state
  available_resources:  int                         # budget remaining this step
  current_step:         int
  max_steps:            int
  done:                 bool
  reward:               float | None
  message:              str | None
```

Each `DistrictObservation` exposes:

| Field | Description | Observability |
| --- | --- | --- |
| `reported_infection_rate` | Active infection fraction | Real-time (easy/medium); **3-day lagged** (hard) |
| `growth_rate_hint` | Noisy signal of true spread rate | Always real-time ± noise |
| `hospital_capacity_remaining` | ICU/ward capacity fraction | Always real-time |
| `population_density` | District's share of city population | Always real-time |
| `restriction_active` | Whether movement restrictions are in place | Always real-time |
| `tested_recently` | Tested within last 2 days | Always real-time |

**The agent never sees:** `true_infection_rate`, `true_spread_rate`, or any ground truth used by the grader.

---

## Epidemiological Model

The simulation uses a realistic discrete-time SIR-inspired model:

```text
new_infection = current + (spread_rate − natural_recovery − intervention) + geographic_spillover
```

| Parameter | Value | Rationale |
| --- | --- | --- |
| Spread rate | 3–8% per day | Realistic for respiratory outbreaks (seasonal flu: 5–10%) |
| Natural recovery | 1% per day | Background case resolution without medical intervention |
| Treatment effect | −5% existing infection | Medical deployment (antivirals, PPE, rapid response) |
| Spread reduction | −10% per allocation | Resource-driven suppression of transmission |
| Geographic spillover | 1% to adjacent districts | Linear topology — no wrap-around (geographically realistic) |
| Hospital breach threshold | ≤10% capacity | Real ICU overflow and triage failure threshold |

---

## Three Tasks

| Task | Districts | Steps | Resources | Data Lag | Challenge |
| --- | --- | --- | --- | --- | --- |
| **easy** | 2 | 10 | 10 | None | Single outbreak, clear signal, abundant resources |
| **medium** | 4 | 15 | 8 | None | Two simultaneous outbreaks; forced triage between competing threats |
| **hard** | 6 | 15 | 7 | **3 days** | Six growing outbreaks; invisible acceleration; scarce resources |

### Task Design Philosophy

**Easy** establishes the core mechanic: D0 starts at 0.50 infection, D1 is clean. The agent must sustain focused allocation on D0 before D1 grows through spillover. A speed bonus fires if the agent acts decisively early.

**Medium** introduces genuine triage. With 8 resources across 4 districts over 15 steps, the agent simply cannot contain all districts. D0 and D2 start in the danger zone; D1 and D3 grow into crisis within 4–6 steps through spillover. The agent must choose which outbreaks to prioritise.

**Hard** adds the most realistic and challenging mechanic: **3-day information lag**. The agent sees infection rates from 3 days ago while true infection has already grown. The `growth_rate_hint` provides a noisy signal to reason about the current true state. A smart agent estimates current infection as `reported + 3 × growth_hint` and allocates accordingly.

---

## Reward Function

Five independent dense reward terms fire every step, providing rich learning signal throughout each episode:

| Term | Value | Fires When |
| --- | --- | --- |
| Infection penalty | `−0.50` per district | District infection > 0.40 |
| Hospital breach | `−1.00` per district | Hospital capacity ≤ 10% |
| Early containment | `+0.50 × (1 − step/max_steps)` | District infection < 0.20 |
| Correct prioritisation | `+0.30` | Allocate to highest-infected district |
| Unnecessary restriction | `−0.20` | Restrict district below 0.20 |

Key design choices:

- The **early containment bonus decays over time** — containing an outbreak on day 3 is worth more than on day 8. This teaches proactive behaviour rather than reactive scrambling.
- The **infection penalty scales with population density** — dense districts contribute more to the penalty, reflecting realistic triage priorities.
- The **hospital breach penalty** fires at 10% capacity, not 0% — reflecting real operational collapse thresholds where triage and diversion begin.
- **Restrictions lift automatically** when infection drops below the safe threshold, giving the agent a natural feedback loop on intervention effectiveness.

---

## Grader

The grader is **fully deterministic** — no randomness, no LLM calls. Identical trajectories always produce identical scores in `[0.0, 1.0]`.

### Score Components

| Component | Weight | Measures |
| --- | --- | --- |
| **Hospital score** | 45% | Average capacity preserved; ×0.6 multiplier if any district collapsed |
| **Containment score** | 30% | Fraction of district-days below infection threshold (grace period: first 2 steps excluded) |
| **Efficiency score** | 15% | Fraction of resource actions targeting the highest-infected district (uses pre-action state) |
| **Speed score** | 10% | `1 − (steps / max_steps)` if episode ends before max steps; else 0 |

Weight rationale: In real epidemic response, preserving healthcare system function (45%) is the primary operational constraint — a functional hospital system is the prerequisite for everything else. WHO and CDC outbreak protocols define success primarily by healthcare capacity preservation, with infection containment as the secondary signal. Efficiency (15%) rewards triage intelligence. Speed (10%) rewards proactive early intervention.

The efficiency score uses the **previous step's infection rates** to evaluate targeting decisions, ensuring that a successful treatment that drives infection below threshold is not retroactively penalised for being "unnecessary."

---

## Baseline Agent — GRPO-Style Episodic Memory

The baseline implements **simulated GRPO with episodic memory** — no weight updates, no gradient computation. The prompt is the policy; memory updates are the policy improvement.

### Learning Loop

```text
Rollout 1: Base prompt, no prior knowledge
  compute advantage = R1 - mean([])
  store positive-reward steps into EpisodicMemory

Rollout 2: Memory-augmented prompt
  retrieve top-5 similar past decisions by L1 distance
  inject as concrete examples into prompt
  compute advantage = R2 - mean([R1])
  reinforce if above average

... repeat for N rollouts
Report best grader score across all rollouts
```

### Memory Retrieval

Past decisions are stored as `(infection_profile, resources, phase, action, reward)` tuples. At each step, the top-5 most similar past situations are retrieved by L1 distance on infection profiles, weighted by episode phase (early/mid/late). This provides the agent with concrete examples of what worked in similar situations without any gradient update.

### Baseline Scores

| Task | Agent | Containment | Hospital | Efficiency | Final Score |
| --- | --- | --- | --- | --- | --- |
| Easy | Dumb greedy (always D0) | 0.50 | 0.92 | 0.45 | ~0.50 |
| Easy | LLM + GRPO memory | 1.00 | 1.00 | 1.00 | **0.88–0.93** |
| Medium | Dumb greedy | 0.18 | 0.21 | 0.40 | ~0.23 |
| Medium | LLM + GRPO memory | 0.44–0.73 | 0.97–1.00 | 0.87–1.00 | **0.70–0.85** |
| Hard | Dumb greedy | 0.12 | 0.18 | 0.25 | ~0.21 |
| Hard | LLM + GRPO memory | 0.28–0.51 | 0.86–0.97 | 0.47–0.73 | **0.58–0.65** |

The gap between dumb greedy and LLM+GRPO — particularly on medium (0.23 → 0.78) — demonstrates that the environment meaningfully discriminates between agent quality. This is the core benchmark property.

---

## Generalisation

This environment is not epidemic-specific. The underlying mechanics apply directly to:

| Domain | Spreading cascade | Delayed data | Resource scarcity |
| --- | --- | --- | --- |
| 🦠 **Epidemic containment** | Infection spreads between districts | Lagged case counts | Medical resources |
| 🔥 **Wildfire deployment** | Fire spreads across terrain | Satellite update delay | Firefighting crews |
| 🛡️ **Cyberattack isolation** | Lateral movement between systems | Detection lag | Security team hours |
| 📢 **Misinformation containment** | Narrative spread through networks | Viral detection lag | Correction budget |
| 🤝 **Poverty intervention** | Deprivation cascades through communities | Census data lag | Aid allocation |

The same trained policy generalises across domains with minimal prompt adaptation — this is the intended use case for the OpenEnv ecosystem.

---

## Project Structure

```text
cascade-containment/
├── inference.py              # Evaluation entry point (mandatory [START][STEP][END] logs)
├── models.py                 # Typed data contracts: Action, Observation, State
├── client.py                 # OpenEnv client interface
├── openenv.yaml              # Environment manifest for OpenEnv registry
│
├── server/
│   ├── app.py                # FastAPI server + judge dashboard + /grade /info /demo endpoints
│   ├── environment.py        # Core RL loop (reset/step/state OpenEnv interface)
│   ├── grader.py             # Deterministic trajectory scorer — no LLM calls
│   ├── constants.py          # Single source of truth for all numeric configuration
│   ├── utils.py              # Spread computation, observation builder, helper functions
│   ├── Dockerfile            # Container definition
│   └── tasks/
│       ├── task_easy.py      # 2 districts, 10 steps, real-time data
│       ├── task_medium.py    # 4 districts, 15 steps, forced triage
│       └── task_hard.py      # 6 districts, 15 steps, 3-day data lag
│
├── baseline/
│   ├── policy.py             # LLM policy with chain-of-thought prompting
│   ├── evaluator.py          # GRPO episodic memory loop
│   └── run.py                # CLI entry point
│
└── core/
    ├── trajectory.py         # EpisodicMemory — L1 similarity retrieval
    ├── reward.py             # Score normalisation utilities
    └── policy_update.py      # Advantage computation, memory gating
```

---

## OpenEnv Compliance

| Requirement | Status |
| --- | --- |
| `reset()` returns `CityObservation` | ✅ |
| `step(action)` returns `CityObservation` | ✅ |
| `state` property returns `State` | ✅ |
| Typed `Action` subclass | ✅ `ContainmentAction(Action)` |
| Typed `Observation` subclass | ✅ `CityObservation(Observation)` |
| `openenv.yaml` manifest | ✅ |
| Dockerfile builds | ✅ |
| HF Space deploys | ✅ |
| `inference.py` at root | ✅ |
| `[START][STEP][END]` structured logs | ✅ |
| Runtime < 20 minutes | ✅ ~16 minutes |
| `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars | ✅ |
| OpenAI client for all LLM calls | ✅ |
| Grader scores in `[0.0, 1.0]` | ✅ |
| 3+ tasks with difficulty progression | ✅ |

---

## Setup and Local Development

### Local Server

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

export ENV_BASE_URL=http://localhost:7860
python baseline/run.py
```

### Docker

```bash
docker build -f server/Dockerfile -t cascade-containment .
docker run -p 7860:7860 cascade-containment
```

---

## Tags

`reinforcement-learning` · `resource-allocation` · `sequential-decision-making` · `partial-observability` · `cascade-dynamics` · `epidemic-response` · `openenv` · `llm-agent` · `grpo` · `episodic-memory` · `triage` · `multi-district` · `docker` · `fastapi`
