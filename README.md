---
title: Cascade Containment
emoji: 🦠
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Cascade Containment

An RL benchmark for epidemic containment policy under uncertainty.
A city health authority must allocate limited resources across districts
to contain a spreading outbreak — with delayed data, resource scarcity,
and cascading hospital stress.

Generalises to wildfire deployment, cyberattack isolation, and misinformation containment.

## Environment

- **3 tasks:** Easy (2 districts), Medium (4 districts), Hard (6 districts with 3-day data lag)
- **Action space:** `action_type` (test/restrict/allocate) + `district_id`
- **Learning:** GRPO-style episodic memory with advantage gating

## Usage

\```python
from client import CascadeContainmentEnv
from models import ContainmentAction

with CascadeContainmentEnv(base_url="https://YOUR-SPACE-URL.hf.space").sync() as env:
    obs = env.reset(task_name="easy")
    result = env.step(ContainmentAction(action_type="allocate", district_id=0))
\```

## Tasks

| Task | Districts | Steps | Resources | Data Lag |
|------|-----------|-------|-----------|----------|
| easy | 2 | 10 | 10 | None |
| medium | 4 | 15 | 8 | None |
| hard | 6 | 20 | 7 | 3 days |

## Reward Function

| Term | Value | Condition |
|------|-------|-----------|
| Infection penalty | -0.50 | Per district above 0.4 threshold |
| Hospital breach | -1.00 | Per breached hospital |
| Early containment | +0.50 | Scaled by time remaining |
| Unnecessary restriction | -0.20 | Restricting below 0.2 threshold |
| Correct prioritisation | +0.30 | Allocating to highest-infected district |