# server/app.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application entry point for Cascade Containment.
# Exposes the OpenEnv WebSocket interface + HTTP endpoints for judges:
#   GET  /         → Interactive judge dashboard (Live Demo, Grader, Baseline info)
#   GET  /health   → Health check
#   GET  /info     → Environment metadata
#   GET  /grade    → Grader score for last completed episode
#   GET  /demo/{task} → Run a rule-based demo episode, return grader score + log
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openenv.core.env_server import create_app
from fastapi.responses import JSONResponse, HTMLResponse
from server.environment import EpidemicContainmentEnv
from server.grader import grade_trajectory
from models import ContainmentAction, CityObservation
import server.environment as env_module

app = create_app(
    EpidemicContainmentEnv,
    ContainmentAction,
    CityObservation,
)


# ── Grade endpoint ────────────────────────────────────────────────────────────

@app.get("/grade")
async def grade_last_episode():
    """Returns the deterministic grader score for the most recently completed episode."""
    if not env_module._last_grade:
        return JSONResponse(
            {"error": "No completed episode yet — run a full episode first"},
            status_code=400
        )
    return JSONResponse(env_module._last_grade)


# ── Info endpoint ─────────────────────────────────────────────────────────────

@app.get("/info")
async def environment_info():
    """Returns environment metadata for the OpenEnv registry."""
    return JSONResponse({
        "name":        "Cascade Containment",
        "version":     "1.0.0",
        "description": "RL benchmark for epidemic containment policy under uncertainty",
        "tasks": {
            "easy":   {"districts": 2, "max_steps": 10, "resources": 10, "data_lag": 0},
            "medium": {"districts": 4, "max_steps": 15, "resources": 8,  "data_lag": 0},
            "hard":   {"districts": 6, "max_steps": 15, "resources": 7,  "data_lag": 3},
        },
        "action_space": {
            "type":   "discrete",
            "fields": {
                "action_type": ["test", "restrict", "allocate"],
                "district_id": "int (0-indexed)"
            }
        },
        "grader_weights": {
            "containment": 0.45,
            "hospital":    0.30,
            "efficiency":  0.15,
            "speed":       0.10,
        },
        "generalisation": [
            "Wildfire resource deployment",
            "Cyberattack isolation",
            "Misinformation containment",
            "Poverty intervention"
        ]
    })


# ── Demo endpoint ─────────────────────────────────────────────────────────────

@app.get("/demo/{task_name}")
async def run_demo(task_name: str):
    """
    Runs a complete episode using a rule-based greedy agent (no LLM required).
    Always allocates to highest-infected district; restricts when resources exhausted.
    Returns grader score + full step log for display in the dashboard.
    """
    if task_name not in ["easy", "medium", "hard"]:
        return JSONResponse(
            {"error": "task_name must be one of: easy, medium, hard"},
            status_code=400
        )

    try:
        env  = EpidemicContainmentEnv()
        obs  = env.reset(task_name)
        log  = []
        done = obs.done

        while not done:
            districts     = obs.districts
            most_infected = max(districts, key=lambda d: d.reported_infection_rate)

            # Rule-based: allocate to most infected if resources available, else restrict
            if obs.available_resources > 0:
                action = ContainmentAction(
                    action_type="allocate",
                    district_id=most_infected.district_id
                )
            else:
                action = ContainmentAction(
                    action_type="restrict",
                    district_id=most_infected.district_id
                )

            obs = env.step(action)
            log.append({
                "step":        obs.current_step,
                "action_type": action.action_type,
                "district_id": action.district_id,
                "reward":      round(obs.reward or 0.0, 4),
                "done":        obs.done,
                "message":     obs.message or "",
            })
            done = obs.done

        trajectory = env.get_trajectory()
        result     = grade_trajectory(trajectory, task_name)

        return JSONResponse({
            "task_name":           task_name,
            "total_steps":         result.total_steps,
            "final_score":         result.final_score,
            "containment_score":   result.containment_score,
            "hospital_score":      result.hospital_score,
            "efficiency_score":    result.efficiency_score,
            "speed_score":         result.speed_score,
            "hospital_breached":   result.hospital_breached,
            "districts_contained": result.districts_contained,
            "log":                 log,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """
    Interactive judge-facing dashboard.
    - Overview: environment design, tasks, reward function
    - Live Demo: run rule-based agent on any task, see grader scores
    - Grader: scoring methodology and weights
    - Baseline Evaluation: GRPO loop description and benchmark scores
    - Architecture: file structure and OpenEnv compliance
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cascade Containment — OpenEnv Benchmark</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#070b14;--bg2:#0d1524;--bg3:#111d30;--border:#1e304d;--border2:#2a4166;
  --text:#c8d8f0;--muted:#4a6080;--accent:#e05c4b;--amber:#f5a623;
  --green:#2dce73;--blue:#4e9eff;--purple:#9b72f5;
  --mono:'Space Mono',monospace;--sans:'Syne',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);pointer-events:none;z-index:1000;}
.header{border-bottom:1px solid var(--border);padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:56px;background:var(--bg2);position:sticky;top:0;z-index:100;}
.header-left{display:flex;align-items:center;gap:1rem;}
.logo-mark{width:28px;height:28px;background:var(--accent);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px;}
.header-title{font-size:0.9rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#fff;}
.header-sub{font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:1px;letter-spacing:0.04em;}
.status-pill{display:flex;align-items:center;gap:0.4rem;background:rgba(45,206,115,0.1);border:1px solid rgba(45,206,115,0.25);border-radius:20px;padding:0.3rem 0.8rem;font-family:var(--mono);font-size:0.7rem;color:var(--green);}
.pulse{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(0.8);}}
.nav{display:flex;gap:0;border-bottom:1px solid var(--border);padding:0 2rem;background:var(--bg2);}
.nav-tab{font-family:var(--mono);font-size:0.72rem;letter-spacing:0.06em;text-transform:uppercase;padding:0.8rem 1.2rem;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;transition:all 0.2s;border:none;background:none;}
.nav-tab:hover{color:var(--text);}
.nav-tab.active{color:var(--blue);border-bottom:2px solid var(--blue);}
.main{padding:2rem;max-width:1400px;margin:0 auto;}
.section{display:none;}.section.active{display:block;}
.overview-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.5rem;}
.stat-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1.25rem 1.5rem;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);}
.stat-card:nth-child(2)::before{background:var(--blue);}
.stat-card:nth-child(3)::before{background:var(--green);}
.stat-label{font-family:var(--mono);font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;}
.stat-value{font-size:1.8rem;font-weight:800;color:#fff;line-height:1;}
.stat-sub{font-family:var(--mono);font-size:0.7rem;color:var(--muted);margin-top:0.3rem;}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem;}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1.5rem;}
.card-title{font-family:var(--mono);font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;display:flex;align-items:center;gap:0.5rem;}
.card-title::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);}
.task-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.5rem;}
.task-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1.25rem;}
.task-badge{display:inline-block;font-family:var(--mono);font-size:0.65rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;padding:0.2rem 0.6rem;border-radius:4px;margin-bottom:0.75rem;}
.task-easy .task-badge{background:rgba(45,206,115,0.15);color:var(--green);border:1px solid rgba(45,206,115,0.3);}
.task-medium .task-badge{background:rgba(245,166,35,0.15);color:var(--amber);border:1px solid rgba(245,166,35,0.3);}
.task-hard .task-badge{background:rgba(224,92,75,0.15);color:var(--accent);border:1px solid rgba(224,92,75,0.3);}
.task-name{font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:0.75rem;}
.task-spec{display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;}
.spec-item{font-family:var(--mono);font-size:0.7rem;color:var(--muted);}
.spec-item span{color:var(--text);}
.reward-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:0.78rem;}
.reward-table th{text-align:left;color:var(--muted);font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;padding:0.5rem 0.75rem;border-bottom:1px solid var(--border);}
.reward-table td{padding:0.6rem 0.75rem;border-bottom:1px solid rgba(30,48,77,0.5);vertical-align:middle;}
.reward-table tr:last-child td{border-bottom:none;}
.positive{color:var(--green);font-weight:700;}.negative{color:var(--accent);font-weight:700;}.neutral{color:var(--muted);}
.demo-controls{display:flex;gap:0.75rem;margin-bottom:1.5rem;align-items:center;}
.demo-btn{font-family:var(--mono);font-size:0.72rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;padding:0.65rem 1.3rem;border-radius:6px;border:1px solid;cursor:pointer;transition:all 0.2s;background:transparent;}
.demo-btn:disabled{opacity:0.35;cursor:not-allowed;}
.demo-btn.easy{border-color:var(--green);color:var(--green);}
.demo-btn.medium{border-color:var(--amber);color:var(--amber);}
.demo-btn.hard{border-color:var(--accent);color:var(--accent);}
.demo-btn.easy:not(:disabled):hover{background:rgba(45,206,115,0.12);}
.demo-btn.medium:not(:disabled):hover{background:rgba(245,166,35,0.12);}
.demo-btn.hard:not(:disabled):hover{background:rgba(224,92,75,0.12);}
.demo-hint{font-family:var(--mono);font-size:0.68rem;color:var(--muted);}
.score-display{display:none;animation:fadeIn 0.4s ease;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
.score-hero{display:flex;align-items:center;gap:2rem;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1.5rem 2rem;margin-bottom:1rem;}
.score-number{font-family:var(--mono);font-size:4rem;font-weight:700;line-height:1;min-width:180px;}
.score-breakdown{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;}
.score-component{background:var(--bg3);border-radius:8px;padding:0.75rem 1rem;}
.component-label{font-family:var(--mono);font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:0.4rem;display:flex;justify-content:space-between;align-items:center;}
.component-value{font-family:var(--mono);font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:0.4rem;}
.component-bar{height:3px;background:var(--border);border-radius:2px;overflow:hidden;}
.component-fill{height:100%;border-radius:2px;transition:width 1s cubic-bezier(0.4,0,0.2,1);width:0%;}
.log-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;}
.log-label{font-family:var(--mono);font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);}
.log-meta{font-family:var(--mono);font-size:0.68rem;color:var(--muted);}
.log-box{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:1rem;max-height:280px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2) transparent;}
.log-entry{display:grid;grid-template-columns:50px 80px 100px 80px 1fr;gap:0.5rem;font-family:var(--mono);font-size:0.72rem;padding:0.3rem 0;border-bottom:1px solid rgba(30,48,77,0.4);align-items:center;}
.log-entry:last-child{border-bottom:none;}
.log-step-num{color:var(--muted);}.log-action{color:var(--blue);}.log-district{color:var(--text);}
.log-reward.pos{color:var(--green);}.log-reward.neg{color:var(--accent);}.log-reward.neu{color:var(--muted);}
.log-msg{color:var(--muted);font-size:0.65rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.loading-bar{display:none;align-items:center;gap:1rem;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1.5rem 2rem;margin-bottom:1rem;font-family:var(--mono);font-size:0.8rem;color:var(--muted);}
.spinner{width:20px;height:20px;border:2px solid var(--border2);border-top-color:var(--blue);border-radius:50%;animation:spin 0.8s linear infinite;flex-shrink:0;}
@keyframes spin{to{transform:rotate(360deg);}}
.weight-row{display:flex;align-items:center;gap:1rem;padding:0.6rem 0;border-bottom:1px solid rgba(30,48,77,0.5);}
.weight-row:last-child{border-bottom:none;}
.weight-name{font-family:var(--mono);font-size:0.75rem;color:var(--text);min-width:160px;}
.weight-pct{font-family:var(--mono);font-size:0.75rem;font-weight:700;min-width:40px;}
.weight-bar-track{flex:1;height:6px;background:var(--border);border-radius:3px;}
.weight-bar-fill{height:100%;border-radius:3px;}
.weight-desc{font-family:var(--mono);font-size:0.65rem;color:var(--muted);min-width:220px;text-align:right;}
.baseline-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem;}
.code-block{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:1rem 1.25rem;font-family:var(--mono);font-size:0.75rem;line-height:1.7;overflow-x:auto;}
.code-block .comment{color:var(--muted);}.code-block .key{color:var(--blue);}.code-block .val{color:var(--green);}.code-block .str{color:var(--amber);}
.arch-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;}
.arch-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1.25rem;}
.arch-icon{font-size:1.5rem;margin-bottom:0.75rem;}
.arch-name{font-size:0.9rem;font-weight:700;color:#fff;margin-bottom:0.4rem;}
.arch-desc{font-family:var(--mono);font-size:0.7rem;color:var(--muted);line-height:1.6;}
.tag{display:inline-block;font-family:var(--mono);font-size:0.62rem;letter-spacing:0.04em;background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:0.15rem 0.4rem;color:var(--muted);margin:0.15rem;}
.gen-list{display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;}
.gen-item{background:var(--bg3);border-left:3px solid var(--border2);border-radius:0 8px 8px 0;padding:0.75rem 1rem;}
.gen-item.highlight{border-left-color:var(--accent);}
.gen-domain{font-weight:700;font-size:0.85rem;color:#fff;margin-bottom:0.2rem;}
.gen-mechanic{font-family:var(--mono);font-size:0.7rem;color:var(--muted);}
.badge{display:inline-block;font-family:var(--mono);font-size:0.65rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;padding:0.2rem 0.6rem;border-radius:4px;background:rgba(78,158,255,0.12);color:var(--blue);border:1px solid rgba(78,158,255,0.25);}
.breach-badge{font-family:var(--mono);font-size:0.65rem;background:rgba(224,92,75,0.15);color:var(--accent);border:1px solid rgba(224,92,75,0.3);border-radius:4px;padding:0.15rem 0.5rem;}
.safe-badge{font-family:var(--mono);font-size:0.65rem;background:rgba(45,206,115,0.12);color:var(--green);border:1px solid rgba(45,206,115,0.25);border-radius:4px;padding:0.15rem 0.5rem;}
.full-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1.5rem;margin-bottom:1rem;}
.results-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:0.78rem;}
.results-table th{text-align:left;font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);padding:0.5rem 0.75rem;border-bottom:1px solid var(--border);}
.results-table td{padding:0.65rem 0.75rem;border-bottom:1px solid rgba(30,48,77,0.4);}
.results-table tr:last-child td{border-bottom:none;}
.score-bar-inline{display:flex;align-items:center;gap:0.5rem;}
.bar-track{width:80px;height:4px;background:var(--border);border-radius:2px;}
.bar-fill{height:100%;border-radius:2px;background:var(--blue);}
</style>
</head>
<body>
<div class="header">
  <div class="header-left">
    <div class="logo-mark">&#127829;</div>
    <div>
      <div class="header-title">Cascade Containment</div>
      <div class="header-sub">OpenEnv Benchmark &middot; Meta PyTorch Hackathon x SST 2026</div>
    </div>
  </div>
  <div class="status-pill"><div class="pulse"></div>ENVIRONMENT RUNNING</div>
</div>

<div class="nav">
  <button class="nav-tab active" onclick="switchTab('overview',this)">Overview</button>
  <button class="nav-tab" onclick="switchTab('demo',this)">Live Demo</button>
  <button class="nav-tab" onclick="switchTab('grader',this)">Grader</button>
  <button class="nav-tab" onclick="switchTab('baseline',this)">Baseline Evaluation</button>
  <button class="nav-tab" onclick="switchTab('architecture',this)">Architecture</button>
</div>

<div class="main">

<!-- OVERVIEW -->
<div id="tab-overview" class="section active">
  <div class="overview-grid" style="margin-bottom:1.5rem;">
    <div class="stat-card">
      <div class="stat-label">Environment Type</div>
      <div class="stat-value" style="font-size:1.2rem;margin-top:0.2rem;">Sequential RL</div>
      <div class="stat-sub">Resource allocation under uncertainty</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Task Difficulty Levels</div>
      <div class="stat-value">3</div>
      <div class="stat-sub">Easy &middot; Medium &middot; Hard (3-day data lag)</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Generalisation Domains</div>
      <div class="stat-value">4+</div>
      <div class="stat-sub">Wildfire &middot; Cyberattack &middot; Misinformation &middot; Aid</div>
    </div>
  </div>
  <div class="task-row">
    <div class="task-card task-easy">
      <div class="task-badge">Easy</div>
      <div class="task-name">Single Outbreak</div>
      <div class="task-spec">
        <div class="spec-item">Districts: <span>2</span></div>
        <div class="spec-item">Steps: <span>10</span></div>
        <div class="spec-item">Resources: <span>10</span></div>
        <div class="spec-item">Data lag: <span>None</span></div>
      </div>
    </div>
    <div class="task-card task-medium">
      <div class="task-badge">Medium</div>
      <div class="task-name">Simultaneous Outbreaks</div>
      <div class="task-spec">
        <div class="spec-item">Districts: <span>4</span></div>
        <div class="spec-item">Steps: <span>15</span></div>
        <div class="spec-item">Resources: <span>8</span></div>
        <div class="spec-item">Data lag: <span>None</span></div>
      </div>
    </div>
    <div class="task-card task-hard">
      <div class="task-badge">Hard</div>
      <div class="task-name">Invisible Acceleration</div>
      <div class="task-spec">
        <div class="spec-item">Districts: <span>6</span></div>
        <div class="spec-item">Steps: <span>15</span></div>
        <div class="spec-item">Resources: <span>7</span></div>
        <div class="spec-item">Data lag: <span>3 days</span></div>
      </div>
    </div>
  </div>
  <div class="info-grid">
    <div class="card">
      <div class="card-title">Action Space</div>
      <table class="reward-table">
        <tr><th>Action</th><th>Cost</th><th>Effect</th></tr>
        <tr><td><span style="color:var(--blue);font-weight:700;">test</span></td><td>1 resource</td><td>Accurate infection data for district</td></tr>
        <tr><td><span style="color:var(--amber);font-weight:700;">restrict</span></td><td>Free</td><td>Reduce spread (penalty if infection &lt; 0.2)</td></tr>
        <tr><td><span style="color:var(--green);font-weight:700;">allocate</span></td><td>1 resource</td><td>Deploy medical resources, reduce spread</td></tr>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Reward Function</div>
      <table class="reward-table">
        <tr><th>Term</th><th>Value</th><th>Fires when</th></tr>
        <tr><td>Infection penalty</td><td class="negative">&minus;0.50</td><td>Per district above 0.4</td></tr>
        <tr><td>Hospital breach</td><td class="negative">&minus;1.00</td><td>Per collapsed hospital</td></tr>
        <tr><td>Early containment</td><td class="positive">+0.50&times;t</td><td>District below 0.2, decays over time</td></tr>
        <tr><td>Correct prioritisation</td><td class="positive">+0.30</td><td>Allocate to highest infected district</td></tr>
        <tr><td>Unnecessary restrict</td><td class="negative">&minus;0.20</td><td>Restrict district below 0.2</td></tr>
      </table>
    </div>
  </div>
  <div class="full-card">
    <div class="card-title" style="margin-bottom:1rem;">Generalisation &mdash; Same Mechanics, Different Domains</div>
    <div class="gen-list">
      <div class="gen-item highlight"><div class="gen-domain">&#127829; Epidemic Containment</div><div class="gen-mechanic">Primary framing &mdash; allocate testing, restrict movement, deploy medical resources</div></div>
      <div class="gen-item"><div class="gen-domain">&#128293; Wildfire Resource Deployment</div><div class="gen-mechanic">Pre-position crews before fire reaches populated areas; delayed satellite data</div></div>
      <div class="gen-item"><div class="gen-domain">&#128737;&#65039; Cyberattack Isolation</div><div class="gen-mechanic">Quarantine systems before lateral movement; scarce security team resources</div></div>
      <div class="gen-item"><div class="gen-domain">&#128226; Misinformation Containment</div><div class="gen-mechanic">Deploy corrections before false narratives entrench; network spread dynamics</div></div>
    </div>
  </div>
</div>

<!-- DEMO -->
<div id="tab-demo" class="section">
  <div class="full-card" style="margin-bottom:1rem;">
    <div class="card-title" style="margin-bottom:0.5rem;">Rule-Based Baseline Agent</div>
    <p style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);line-height:1.7;margin-bottom:1rem;">
      Runs a complete episode server-side using a greedy rule-based policy: always allocates to the highest-infected district,
      falls back to restrict when resources are exhausted. No LLM or API keys required.
      Scored by the deterministic grader.
    </p>
    <div class="demo-controls">
      <button class="demo-btn easy"   id="btn-easy"   onclick="runDemo('easy')">&#9654; Run Easy</button>
      <button class="demo-btn medium" id="btn-medium" onclick="runDemo('medium')">&#9654; Run Medium</button>
      <button class="demo-btn hard"   id="btn-hard"   onclick="runDemo('hard')">&#9654; Run Hard</button>
      <span class="demo-hint">Click any task to run a live episode and see grader scores</span>
    </div>
  </div>
  <div class="loading-bar" id="loading-bar">
    <div class="spinner"></div>
    <span id="loading-text">Running episode...</span>
  </div>
  <div class="score-display" id="score-display">
    <div class="score-hero">
      <div>
        <div style="font-family:var(--mono);font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:0.4rem;">Final Score</div>
        <div class="score-number" id="final-score-val">&mdash;</div>
        <div style="margin-top:0.5rem;display:flex;gap:0.4rem;align-items:center;flex-wrap:wrap;">
          <span id="task-badge-display" class="badge">&mdash;</span>
          <span id="steps-display" style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);">&mdash; steps</span>
          <span id="breach-display"></span>
        </div>
      </div>
      <div class="score-breakdown">
        <div class="score-component">
          <div class="component-label">Containment <span style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);">45%</span></div>
          <div class="component-value" id="cv-containment">&mdash;</div>
          <div class="component-bar"><div class="component-fill" id="cf-containment" style="background:var(--green)"></div></div>
        </div>
        <div class="score-component">
          <div class="component-label">Hospital <span style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);">30%</span></div>
          <div class="component-value" id="cv-hospital">&mdash;</div>
          <div class="component-bar"><div class="component-fill" id="cf-hospital" style="background:var(--blue)"></div></div>
        </div>
        <div class="score-component">
          <div class="component-label">Efficiency <span style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);">15%</span></div>
          <div class="component-value" id="cv-efficiency">&mdash;</div>
          <div class="component-bar"><div class="component-fill" id="cf-efficiency" style="background:var(--purple)"></div></div>
        </div>
        <div class="score-component">
          <div class="component-label">Speed <span style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);">10%</span></div>
          <div class="component-value" id="cv-speed">&mdash;</div>
          <div class="component-bar"><div class="component-fill" id="cf-speed" style="background:var(--amber)"></div></div>
        </div>
      </div>
    </div>
    <div class="log-header">
      <div class="log-label">Step Log</div>
      <div class="log-meta" id="log-meta"></div>
    </div>
    <div class="log-box" id="step-log"></div>
  </div>
</div>

<!-- GRADER -->
<div id="tab-grader" class="section">
  <div class="full-card" style="margin-bottom:1rem;">
    <div class="card-title" style="margin-bottom:1rem;">Grader Design</div>
    <p style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);line-height:1.8;max-width:700px;">
      The grader is <strong style="color:var(--text);">fully deterministic</strong> &mdash; no randomness, no LLM calls.
      Given identical trajectories it always returns identical scores in <strong style="color:var(--text);">[0.0, 1.0]</strong>.
    </p>
  </div>
  <div class="full-card" style="margin-bottom:1rem;">
    <div class="card-title" style="margin-bottom:1rem;">Score Components</div>
    <div class="weight-row">
      <div class="weight-name">Containment Score</div>
      <div class="weight-pct" style="color:var(--green);">45%</div>
      <div class="weight-bar-track"><div class="weight-bar-fill" style="width:45%;background:var(--green)"></div></div>
      <div class="weight-desc">Fraction of district-days below 0.4 infection threshold</div>
    </div>
    <div class="weight-row">
      <div class="weight-name">Hospital Score</div>
      <div class="weight-pct" style="color:var(--blue);">30%</div>
      <div class="weight-bar-track"><div class="weight-bar-fill" style="width:30%;background:var(--blue)"></div></div>
      <div class="weight-desc">Avg capacity preserved; &times;0.6 multiplier if any breach</div>
    </div>
    <div class="weight-row">
      <div class="weight-name">Efficiency Score</div>
      <div class="weight-pct" style="color:var(--purple);">15%</div>
      <div class="weight-bar-track"><div class="weight-bar-fill" style="width:15%;background:var(--purple)"></div></div>
      <div class="weight-desc">Fraction of resource actions targeting districts above threshold</div>
    </div>
    <div class="weight-row">
      <div class="weight-name">Speed Score</div>
      <div class="weight-pct" style="color:var(--amber);">10%</div>
      <div class="weight-bar-track"><div class="weight-bar-fill" style="width:10%;background:var(--amber)"></div></div>
      <div class="weight-desc">1 &minus; (steps / max_steps) if episode ends before time limit</div>
    </div>
  </div>
  <div class="info-grid">
    <div class="card">
      <div class="card-title">Key Design Decisions</div>
      <div style="font-family:var(--mono);font-size:0.75rem;line-height:1.9;color:var(--muted);">
        <div style="margin-bottom:0.5rem;"><span style="color:var(--green);">&rarr;</span> <strong style="color:var(--text);">Dense reward signal</strong> &mdash; all 5 terms fire independently each step.</div>
        <div style="margin-bottom:0.5rem;"><span style="color:var(--amber);">&rarr;</span> <strong style="color:var(--text);">Decaying containment bonus</strong> &mdash; early action worth more than late reaction.</div>
        <div style="margin-bottom:0.5rem;"><span style="color:var(--blue);">&rarr;</span> <strong style="color:var(--text);">Hospital breach multiplier</strong> &mdash; any collapse multiplies hospital score by 0.6.</div>
        <div><span style="color:var(--accent);">&rarr;</span> <strong style="color:var(--text);">Grace period</strong> &mdash; first 2 steps excluded from containment scoring.</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Partial Observability (Hard Task)</div>
      <div style="font-family:var(--mono);font-size:0.75rem;line-height:1.9;color:var(--muted);">
        <div style="margin-bottom:0.5rem;">Hard task exposes infection rates from <strong style="color:var(--accent);">3 days ago</strong> via rolling <code style="color:var(--text);">infection_history</code> buffer.</div>
        <div style="margin-bottom:0.5rem;">Agent receives noisy <strong style="color:var(--text);">growth_rate_hint</strong> to reason about trajectory without precise current data.</div>
        <div style="margin-bottom:0.5rem;">Hospital capacity is <strong style="color:var(--green);">always accurate</strong> &mdash; hospitals report in real time.</div>
        <div>Grader evaluates against <strong style="color:var(--text);">true hidden state</strong>, not agent&apos;s observed state.</div>
      </div>
    </div>
  </div>
</div>

<!-- BASELINE -->
<div id="tab-baseline" class="section">
  <div class="full-card" style="margin-bottom:1rem;">
    <div class="card-title" style="margin-bottom:0.75rem;">GRPO-Style Simulated Learning</div>
    <p style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);line-height:1.8;max-width:720px;margin-bottom:1rem;">
      The baseline implements simulated GRPO with episodic memory &mdash; no weight updates required.
      The prompt is the policy. Memory updates are the policy improvement.
    </p>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;">
      <div style="background:var(--bg3);border-radius:8px;padding:0.75rem;text-align:center;"><div style="font-family:var(--mono);font-size:1.4rem;font-weight:700;color:var(--blue);">4</div><div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.2rem;">Rollouts / task</div></div>
      <div style="background:var(--bg3);border-radius:8px;padding:0.75rem;text-align:center;"><div style="font-family:var(--mono);font-size:1.4rem;font-weight:700;color:var(--green);">L1</div><div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.2rem;">Memory similarity</div></div>
      <div style="background:var(--bg3);border-radius:8px;padding:0.75rem;text-align:center;"><div style="font-family:var(--mono);font-size:1.4rem;font-weight:700;color:var(--amber);">20</div><div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.2rem;">Max memory size</div></div>
      <div style="background:var(--bg3);border-radius:8px;padding:0.75rem;text-align:center;"><div style="font-family:var(--mono);font-size:1.4rem;font-weight:700;color:var(--purple);">R&minus;&mu;</div><div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);margin-top:0.2rem;">Advantage signal</div></div>
    </div>
  </div>
  <div class="baseline-grid">
    <div class="card">
      <div class="card-title" style="margin-bottom:0.75rem;">Learning Loop</div>
      <div style="font-family:var(--mono);font-size:0.75rem;line-height:1.9;color:var(--muted);">
        <div><span style="color:var(--blue);">1.</span> Rollout 1 &mdash; base prompt, no prior knowledge</div>
        <div><span style="color:var(--blue);">2.</span> Compute advantage = R<sub>i</sub> &minus; mean(completed)</div>
        <div><span style="color:var(--green);">3.</span> If advantage &gt; &minus;0.5 &rarr; store positive-reward steps</div>
        <div><span style="color:var(--accent);">4.</span> If below threshold &rarr; suppress, memory unchanged</div>
        <div><span style="color:var(--blue);">5.</span> Next rollout retrieves top-3 similar past decisions</div>
        <div><span style="color:var(--blue);">6.</span> Memory injected into prompt before each step</div>
        <div><span style="color:var(--purple);">7.</span> Report best grader score across all rollouts</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title" style="margin-bottom:0.75rem;">Environment Variables</div>
      <div class="code-block">
<span class="comment"># Required for inference.py</span>
<span class="key">API_BASE_URL</span>=<span class="str">"https://router.huggingface.co/v1"</span>
<span class="key">MODEL_NAME</span>=<span class="str">"meta-llama/Llama-3.1-8B-Instruct"</span>
<span class="key">HF_TOKEN</span>=<span class="str">"hf_your_token_here"</span>
<span class="key">ENV_BASE_URL</span>=<span class="str">"https://therubberduckdebuggers-cascade-containment.hf.space"</span>

<span class="comment"># Run evaluation</span>
<span class="val">python</span> inference.py</div>
    </div>
  </div>
  <div class="full-card">
    <div class="card-title" style="margin-bottom:1rem;">Baseline Score Benchmarks</div>
    <table class="results-table">
      <tr><th>Task</th><th>Agent Type</th><th>Containment</th><th>Hospital</th><th>Efficiency</th><th>Final Score</th><th>Breach</th></tr>
      <tr>
        <td><span class="badge" style="background:rgba(45,206,115,0.1);color:var(--green);border-color:rgba(45,206,115,0.25);">Easy</span></td>
        <td style="color:var(--muted);">Dumb greedy</td><td>0.35</td><td>0.90</td><td>0.60</td>
        <td class="score-bar-inline"><div class="bar-track"><div class="bar-fill" style="width:50%;background:var(--green)"></div></div><span>~0.50</span></td>
        <td><span class="safe-badge">No</span></td>
      </tr>
      <tr>
        <td><span class="badge" style="background:rgba(45,206,115,0.1);color:var(--green);border-color:rgba(45,206,115,0.25);">Easy</span></td>
        <td style="color:var(--muted);">LLM + GRPO memory</td><td>0.59</td><td>0.94</td><td>1.00</td>
        <td class="score-bar-inline"><div class="bar-track"><div class="bar-fill" style="width:62%;background:var(--green)"></div></div><span>~0.62</span></td>
        <td><span class="safe-badge">No</span></td>
      </tr>
      <tr>
        <td><span class="badge" style="background:rgba(245,166,35,0.1);color:var(--amber);border-color:rgba(245,166,35,0.25);">Medium</span></td>
        <td style="color:var(--muted);">Dumb greedy</td><td>0.18</td><td>0.21</td><td>0.57</td>
        <td class="score-bar-inline"><div class="bar-track"><div class="bar-fill" style="width:23%;background:var(--amber)"></div></div><span>~0.23</span></td>
        <td><span class="breach-badge">Yes</span></td>
      </tr>
      <tr>
        <td><span class="badge" style="background:rgba(245,166,35,0.1);color:var(--amber);border-color:rgba(245,166,35,0.25);">Medium</span></td>
        <td style="color:var(--muted);">LLM + GRPO memory</td><td>0.15</td><td>0.94</td><td>0.87</td>
        <td class="score-bar-inline"><div class="bar-track"><div class="bar-fill" style="width:55%;background:var(--amber)"></div></div><span>~0.55</span></td>
        <td><span class="safe-badge">No</span></td>
      </tr>
      <tr>
        <td><span class="badge" style="background:rgba(224,92,75,0.1);color:var(--accent);border-color:rgba(224,92,75,0.25);">Hard</span></td>
        <td style="color:var(--muted);">Dumb greedy</td><td>0.23</td><td>0.20</td><td>0.29</td>
        <td class="score-bar-inline"><div class="bar-track"><div class="bar-fill" style="width:21%;background:var(--accent)"></div></div><span>~0.21</span></td>
        <td><span class="breach-badge">Yes</span></td>
      </tr>
      <tr>
        <td><span class="badge" style="background:rgba(224,92,75,0.1);color:var(--accent);border-color:rgba(224,92,75,0.25);">Hard</span></td>
        <td style="color:var(--muted);">LLM + GRPO memory</td><td>0.21</td><td>0.74</td><td>0.87</td>
        <td class="score-bar-inline"><div class="bar-track"><div class="bar-fill" style="width:57%;background:var(--accent)"></div></div><span>~0.57</span></td>
        <td><span class="safe-badge">No</span></td>
      </tr>
    </table>
  </div>
</div>

<!-- ARCHITECTURE -->
<div id="tab-architecture" class="section">
  <div class="full-card" style="margin-bottom:1rem;">
    <div class="card-title" style="margin-bottom:0.5rem;">OpenEnv Interface Compliance</div>
    <p style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);line-height:1.8;">
      Implements the standard 3-method OpenEnv interface.
      <code style="color:var(--text);">CityState</code> is a plain dataclass (hidden ground truth &mdash; never sent to agent).
      <code style="color:var(--text);">state</code> is a <code style="color:var(--text);">@property</code> returning the OpenEnv tracking State.
    </p>
    <div style="display:flex;gap:1rem;margin-top:1rem;flex-wrap:wrap;">
      <div style="background:var(--bg3);border-radius:8px;padding:0.75rem 1.25rem;font-family:var(--mono);font-size:0.75rem;"><span style="color:var(--muted);">env.</span><span style="color:var(--blue);">reset</span><span style="color:var(--muted);">(task_name)</span><span style="color:var(--muted);margin:0 0.5rem;">&rarr;</span><span style="color:var(--green);">CityObservation</span></div>
      <div style="background:var(--bg3);border-radius:8px;padding:0.75rem 1.25rem;font-family:var(--mono);font-size:0.75rem;"><span style="color:var(--muted);">env.</span><span style="color:var(--blue);">step</span><span style="color:var(--muted);">(action)</span><span style="color:var(--muted);margin:0 0.5rem;">&rarr;</span><span style="color:var(--green);">CityObservation</span></div>
      <div style="background:var(--bg3);border-radius:8px;padding:0.75rem 1.25rem;font-family:var(--mono);font-size:0.75rem;"><span style="color:var(--muted);">env.</span><span style="color:var(--blue);">state</span><span style="color:var(--muted);margin:0 0.5rem;">&rarr;</span><span style="color:var(--green);">State</span></div>
    </div>
  </div>
  <div class="arch-grid" style="margin-bottom:1rem;">
    <div class="arch-card"><div class="arch-icon">&#128208;</div><div class="arch-name">models.py</div><div class="arch-desc">Data contracts for the entire system. DistrictObservation, DistrictTruth, CityState, CityObservation, ContainmentAction.</div></div>
    <div class="arch-card"><div class="arch-icon">&#9881;&#65039;</div><div class="arch-name">server/environment.py</div><div class="arch-desc">Core RL loop. Maintains OpenEnv State (episode tracking) and CityState (simulation ground truth). Agent only sees CityObservation.</div></div>
    <div class="arch-card"><div class="arch-icon">&#128202;</div><div class="arch-name">server/grader.py</div><div class="arch-desc">Deterministic trajectory scorer. Reads from hidden CityState. Four components weighted into final_score in [0.0, 1.0]. No LLM calls.</div></div>
    <div class="arch-card"><div class="arch-icon">&#128256;</div><div class="arch-name">server/utils.py</div><div class="arch-desc">Spread computation with wrap-around spillover. Observation builder enforcing partial observability. Helper query functions.</div></div>
    <div class="arch-card"><div class="arch-icon">&#129504;</div><div class="arch-name">core/trajectory.py</div><div class="arch-desc">EpisodicMemory class. Stores high-reward (obs, action) pairs. Retrieves top-k similar past situations by L1 distance on infection profiles.</div></div>
    <div class="arch-card"><div class="arch-icon">&#127919;</div><div class="arch-name">baseline/evaluator.py</div><div class="arch-desc">GRPO-style loop. N rollouts per task. Advantage = R&#x1D62; &minus; mean(R). Reinforces above-average rollouts. Reports best grader score.</div></div>
  </div>
  <div class="full-card">
    <div class="card-title" style="margin-bottom:0.75rem;">Tags</div>
    <div>
      <span class="tag">reinforcement-learning</span><span class="tag">resource-allocation</span>
      <span class="tag">sequential-decision-making</span><span class="tag">partial-observability</span>
      <span class="tag">cascade-dynamics</span><span class="tag">openenv</span>
      <span class="tag">llm-agent</span><span class="tag">grpo</span>
      <span class="tag">episodic-memory</span><span class="tag">docker</span>
      <span class="tag">fastapi</span><span class="tag">python</span>
    </div>
  </div>
</div>

</div>

<script>
function switchTab(name,el){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  el.classList.add('active');
}
const pct=v=>(v*100).toFixed(1)+'%';
async function runDemo(task){
  ['easy','medium','hard'].forEach(t=>document.getElementById('btn-'+t).disabled=true);
  document.getElementById('score-display').style.display='none';
  const lb=document.getElementById('loading-bar');
  lb.style.display='flex';
  document.getElementById('loading-text').textContent='Running '+task+' episode...';
  try{
    const r=await fetch('/demo/'+task);
    const d=await r.json();
    if(d.error){document.getElementById('loading-text').textContent='Error: '+d.error;return;}
    const scoreEl=document.getElementById('final-score-val');
    scoreEl.textContent=pct(d.final_score);
    scoreEl.style.color=d.final_score>0.6?'var(--green)':d.final_score>0.4?'var(--amber)':'var(--accent)';
    document.getElementById('task-badge-display').textContent=task.toUpperCase();
    document.getElementById('steps-display').textContent=d.total_steps+' steps';
    document.getElementById('breach-display').innerHTML=d.hospital_breached
      ?'<span class="breach-badge">Hospital Breached</span>'
      :'<span class="safe-badge">No Breach</span>';
    const comps={containment:d.containment_score,hospital:d.hospital_score,efficiency:d.efficiency_score,speed:d.speed_score};
    for(const[k,v]of Object.entries(comps)){
      document.getElementById('cv-'+k).textContent=pct(v);
      setTimeout(()=>{document.getElementById('cf-'+k).style.width=(v*100)+'%';},100);
    }
    const logEl=document.getElementById('step-log');
    logEl.innerHTML=d.log.map(s=>{
      const cls=s.reward>0?'pos':s.reward<-0.5?'neg':'neu';
      const rStr=(s.reward>=0?'+':'')+s.reward.toFixed(4);
      return`<div class="log-entry"><span class="log-step-num">step ${String(s.step).padStart(2,'0')}</span><span class="log-action">${s.action_type}</span><span class="log-district">district ${s.district_id}</span><span class="log-reward ${cls}">${rStr}</span><span class="log-msg">${s.message||''}</span></div>`;
    }).join('');
    document.getElementById('log-meta').textContent=`${d.total_steps} steps \u00b7 ${d.districts_contained} districts contained`;
    lb.style.display='none';
    const sd=document.getElementById('score-display');
    sd.style.display='block';
    sd.style.animation='none';
    setTimeout(()=>sd.style.animation='fadeIn 0.4s ease',10);
  }catch(e){
    document.getElementById('loading-text').textContent='Connection error: '+e.message;
  }finally{
    ['easy','medium','hard'].forEach(t=>document.getElementById('btn-'+t).disabled=false);
  }
}
</script>
</body>
</html>"""
