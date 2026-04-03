# server/app.py
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
    if not env_module._last_grade:
        return JSONResponse(
            {"error": "No completed episode yet"},
            status_code=400
        )
    return JSONResponse(env_module._last_grade)


# ── Info endpoint ─────────────────────────────────────────────────────────────

@app.get("/info")
async def environment_info():
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
        "reward_terms": [
            {"name": "Infection penalty",        "value": -0.50, "condition": "per district above 0.4 threshold"},
            {"name": "Hospital breach",           "value": -1.00, "condition": "per collapsed hospital"},
            {"name": "Early containment bonus",   "value": "+0.50 × (1 - step/max_steps)", "condition": "per contained district"},
            {"name": "Unnecessary restriction",   "value": -0.20, "condition": "restricting district below 0.2"},
            {"name": "Correct prioritisation",    "value": +0.30, "condition": "allocating to highest-infected district"},
        ],
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
    Run a quick demo episode using a rule-based agent (no LLM required).
    Returns grader score and step-by-step log for display.
    """
    if task_name not in ["easy", "medium", "hard"]:
        return JSONResponse({"error": "task_name must be easy, medium, or hard"}, status_code=400)

    try:
        env  = EpidemicContainmentEnv()
        obs  = env.reset(task_name)
        log  = []
        done = obs.done

        while not done:
            # Rule-based agent: allocate to most infected, restrict if no resources
            districts     = obs.districts
            most_infected = max(districts, key=lambda d: d.reported_infection_rate)

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
                "message":     obs.message,
            })
            done = obs.done

        # Grade the episode
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
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cascade Containment — RL Environment</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem; }
  h1 { font-size: 1.8rem; color: #f1f5f9; margin-bottom: 0.25rem; }
  .subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 2rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; }
  .card h2 { font-size: 1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem; }
  .status { display: flex; align-items: center; gap: 0.5rem; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #22c55e; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .tag { display: inline-block; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.8rem; color: #94a3b8; margin: 0.2rem; }
  .btn { border: none; border-radius: 8px; padding: 0.6rem 1.2rem; font-size: 0.9rem; cursor: pointer; transition: all 0.2s; font-weight: 600; }
  .btn-easy   { background: #166534; color: #86efac; }
  .btn-medium { background: #854d0e; color: #fde68a; }
  .btn-hard   { background: #7f1d1d; color: #fca5a5; }
  .btn:hover  { opacity: 0.85; transform: translateY(-1px); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 1rem; }
  .score-item { background: #0f172a; border-radius: 8px; padding: 0.75rem; }
  .score-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; }
  .score-value { font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem; }
  .score-bar { height: 4px; background: #1e293b; border-radius: 2px; margin-top: 0.5rem; }
  .score-fill { height: 100%; border-radius: 2px; transition: width 0.8s ease; }
  .log-container { background: #0f172a; border-radius: 8px; padding: 1rem; max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.8rem; margin-top: 1rem; }
  .log-step { padding: 0.2rem 0; border-bottom: 1px solid #1e293b; color: #94a3b8; }
  .log-step span { color: #38bdf8; }
  .positive { color: #4ade80; }
  .negative { color: #f87171; }
  .neutral  { color: #94a3b8; }
  .reward-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .reward-table th { text-align: left; color: #64748b; padding: 0.5rem; border-bottom: 1px solid #334155; }
  .reward-table td { padding: 0.5rem; border-bottom: 1px solid #1e293b; }
  .big-score { font-size: 3rem; font-weight: 800; text-align: center; padding: 1rem 0; }
  .task-btns { display: flex; gap: 0.75rem; margin-bottom: 1rem; }
  .loading { color: #94a3b8; font-style: italic; }
  .breach-badge { display: inline-block; background: #7f1d1d; color: #fca5a5; border-radius: 4px; padding: 0.1rem 0.4rem; font-size: 0.75rem; margin-left: 0.5rem; }
  .safe-badge   { display: inline-block; background: #166534; color: #86efac; border-radius: 4px; padding: 0.1rem 0.4rem; font-size: 0.75rem; margin-left: 0.5rem; }
</style>
</head>
<body>

<h1>🦠 Cascade Containment</h1>
<p class="subtitle">RL benchmark for epidemic containment policy — Meta PyTorch OpenEnv Hackathon x SST 2026</p>

<div class="grid">

  <!-- Status Card -->
  <div class="card">
    <h2>Environment Status</h2>
    <div class="status" style="margin-bottom:1rem;">
      <div class="dot"></div>
      <span style="color:#22c55e;font-weight:600;">Running</span>
    </div>
    <div style="margin-bottom:0.75rem;">
      <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.4rem;">ENDPOINTS</div>
      <div><span class="tag">GET /health</span><span class="tag">POST /reset</span><span class="tag">POST /step</span><span class="tag">GET /grade</span><span class="tag">GET /demo/{task}</span></div>
    </div>
    <div>
      <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.4rem;">TASKS</div>
      <div><span class="tag">easy — 2 districts</span><span class="tag">medium — 4 districts</span><span class="tag">hard — 6 districts + 3d lag</span></div>
    </div>
  </div>

  <!-- Reward Function Card -->
  <div class="card">
    <h2>Reward Function</h2>
    <table class="reward-table">
      <tr><th>Term</th><th>Value</th></tr>
      <tr><td>Infection penalty</td><td class="negative">-0.50 / district</td></tr>
      <tr><td>Hospital breach</td><td class="negative">-1.00 / district</td></tr>
      <tr><td>Early containment</td><td class="positive">+0.50 × time_factor</td></tr>
      <tr><td>Unnecessary restrict</td><td class="negative">-0.20</td></tr>
      <tr><td>Correct prioritisation</td><td class="positive">+0.30</td></tr>
    </table>
  </div>

</div>

<!-- Live Demo Card -->
<div class="card" style="margin-bottom:1.5rem;">
  <h2>Live Demo — Rule-Based Agent</h2>
  <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;">
    Runs a complete episode using a greedy rule-based agent (always allocates to highest-infected district).
    Scored by the deterministic grader — containment, hospital, efficiency, speed.
  </p>
  <div class="task-btns">
    <button class="btn btn-easy"   onclick="runDemo('easy')"  id="btn-easy">▶ Run Easy</button>
    <button class="btn btn-medium" onclick="runDemo('medium')" id="btn-medium">▶ Run Medium</button>
    <button class="btn btn-hard"   onclick="runDemo('hard')"  id="btn-hard">▶ Run Hard</button>
  </div>

  <div id="demo-results" style="display:none;">
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
      <div>
        <div style="color:#64748b;font-size:0.8rem;">FINAL SCORE</div>
        <div class="big-score" id="final-score" style="color:#38bdf8;">—</div>
      </div>
      <div style="flex:1;">
        <div class="score-grid">
          <div class="score-item">
            <div class="score-label">Containment</div>
            <div class="score-value positive" id="s-containment">—</div>
            <div class="score-bar"><div class="score-fill" id="b-containment" style="background:#4ade80;width:0%"></div></div>
          </div>
          <div class="score-item">
            <div class="score-label">Hospital</div>
            <div class="score-value" id="s-hospital">—</div>
            <div class="score-bar"><div class="score-fill" id="b-hospital" style="background:#38bdf8;width:0%"></div></div>
          </div>
          <div class="score-item">
            <div class="score-label">Efficiency</div>
            <div class="score-value" id="s-efficiency">—</div>
            <div class="score-bar"><div class="score-fill" id="b-efficiency" style="background:#a78bfa;width:0%"></div></div>
          </div>
          <div class="score-item">
            <div class="score-label">Speed</div>
            <div class="score-value" id="s-speed">—</div>
            <div class="score-bar"><div class="score-fill" id="b-speed" style="background:#fb923c;width:0%"></div></div>
          </div>
        </div>
      </div>
    </div>

    <div id="meta-info" style="color:#94a3b8;font-size:0.85rem;margin-bottom:0.75rem;"></div>

    <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.4rem;">STEP LOG</div>
    <div class="log-container" id="step-log"></div>
  </div>

  <div id="demo-loading" style="display:none;" class="loading">Running episode...</div>
</div>

<!-- Grader Weights Card -->
<div class="card">
  <h2>Grader Weights</h2>
  <div class="score-grid">
    <div class="score-item">
      <div class="score-label">Containment</div>
      <div class="score-value" style="color:#4ade80;">45%</div>
      <div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem;">District-days below 0.4 threshold</div>
    </div>
    <div class="score-item">
      <div class="score-label">Hospital</div>
      <div class="score-value" style="color:#38bdf8;">30%</div>
      <div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem;">Capacity preserved across episode</div>
    </div>
    <div class="score-item">
      <div class="score-label">Efficiency</div>
      <div class="score-value" style="color:#a78bfa;">15%</div>
      <div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem;">Resources directed to high-need districts</div>
    </div>
    <div class="score-item">
      <div class="score-label">Speed</div>
      <div class="score-value" style="color:#fb923c;">10%</div>
      <div style="color:#64748b;font-size:0.75rem;margin-top:0.3rem;">Containment faster than max steps</div>
    </div>
  </div>
</div>

<script>
async function runDemo(taskName) {
  ['easy','medium','hard'].forEach(t => {
    document.getElementById('btn-'+t).disabled = true;
  });
  document.getElementById('demo-results').style.display = 'none';
  document.getElementById('demo-loading').style.display = 'block';

  try {
    const resp = await fetch('/demo/' + taskName);
    const data = await resp.json();

    if (data.error) {
      document.getElementById('demo-loading').textContent = 'Error: ' + data.error;
      return;
    }

    // Update scores
    const fmt = v => (v * 100).toFixed(1) + '%';
    document.getElementById('final-score').textContent     = fmt(data.final_score);
    document.getElementById('s-containment').textContent   = fmt(data.containment_score);
    document.getElementById('s-hospital').textContent      = fmt(data.hospital_score);
    document.getElementById('s-efficiency').textContent    = fmt(data.efficiency_score);
    document.getElementById('s-speed').textContent         = fmt(data.speed_score);

    // Update bars
    document.getElementById('b-containment').style.width  = (data.containment_score * 100) + '%';
    document.getElementById('b-hospital').style.width     = (data.hospital_score * 100) + '%';
    document.getElementById('b-efficiency').style.width   = (data.efficiency_score * 100) + '%';
    document.getElementById('b-speed').style.width        = (data.speed_score * 100) + '%';

    // Color final score by value
    const scoreEl = document.getElementById('final-score');
    scoreEl.style.color = data.final_score > 0.6 ? '#4ade80' :
                          data.final_score > 0.4 ? '#fbbf24' : '#f87171';

    // Meta info
    const breachBadge = data.hospital_breached
      ? '<span class="breach-badge">Hospital Breached</span>'
      : '<span class="safe-badge">No Breach</span>';
    document.getElementById('meta-info').innerHTML =
      `Task: <strong style="color:#f1f5f9">${data.task_name.toUpperCase()}</strong> &nbsp;|&nbsp; ` +
      `Steps: <strong style="color:#f1f5f9">${data.total_steps}</strong> &nbsp;|&nbsp; ` +
      `Districts contained: <strong style="color:#f1f5f9">${data.districts_contained}</strong>` +
      breachBadge;

    // Step log
    const logEl = document.getElementById('step-log');
    logEl.innerHTML = data.log.map(s => {
      const cls = s.reward > 0 ? 'positive' : s.reward < -0.5 ? 'negative' : 'neutral';
      return `<div class="log-step">
        Step ${String(s.step).padStart(2,'0')} &nbsp;
        <span>${s.action_type}</span> → district ${s.district_id} &nbsp;
        <span class="${cls}">${s.reward >= 0 ? '+' : ''}${s.reward.toFixed(4)}</span>
        ${s.message ? '&nbsp;<span style="color:#475569">' + s.message + '</span>' : ''}
      </div>`;
    }).join('');

    document.getElementById('demo-loading').style.display  = 'none';
    document.getElementById('demo-results').style.display  = 'block';

  } catch(e) {
    document.getElementById('demo-loading').textContent = 'Connection error: ' + e.message;
  } finally {
    ['easy','medium','hard'].forEach(t => {
      document.getElementById('btn-'+t).disabled = false;
    });
  }
}
</script>

</body>
</html>
"""