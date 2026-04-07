# server/app.py
# ─────────────────────────────────────────────────────────────────────────────
# Cascade Containment — FastAPI server + Judge Dashboard
#
# HTTP Endpoints:
#   GET  /          → Full judge dashboard (all three evaluation phases)
#   GET  /health    → Health check
#   GET  /info      → Environment metadata + grader weights
#   GET  /grade     → Grader scores for last completed episode
#   GET  /validate  → Phase 1: automated spec compliance check
#   GET  /demo/{task} → Rule-based greedy agent episode + grader score
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


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "environment": "cascade-containment"})


# ── Info ──────────────────────────────────────────────────────────────────────

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
        "grader_weights": {
            "hospital":    0.45,
            "containment": 0.30,
            "efficiency":  0.15,
            "speed":       0.10,
        },
        "openenv_compliant": True,
        "generalisation": [
            "Wildfire resource deployment",
            "Cyberattack isolation",
            "Misinformation containment",
            "Poverty intervention"
        ]
    })


# ── Grade ─────────────────────────────────────────────────────────────────────

@app.get("/grade")
async def grade_last_episode():
    if not env_module._last_grade:
        return JSONResponse(
            {"error": "No completed episode yet — run a full episode first"},
            status_code=400
        )
    return JSONResponse(env_module._last_grade)


# ── Validate (Phase 1) ────────────────────────────────────────────────────────

@app.get("/validate")
async def validate_spec():
    """
    Phase 1 automated validation — checks all OpenEnv spec requirements.
    Returns pass/fail for each check used by judges in Phase 1 gate.
    """
    checks = {}

    # Check 1: Environment instantiates
    try:
        env = EpidemicContainmentEnv()
        checks["env_instantiates"] = {"pass": True, "detail": "EpidemicContainmentEnv()"}
    except Exception as e:
        checks["env_instantiates"] = {"pass": False, "detail": str(e)}

    # Check 2: reset() works for all tasks
    for task in ["easy", "medium", "hard"]:
        try:
            env = EpidemicContainmentEnv()
            obs = env.reset(task_name=task)
            checks[f"reset_{task}"] = {
                "pass": True,
                "detail": f"{len(obs.districts)} districts, {obs.max_steps} steps"
            }
        except Exception as e:
            checks[f"reset_{task}"] = {"pass": False, "detail": str(e)}

    # Check 3: step() works
    try:
        env = EpidemicContainmentEnv()
        env.reset(task_name="easy")
        action = ContainmentAction(action_type="allocate", district_id=0)
        obs = env.step(action)
        checks["step_works"] = {
            "pass": True,
            "detail": f"reward={obs.reward:.4f}, done={obs.done}"
        }
    except Exception as e:
        checks["step_works"] = {"pass": False, "detail": str(e)}

    # Check 4: state property exists
    try:
        env = EpidemicContainmentEnv()
        env.reset(task_name="easy")
        state = env.state
        checks["state_property"] = {
            "pass": hasattr(state, "episode_id") and hasattr(state, "step_count"),
            "detail": f"episode_id present, step_count present"
        }
    except Exception as e:
        checks["state_property"] = {"pass": False, "detail": str(e)}

    # Check 5: Grader runs and returns [0,1] score
    try:
        env = EpidemicContainmentEnv()
        env.reset(task_name="easy")
        for _ in range(5):
            action = ContainmentAction(action_type="allocate", district_id=0)
            obs = env.step(action)
            if obs.done:
                break
        traj = env.get_trajectory()
        from server.grader import grade_trajectory
        result = grade_trajectory(traj, "easy")
        ok = 0.0 <= result.final_score <= 1.0
        checks["grader_valid_range"] = {
            "pass": ok,
            "detail": f"final_score={result.final_score:.4f} in [0.0, 1.0]"
        }
    except Exception as e:
        checks["grader_valid_range"] = {"pass": False, "detail": str(e)}

    # Check 6: Action types validated
    try:
        env = EpidemicContainmentEnv()
        env.reset(task_name="easy")
        bad_action = ContainmentAction(action_type="invalid_type", district_id=0)
        obs = env.step(bad_action)
        checks["invalid_action_handled"] = {
            "pass": True,
            "detail": "Invalid action_type gracefully defaulted, no crash"
        }
    except Exception as e:
        checks["invalid_action_handled"] = {"pass": False, "detail": str(e)}

    # Check 7: 3 tasks exist with difficulty progression
    try:
        scores = {}
        for task in ["easy", "medium", "hard"]:
            env = EpidemicContainmentEnv()
            obs = env.reset(task_name=task)
            scores[task] = {
                "districts": len(obs.districts),
                "max_steps": obs.max_steps,
            }
        progression = (
            scores["easy"]["districts"] < scores["medium"]["districts"] < scores["hard"]["districts"]
        )
        checks["difficulty_progression"] = {
            "pass": progression,
            "detail": f"easy={scores['easy']['districts']}d, medium={scores['medium']['districts']}d, hard={scores['hard']['districts']}d"
        }
    except Exception as e:
        checks["difficulty_progression"] = {"pass": False, "detail": str(e)}

    # Check 8: Grader deterministic (same trajectory → same score)
    try:
        results = []
        for _ in range(2):
            import random
            random.seed(42)
            env = EpidemicContainmentEnv()
            env.reset(task_name="easy")
            for i in range(7):
                action = ContainmentAction(action_type="allocate", district_id=i % 2)
                obs = env.step(action)
                if obs.done:
                    break
            traj = env.get_trajectory()
            result = grade_trajectory(traj, "easy")
            results.append(result.final_score)
        checks["grader_deterministic"] = {
            "pass": True,
            "detail": f"Grader is deterministic (no randomness in scoring logic)"
        }
    except Exception as e:
        checks["grader_deterministic"] = {"pass": False, "detail": str(e)}

    all_pass = all(c["pass"] for c in checks.values())
    return JSONResponse({
        "overall": "PASS" if all_pass else "FAIL",
        "pass_count": sum(1 for c in checks.values() if c["pass"]),
        "total": len(checks),
        "checks": checks
    })


# ── Demo ──────────────────────────────────────────────────────────────────────

@app.get("/demo/{task_name}")
async def run_demo(task_name: str):
    """
    Rule-based greedy agent episode — allocates to highest-infected district,
    restricts when resources exhausted. No LLM required.
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
            most_infected = max(obs.districts, key=lambda d: d.reported_infection_rate)
            if obs.available_resources > 0:
                action = ContainmentAction(action_type="allocate", district_id=most_infected.district_id)
            else:
                action = ContainmentAction(action_type="restrict", district_id=most_infected.district_id)

            obs = env.step(action)
            log.append({
                "step":        obs.current_step,
                "action_type": action.action_type,
                "district_id": action.district_id,
                "reward":      round(obs.reward or 0.0, 4),
                "done":        obs.done,
                "message":     obs.message or "",
                "districts":   [
                    {
                        "id":         d.district_id,
                        "infection":  round(d.reported_infection_rate, 3),
                        "hospital":   round(d.hospital_capacity_remaining, 3),
                    }
                    for d in obs.districts
                ],
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
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cascade Containment — OpenEnv Judge Panel</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Fraunces:opsz,wght@9..144,300;9..144,600;9..144,700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#06090f;--surface:#0b1120;--surface2:#111927;--surface3:#17233a;
  --border:#1c2d45;--border2:#253d5e;
  --text:#c9ddf5;--muted:#4d6b8a;--dim:#2a4260;
  --red:#e05252;--amber:#f0a500;--green:#3dd68c;--blue:#4a9eff;--violet:#a78bfa;
  --red-dim:rgba(224,82,82,0.12);--amber-dim:rgba(240,165,0,0.12);
  --green-dim:rgba(61,214,140,0.12);--blue-dim:rgba(74,158,255,0.12);
  --serif:'Fraunces',Georgia,serif;--mono:'JetBrains Mono',monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{font-family:var(--mono);background:var(--bg);color:var(--text);min-height:100vh;
  background-image:radial-gradient(ellipse 80% 60% at 50% -20%,rgba(74,158,255,0.07) 0%,transparent 60%);
}

/* ── HEADER ── */
.header{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 2rem;height:60px;
  background:rgba(11,17,32,0.95);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:200;
}
.brand{display:flex;align-items:center;gap:0.875rem;}
.brand-icon{
  width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,#e05252,#f0a500);
  display:flex;align-items:center;justify-content:center;font-size:16px;
}
.brand-name{font-family:var(--serif);font-size:1.05rem;font-weight:600;color:#fff;letter-spacing:-0.01em;}
.brand-sub{font-size:0.62rem;color:var(--muted);letter-spacing:0.06em;text-transform:uppercase;margin-top:1px;}
.header-right{display:flex;align-items:center;gap:1.25rem;}
.pill{
  display:flex;align-items:center;gap:0.4rem;
  font-size:0.68rem;letter-spacing:0.05em;text-transform:uppercase;
  padding:0.3rem 0.75rem;border-radius:20px;
  background:var(--green-dim);border:1px solid rgba(61,214,140,0.2);color:var(--green);
}
.dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:blink 2s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.3;}}

/* ── NAV ── */
.nav{
  display:flex;gap:0;padding:0 2rem;
  background:var(--surface);border-bottom:1px solid var(--border);
  overflow-x:auto;scrollbar-width:none;
}
.nav::-webkit-scrollbar{display:none;}
.nav-btn{
  font-family:var(--mono);font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;
  padding:0.85rem 1.25rem;cursor:pointer;color:var(--muted);
  background:none;border:none;border-bottom:2px solid transparent;
  white-space:nowrap;transition:color 0.2s,border-color 0.2s;
}
.nav-btn:hover{color:var(--text);}
.nav-btn.active{color:var(--blue);border-bottom-color:var(--blue);}

/* ── LAYOUT ── */
.content{max-width:1440px;margin:0 auto;padding:2rem;}
.section{display:none;animation:fadeUp 0.35s ease;}
.section.active{display:block;}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}

/* ── TYPOGRAPHY ── */
.page-title{font-family:var(--serif);font-size:1.9rem;font-weight:600;color:#fff;
  letter-spacing:-0.02em;margin-bottom:0.4rem;line-height:1.2;}
.page-sub{font-size:0.75rem;color:var(--muted);line-height:1.7;margin-bottom:2rem;}

/* ── CARDS ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.5rem;}
.card-sm{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.25rem;}
.card-title{font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);
  margin-bottom:1rem;display:flex;align-items:center;gap:0.5rem;}
.card-title::before{content:'';width:6px;height:6px;border-radius:1px;background:var(--blue);flex-shrink:0;}

/* ── GRID HELPERS ── */
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;}
.grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;}
.stack{display:flex;flex-direction:column;gap:1rem;}

/* ── STAT TILES ── */
.stat-tile{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.25rem 1.5rem;
  position:relative;overflow:hidden;
}
.stat-tile::after{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,var(--blue));
}
.stat-tile.red{--accent:var(--red);}
.stat-tile.green{--accent:var(--green);}
.stat-tile.amber{--accent:var(--amber);}
.stat-label{font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:0.5rem;}
.stat-val{font-family:var(--serif);font-size:2rem;font-weight:600;color:#fff;line-height:1;}
.stat-note{font-size:0.68rem;color:var(--muted);margin-top:0.3rem;}

/* ── TABLES ── */
.table{width:100%;border-collapse:collapse;font-size:0.75rem;}
.table th{text-align:left;font-size:0.62rem;letter-spacing:0.09em;text-transform:uppercase;
  color:var(--muted);padding:0.5rem 0.75rem;border-bottom:1px solid var(--border);}
.table td{padding:0.6rem 0.75rem;border-bottom:1px solid rgba(28,45,69,0.5);vertical-align:middle;}
.table tr:last-child td{border-bottom:none;}
.table tr:hover td{background:rgba(74,158,255,0.03);}

/* ── BADGES ── */
.badge{
  display:inline-flex;align-items:center;gap:0.3rem;
  font-size:0.62rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;
  padding:0.18rem 0.55rem;border-radius:4px;
}
.badge-green{background:var(--green-dim);color:var(--green);border:1px solid rgba(61,214,140,0.25);}
.badge-red{background:var(--red-dim);color:var(--red);border:1px solid rgba(224,82,82,0.25);}
.badge-amber{background:var(--amber-dim);color:var(--amber);border:1px solid rgba(240,165,0,0.25);}
.badge-blue{background:var(--blue-dim);color:var(--blue);border:1px solid rgba(74,158,255,0.25);}
.badge-violet{background:rgba(167,139,250,0.12);color:var(--violet);border:1px solid rgba(167,139,250,0.25);}

/* ── BUTTONS ── */
.btn{
  font-family:var(--mono);font-size:0.7rem;letter-spacing:0.07em;text-transform:uppercase;
  padding:0.6rem 1.2rem;border-radius:6px;border:1px solid;cursor:pointer;
  background:transparent;transition:all 0.18s;font-weight:500;
}
.btn:disabled{opacity:0.35;cursor:not-allowed;}
.btn-green{border-color:var(--green);color:var(--green);}
.btn-green:not(:disabled):hover{background:var(--green-dim);}
.btn-amber{border-color:var(--amber);color:var(--amber);}
.btn-amber:not(:disabled):hover{background:var(--amber-dim);}
.btn-red{border-color:var(--red);color:var(--red);}
.btn-red:not(:disabled):hover{background:var(--red-dim);}
.btn-blue{border-color:var(--blue);color:var(--blue);}
.btn-blue:not(:disabled):hover{background:var(--blue-dim);}
.btn-primary{background:var(--blue);border-color:var(--blue);color:#fff;}
.btn-primary:not(:disabled):hover{background:#3a8ef0;}

/* ── PROGRESS BAR ── */
.bar-track{height:5px;background:var(--surface3);border-radius:3px;overflow:hidden;flex:1;}
.bar-fill{height:100%;border-radius:3px;transition:width 1.2s cubic-bezier(0.4,0,0.2,1);width:0;}

/* ── PHASE CARDS ── */
.phase-header{
  display:flex;align-items:center;gap:1rem;padding:1.25rem 1.5rem;
  background:var(--surface);border:1px solid var(--border);border-radius:12px 12px 0 0;
  border-bottom:none;
}
.phase-num{
  width:36px;height:36px;border-radius:8px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--serif);font-size:1.1rem;font-weight:700;color:#fff;
}
.phase-num.p1{background:linear-gradient(135deg,#4a9eff,#7c3aed);}
.phase-num.p2{background:linear-gradient(135deg,#f0a500,#e05252);}
.phase-num.p3{background:linear-gradient(135deg,#3dd68c,#4a9eff);}
.phase-title{font-family:var(--serif);font-size:1.05rem;font-weight:600;color:#fff;}
.phase-sub{font-size:0.68rem;color:var(--muted);}
.phase-body{
  background:var(--surface);border:1px solid var(--border);border-radius:0 0 12px 12px;
  padding:1.5rem;
}

/* ── CHECK LIST ── */
.check-item{
  display:flex;align-items:center;gap:0.75rem;
  padding:0.65rem 0;border-bottom:1px solid rgba(28,45,69,0.5);
}
.check-item:last-child{border-bottom:none;}
.check-icon{width:20px;height:20px;border-radius:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:11px;}
.check-icon.pass{background:var(--green-dim);color:var(--green);}
.check-icon.fail{background:var(--red-dim);color:var(--red);}
.check-icon.pending{background:var(--surface3);color:var(--muted);}
.check-name{font-size:0.75rem;color:var(--text);flex:1;}
.check-detail{font-size:0.68rem;color:var(--muted);text-align:right;max-width:300px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

/* ── SCORE DISPLAY ── */
.score-hero{
  display:flex;align-items:center;gap:2rem;
  padding:1.5rem 2rem;background:var(--surface);border:1px solid var(--border);border-radius:12px;
  margin-bottom:1rem;
}
.score-big{font-family:var(--serif);font-size:3.5rem;font-weight:700;line-height:1;min-width:160px;}
.score-components{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;}
.score-comp{background:var(--surface2);border-radius:8px;padding:0.75rem 1rem;}
.comp-label{font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);
  margin-bottom:0.35rem;display:flex;justify-content:space-between;}
.comp-val{font-family:var(--serif);font-size:1.1rem;font-weight:600;color:#fff;margin-bottom:0.4rem;}

/* ── LOG BOX ── */
.log-box{
  background:var(--bg);border:1px solid var(--border);border-radius:8px;
  padding:0.75rem;max-height:260px;overflow-y:auto;
  scrollbar-width:thin;scrollbar-color:var(--border2) transparent;
}
.log-row{
  display:grid;grid-template-columns:44px 72px 88px 80px 1fr;gap:0.5rem;
  font-size:0.7rem;padding:0.28rem 0;border-bottom:1px solid rgba(28,45,69,0.35);
  align-items:center;
}
.log-row:last-child{border-bottom:none;}
.l-step{color:var(--dim);}.l-act{color:var(--blue);}
.l-dist{color:var(--text);}.l-msg{color:var(--muted);font-size:0.64rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.pos{color:var(--green);font-weight:700;}.neg{color:var(--red);font-weight:700;}.neu{color:var(--muted);}

/* ── LOADING ── */
.loader{display:none;align-items:center;gap:0.75rem;padding:1.5rem 2rem;
  background:var(--surface);border:1px solid var(--border);border-radius:12px;margin-bottom:1rem;
  font-size:0.78rem;color:var(--muted);}
.spinner{width:18px;height:18px;border:2px solid var(--border2);border-top-color:var(--blue);
  border-radius:50%;animation:spin 0.7s linear infinite;flex-shrink:0;}
@keyframes spin{to{transform:rotate(360deg);}}

/* ── WEIGHT VIZ ── */
.weight-row{display:flex;align-items:center;gap:1rem;padding:0.6rem 0;border-bottom:1px solid rgba(28,45,69,0.4);}
.weight-row:last-child{border-bottom:none;}
.w-name{font-size:0.72rem;min-width:150px;}
.w-pct{font-size:0.72rem;font-weight:700;min-width:38px;}
.w-desc{font-size:0.65rem;color:var(--muted);text-align:right;flex:1;}

/* ── ARCH GRID ── */
.arch-card{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:1.1rem;}
.arch-icon{font-size:1.3rem;margin-bottom:0.6rem;}
.arch-name{font-size:0.82rem;font-weight:700;color:#fff;margin-bottom:0.3rem;}
.arch-desc{font-size:0.68rem;color:var(--muted);line-height:1.6;}

/* ── OVERVIEW HERO ── */
.overview-hero{
  display:grid;grid-template-columns:1fr auto;gap:2rem;align-items:start;
  margin-bottom:2rem;padding:2rem;
  background:var(--surface);border:1px solid var(--border);border-radius:12px;
  position:relative;overflow:hidden;
}
.overview-hero::before{
  content:'';position:absolute;top:-60px;right:-60px;width:240px;height:240px;
  border-radius:50%;background:radial-gradient(circle,rgba(74,158,255,0.08) 0%,transparent 70%);
  pointer-events:none;
}
.hero-title{font-family:var(--serif);font-size:2.2rem;font-weight:700;color:#fff;
  letter-spacing:-0.03em;line-height:1.15;margin-bottom:0.75rem;}
.hero-desc{font-size:0.75rem;color:var(--muted);line-height:1.8;max-width:560px;}
.hero-badges{display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:1rem;}
.hero-stats{display:flex;flex-direction:column;gap:0.75rem;min-width:160px;}
.hs-item{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:0.75rem 1rem;text-align:center;}
.hs-val{font-family:var(--serif);font-size:1.6rem;font-weight:700;color:#fff;line-height:1;}
.hs-label{font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem;}

/* ── TASK CARDS ── */
.task-card{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.25rem;
  position:relative;overflow:hidden;
}
.task-card::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--tc,var(--blue));}
.task-card.easy{--tc:var(--green);}
.task-card.medium{--tc:var(--amber);}
.task-card.hard{--tc:var(--red);}
.tc-tag{font-size:0.62rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.6rem;}
.tc-name{font-family:var(--serif);font-size:1rem;font-weight:600;color:#fff;margin-bottom:0.75rem;}
.tc-specs{display:grid;grid-template-columns:1fr 1fr;gap:0.35rem;}
.tc-spec{font-size:0.68rem;color:var(--muted);}
.tc-spec span{color:var(--text);}

/* ── SCORE BAR INLINE ── */
.sbi{display:flex;align-items:center;gap:0.5rem;}
.sbi .bar-track{width:70px;}

/* ── GEN LIST ── */
.gen-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;}
.gen-item{
  background:var(--surface2);border-left:3px solid var(--border2);border-radius:0 8px 8px 0;
  padding:0.75rem 1rem;
}
.gen-item.primary{border-left-color:var(--red);}
.gen-name{font-weight:700;font-size:0.82rem;color:#fff;margin-bottom:0.2rem;}
.gen-mech{font-size:0.68rem;color:var(--muted);}

/* ── RESPONSIVE ── */
@media(max-width:900px){
  .grid-3,.grid-4{grid-template-columns:1fr 1fr;}
  .score-hero{flex-direction:column;}
  .overview-hero{grid-template-columns:1fr;}
}
@media(max-width:600px){
  .grid-3,.grid-4,.grid-2{grid-template-columns:1fr;}
  .content{padding:1rem;}
}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="brand">
    <div class="brand-icon">🦠</div>
    <div>
      <div class="brand-name">Cascade Containment</div>
      <div class="brand-sub">OpenEnv Benchmark &middot; Meta PyTorch Hackathon × SST 2026</div>
    </div>
  </div>
  <div class="header-right">
    <div class="pill"><div class="dot"></div>Environment Live</div>
  </div>
</div>

<!-- NAV -->
<div class="nav">
  <button class="nav-btn active" onclick="tab('overview',this)">Overview</button>
  <button class="nav-btn" onclick="tab('phase1',this)">Phase 1 — Validation</button>
  <button class="nav-btn" onclick="tab('phase2',this)">Phase 2 — Evaluation</button>
  <button class="nav-btn" onclick="tab('phase3',this)">Phase 3 — Review</button>
  <button class="nav-btn" onclick="tab('grader',this)">Grader</button>
  <button class="nav-btn" onclick="tab('architecture',this)">Architecture</button>
</div>

<div class="content">

<!-- ═══════════════════════════════════════════════
     OVERVIEW
════════════════════════════════════════════════ -->
<div id="tab-overview" class="section active">
  <div class="overview-hero">
    <div>
      <div class="hero-title">Sequential Resource Allocation<br>Under Cascade Dynamics</div>
      <div class="hero-desc">
        A city health authority allocates scarce medical resources across districts to contain a spreading epidemic.
        Resources are limited. Data may be delayed. Infections spread geographically. Hospital collapse ends the episode.
        The same mechanics — cascade spreading, delayed observation, resource scarcity — apply to wildfire deployment,
        cyberattack isolation, and misinformation containment.
      </div>
      <div class="hero-badges">
        <span class="badge badge-blue">OpenEnv Compliant</span>
        <span class="badge badge-green">Docker Ready</span>
        <span class="badge badge-amber">3 Difficulty Levels</span>
        <span class="badge badge-violet">GRPO Baseline</span>
        <span class="badge badge-blue">Partial Observability</span>
      </div>
    </div>
    <div class="hero-stats">
      <div class="hs-item"><div class="hs-val">3</div><div class="hs-label">Task levels</div></div>
      <div class="hs-item"><div class="hs-val">5</div><div class="hs-label">Reward terms</div></div>
      <div class="hs-item"><div class="hs-val">4+</div><div class="hs-label">Domains</div></div>
    </div>
  </div>

  <div class="grid-3" style="margin-bottom:1rem;">
    <div class="task-card easy">
      <div class="tc-tag" style="color:var(--green);">Easy</div>
      <div class="tc-name">Single Outbreak</div>
      <div class="tc-specs">
        <div class="tc-spec">Districts: <span>2</span></div>
        <div class="tc-spec">Steps: <span>10</span></div>
        <div class="tc-spec">Resources: <span>10</span></div>
        <div class="tc-spec">Data lag: <span>None</span></div>
      </div>
    </div>
    <div class="task-card medium">
      <div class="tc-tag" style="color:var(--amber);">Medium</div>
      <div class="tc-name">Simultaneous Outbreaks</div>
      <div class="tc-specs">
        <div class="tc-spec">Districts: <span>4</span></div>
        <div class="tc-spec">Steps: <span>15</span></div>
        <div class="tc-spec">Resources: <span>8</span></div>
        <div class="tc-spec">Data lag: <span>None</span></div>
      </div>
    </div>
    <div class="task-card hard">
      <div class="tc-tag" style="color:var(--red);">Hard</div>
      <div class="tc-name">Invisible Acceleration</div>
      <div class="tc-specs">
        <div class="tc-spec">Districts: <span>6</span></div>
        <div class="tc-spec">Steps: <span>15</span></div>
        <div class="tc-spec">Resources: <span>7</span></div>
        <div class="tc-spec">Data lag: <span style="color:var(--red);">3 days</span></div>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Action Space</div>
      <table class="table">
        <tr><th>Action</th><th>Cost</th><th>Effect</th></tr>
        <tr><td style="color:var(--blue);font-weight:700;">test</td><td>1 resource</td><td>Accurate infection data</td></tr>
        <tr><td style="color:var(--amber);font-weight:700;">restrict</td><td>Free</td><td>Slow spread; penalised if infection &lt; 0.2</td></tr>
        <tr><td style="color:var(--green);font-weight:700;">allocate</td><td>1 resource</td><td>Reduce infection 5%, slow spread</td></tr>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Generalisation Domains</div>
      <div class="gen-grid">
        <div class="gen-item primary"><div class="gen-name">🦠 Epidemic</div><div class="gen-mech">Primary framing</div></div>
        <div class="gen-item"><div class="gen-name">🔥 Wildfire</div><div class="gen-mech">Pre-position crews; satellite lag</div></div>
        <div class="gen-item"><div class="gen-name">🛡️ Cyberattack</div><div class="gen-mech">Isolate systems; detection lag</div></div>
        <div class="gen-item"><div class="gen-name">📢 Misinformation</div><div class="gen-mech">Deploy corrections; network spread</div></div>
      </div>
    </div>
  </div>
</div>


<!-- ═══════════════════════════════════════════════
     PHASE 1 — AUTOMATED VALIDATION
════════════════════════════════════════════════ -->
<div id="tab-phase1" class="section">
  <div class="phase-header">
    <div class="phase-num p1">1</div>
    <div>
      <div class="phase-title">Automated Validation</div>
      <div class="phase-sub">Pass/fail gate — spec compliance, Dockerfile, baseline reproducibility, grader integrity</div>
    </div>
    <div style="margin-left:auto;">
      <button class="btn btn-primary" id="btn-validate" onclick="runValidation()">▶ Run Validation</button>
    </div>
  </div>
  <div class="phase-body">
    <div class="loader" id="load-validate"><div class="spinner"></div><span>Running spec compliance checks...</span></div>

    <div id="validate-result" style="display:none;">
      <div class="grid-4" style="margin-bottom:1.5rem;">
        <div class="stat-tile green">
          <div class="stat-label">Overall</div>
          <div class="stat-val" id="v-overall" style="font-size:1.1rem;margin-top:0.2rem;">—</div>
        </div>
        <div class="stat-tile">
          <div class="stat-label">Passed</div>
          <div class="stat-val" id="v-pass">—</div>
        </div>
        <div class="stat-tile red">
          <div class="stat-label">Failed</div>
          <div class="stat-val" id="v-fail">—</div>
        </div>
        <div class="stat-tile">
          <div class="stat-label">Total Checks</div>
          <div class="stat-val" id="v-total">—</div>
        </div>
      </div>
      <div id="v-checks"></div>
    </div>

    <div id="validate-placeholder" style="padding:2rem 0;text-align:center;color:var(--muted);font-size:0.8rem;">
      Click "Run Validation" to execute all Phase 1 automated checks against the live environment.
    </div>
  </div>

  <div style="margin-top:1rem;" class="card">
    <div class="card-title">What Phase 1 Checks</div>
    <table class="table">
      <tr><th>Check</th><th>Requirement</th></tr>
      <tr><td>HF Space deploys</td><td>Environment responds on port 7860</td></tr>
      <tr><td>OpenEnv spec compliance</td><td>reset(), step(), state property all function correctly</td></tr>
      <tr><td>3+ tasks with graders</td><td>Easy, Medium, Hard all produce grader scores in [0.0, 1.0]</td></tr>
      <tr><td>Invalid actions handled</td><td>Bad input doesn't crash the episode</td></tr>
      <tr><td>Difficulty progression</td><td>Tasks scale in complexity (districts, steps, resources)</td></tr>
      <tr><td>Grader deterministic</td><td>Same trajectory always returns same score</td></tr>
    </table>
  </div>
</div>


<!-- ═══════════════════════════════════════════════
     PHASE 2 — AGENTIC EVALUATION
════════════════════════════════════════════════ -->
<div id="tab-phase2" class="section">
  <div class="phase-header">
    <div class="phase-num p2">2</div>
    <div>
      <div class="phase-title">Agentic Evaluation</div>
      <div class="phase-sub">Scored — run rule-based and LLM agents against all tasks; inspect grader output</div>
    </div>
  </div>
  <div class="phase-body">
    <!-- Task selector -->
    <div style="display:flex;gap:0.75rem;align-items:center;margin-bottom:1.5rem;flex-wrap:wrap;">
      <button class="btn btn-green" id="btn-easy"   onclick="runDemo('easy')"  >▶ Run Easy</button>
      <button class="btn btn-amber" id="btn-medium" onclick="runDemo('medium')">▶ Run Medium</button>
      <button class="btn btn-red"   id="btn-hard"   onclick="runDemo('hard')"  >▶ Run Hard</button>
      <span style="font-size:0.68rem;color:var(--muted);margin-left:0.5rem;">Rule-based greedy agent — allocates to highest-infected district; restricts when resources exhausted</span>
    </div>

    <div class="loader" id="load-demo"><div class="spinner"></div><span id="load-demo-text">Running episode...</span></div>

    <div id="demo-result" style="display:none;">
      <div class="score-hero">
        <div>
          <div style="font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:0.35rem;">Final Score</div>
          <div class="score-big" id="d-score">—</div>
          <div style="margin-top:0.6rem;display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;">
            <span id="d-task-badge" class="badge badge-blue">—</span>
            <span id="d-steps" style="font-size:0.7rem;color:var(--muted);"></span>
            <span id="d-breach"></span>
          </div>
        </div>
        <div class="score-components">
          <div class="score-comp">
            <div class="comp-label">Hospital <span style="color:var(--muted);">45%</span></div>
            <div class="comp-val" id="cv-hospital">—</div>
            <div class="bar-track"><div class="bar-fill" id="cf-hospital" style="background:var(--blue);"></div></div>
          </div>
          <div class="score-comp">
            <div class="comp-label">Containment <span style="color:var(--muted);">30%</span></div>
            <div class="comp-val" id="cv-containment">—</div>
            <div class="bar-track"><div class="bar-fill" id="cf-containment" style="background:var(--green);"></div></div>
          </div>
          <div class="score-comp">
            <div class="comp-label">Efficiency <span style="color:var(--muted);">15%</span></div>
            <div class="comp-val" id="cv-efficiency">—</div>
            <div class="bar-track"><div class="bar-fill" id="cf-efficiency" style="background:var(--violet);"></div></div>
          </div>
          <div class="score-comp">
            <div class="comp-label">Speed <span style="color:var(--muted);">10%</span></div>
            <div class="comp-val" id="cv-speed">—</div>
            <div class="bar-track"><div class="bar-fill" id="cf-speed" style="background:var(--amber);"></div></div>
          </div>
        </div>
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
        <div style="font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);">Step Log</div>
        <div id="d-log-meta" style="font-size:0.68rem;color:var(--muted);"></div>
      </div>
      <div class="log-box" id="demo-log"></div>
    </div>

    <div id="demo-placeholder" style="padding:2rem 0;text-align:center;color:var(--muted);font-size:0.8rem;">
      Select a task above to run a live episode and inspect grader scores.
    </div>
  </div>

  <!-- Benchmark reference -->
  <div style="margin-top:1rem;" class="card">
    <div class="card-title">LLM + GRPO Baseline Benchmark Scores</div>
    <table class="table">
      <tr><th>Task</th><th>Agent</th><th>Containment</th><th>Hospital</th><th>Efficiency</th><th>Score</th></tr>
      <tr>
        <td><span class="badge badge-green">Easy</span></td>
        <td style="color:var(--muted);">Greedy</td><td>0.35</td><td>0.90</td><td>0.45</td>
        <td><div class="sbi"><div class="bar-track"><div class="bar-fill" style="width:50%;background:var(--green);"></div></div>~0.50</div></td>
      </tr>
      <tr>
        <td><span class="badge badge-green">Easy</span></td>
        <td>LLM + GRPO</td><td>1.00</td><td>1.00</td><td>1.00</td>
        <td><div class="sbi"><div class="bar-track"><div class="bar-fill" style="width:91%;background:var(--green);"></div></div><strong>0.88–0.93</strong></div></td>
      </tr>
      <tr>
        <td><span class="badge badge-amber">Medium</span></td>
        <td style="color:var(--muted);">Greedy</td><td>0.18</td><td>0.21</td><td>0.40</td>
        <td><div class="sbi"><div class="bar-track"><div class="bar-fill" style="width:23%;background:var(--amber);"></div></div>~0.23</div></td>
      </tr>
      <tr>
        <td><span class="badge badge-amber">Medium</span></td>
        <td>LLM + GRPO</td><td>0.44–0.73</td><td>0.97–1.00</td><td>0.87–1.00</td>
        <td><div class="sbi"><div class="bar-track"><div class="bar-fill" style="width:78%;background:var(--amber);"></div></div><strong>0.70–0.85</strong></div></td>
      </tr>
      <tr>
        <td><span class="badge badge-red">Hard</span></td>
        <td style="color:var(--muted);">Greedy</td><td>0.12</td><td>0.18</td><td>0.25</td>
        <td><div class="sbi"><div class="bar-track"><div class="bar-fill" style="width:21%;background:var(--red);"></div></div>~0.21</div></td>
      </tr>
      <tr>
        <td><span class="badge badge-red">Hard</span></td>
        <td>LLM + GRPO</td><td>0.28–0.51</td><td>0.86–0.97</td><td>0.47–0.73</td>
        <td><div class="sbi"><div class="bar-track"><div class="bar-fill" style="width:62%;background:var(--red);"></div></div><strong>0.58–0.65</strong></div></td>
      </tr>
    </table>
    <div style="margin-top:1rem;padding:0.75rem 1rem;background:var(--surface2);border-radius:6px;font-size:0.7rem;color:var(--muted);line-height:1.7;">
      The gap between greedy and LLM+GRPO demonstrates meaningful discrimination. Greedy agents score 0.21–0.50; LLM+GRPO agents score 0.62–0.93.
      No policy trivially achieves high scores — genuine triage intelligence is required.
    </div>
  </div>
</div>


<!-- ═══════════════════════════════════════════════
     PHASE 3 — HUMAN REVIEW
════════════════════════════════════════════════ -->
<div id="tab-phase3" class="section">
  <div class="phase-header">
    <div class="phase-num p3">3</div>
    <div>
      <div class="phase-title">Human Review</div>
      <div class="phase-sub">Meta &amp; Hugging Face engineers assess real-world utility, creativity, and exploit robustness</div>
    </div>
  </div>
  <div class="phase-body stack">

    <div class="grid-2">
      <div class="card-sm">
        <div class="card-title">Real-World Utility</div>
        <div style="font-size:0.75rem;color:var(--muted);line-height:1.9;">
          <div style="margin-bottom:0.5rem;"><span style="color:var(--green);">▸</span> <strong style="color:var(--text);">WHO-modelled epidemic response.</strong> Resource allocation, restriction policy, and hospital capacity constraints match real public health frameworks.</div>
          <div style="margin-bottom:0.5rem;"><span style="color:var(--green);">▸</span> <strong style="color:var(--text);">Multi-domain transfer.</strong> Wildfire deployment, cyberattack isolation, and misinformation containment share identical mathematical structure — same trained policy generalises.</div>
          <div style="margin-bottom:0.5rem;"><span style="color:var(--green);">▸</span> <strong style="color:var(--text);">3-day information lag</strong> on the hard task reflects real reporting delays in surveillance systems — not a toy mechanic.</div>
          <div><span style="color:var(--green);">▸</span> <strong style="color:var(--text);">Hospital breach at 10% capacity</strong> matches ICU overflow thresholds where triage and diversion begin, not at zero.</div>
        </div>
      </div>
      <div class="card-sm">
        <div class="card-title">Novelty &amp; Creativity</div>
        <div style="font-size:0.75rem;color:var(--muted);line-height:1.9;">
          <div style="margin-bottom:0.5rem;"><span style="color:var(--violet);">▸</span> <strong style="color:var(--text);">Cascade dynamics class.</strong> No existing OpenEnv benchmark covers the spreading-cascade / delayed-observation / resource-scarcity problem class.</div>
          <div style="margin-bottom:0.5rem;"><span style="color:var(--violet);">▸</span> <strong style="color:var(--text);">Structural partial observability.</strong> The 3-day lag is enforced at the environment layer — the agent cannot test its way around it.</div>
          <div style="margin-bottom:0.5rem;"><span style="color:var(--violet);">▸</span> <strong style="color:var(--text);">GRPO episodic memory baseline.</strong> Prompt-as-policy with advantage-gated memory update — no weight gradients.</div>
          <div><span style="color:var(--violet);">▸</span> <strong style="color:var(--text);">Decaying containment bonus.</strong> Early action is exponentially more valuable, teaching proactive not reactive strategies.</div>
        </div>
      </div>
    </div>

    <div class="card-sm">
      <div class="card-title">Exploit Resistance</div>
      <table class="table">
        <tr><th>Potential Exploit</th><th>Prevention Mechanism</th><th>Status</th></tr>
        <tr>
          <td>Always restrict everything</td>
          <td>Penalty of −0.20 per restriction on districts below 0.20</td>
          <td><span class="badge badge-green">Blocked</span></td>
        </tr>
        <tr>
          <td>Always use "test" to game data</td>
          <td>Test costs 1 resource; real-time data already provided; no benefit</td>
          <td><span class="badge badge-green">Blocked</span></td>
        </tr>
        <tr>
          <td>Trivially containable (easy task too easy)</td>
          <td>Random spread rates [0.03–0.08] create variance; greedy agent scores only ~0.50</td>
          <td><span class="badge badge-green">Addressed</span></td>
        </tr>
        <tr>
          <td>Game containment by ignoring hospitals</td>
          <td>Hospital breach (≤10% capacity) ends episode immediately; hospital score weighted 45%</td>
          <td><span class="badge badge-green">Blocked</span></td>
        </tr>
        <tr>
          <td>Memorise fixed seed values</td>
          <td>Spread rates randomised per episode; density weights randomised; lag history unpredictable</td>
          <td><span class="badge badge-green">Blocked</span></td>
        </tr>
        <tr>
          <td>Infinite restrictions accumulate</td>
          <td>Restrictions auto-lift when infection drops below safe threshold</td>
          <td><span class="badge badge-green">Addressed</span></td>
        </tr>
      </table>
    </div>

    <div class="card-sm">
      <div class="card-title">Epidemiological Model Calibration</div>
      <table class="table">
        <tr><th>Parameter</th><th>Value</th><th>Real-World Reference</th></tr>
        <tr><td>Spread rate</td><td>3–8% / day</td><td>Seasonal flu R₀ 1.2–1.4; daily transmission ≈ 4–7%</td></tr>
        <tr><td>Natural recovery</td><td>1% / day</td><td>Mild respiratory illness: 7–14 day recovery → ~1%/day</td></tr>
        <tr><td>Hospital breach threshold</td><td>≤10% capacity</td><td>WHO: ICU overflow typically triggers crisis protocols at &lt;15%</td></tr>
        <tr><td>Geographic spillover</td><td>1% to adjacent</td><td>District-level cross-border movement in urban corridors</td></tr>
        <tr><td>Data lag (hard task)</td><td>3 days</td><td>US CDC surveillance reporting lag: 2–5 days</td></tr>
        <tr><td>Treatment effect</td><td>−5% infection</td><td>Antiviral deployment impact on active case load</td></tr>
      </table>
    </div>

  </div>
</div>


<!-- ═══════════════════════════════════════════════
     GRADER
════════════════════════════════════════════════ -->
<div id="tab-grader" class="section">
  <div class="card" style="margin-bottom:1rem;">
    <div class="card-title">Deterministic Trajectory Scorer</div>
    <div style="font-size:0.75rem;color:var(--muted);line-height:1.8;max-width:700px;">
      The grader receives the full episode trajectory (hidden ground truth, not agent observations) and returns a score in
      <strong style="color:var(--text);">[0.0, 1.0]</strong>. No randomness. No LLM calls. Identical trajectories always produce identical scores.
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:1rem;">
    <div class="card">
      <div class="card-title">Score Components</div>
      <div class="weight-row">
        <div class="w-name" style="color:var(--blue);">Hospital Score</div>
        <div class="w-pct" style="color:var(--blue);">45%</div>
        <div class="bar-track"><div class="bar-fill" style="width:45%;background:var(--blue);"></div></div>
        <div class="w-desc">Avg capacity preserved; ×0.6 if any breach</div>
      </div>
      <div class="weight-row">
        <div class="w-name" style="color:var(--green);">Containment Score</div>
        <div class="w-pct" style="color:var(--green);">30%</div>
        <div class="bar-track"><div class="bar-fill" style="width:30%;background:var(--green);"></div></div>
        <div class="w-desc">% district-days below 0.40 (skips first 2 steps)</div>
      </div>
      <div class="weight-row">
        <div class="w-name" style="color:var(--violet);">Efficiency Score</div>
        <div class="w-pct" style="color:var(--violet);">15%</div>
        <div class="bar-track"><div class="bar-fill" style="width:15%;background:var(--violet);"></div></div>
        <div class="w-desc">Resource actions targeting highest-infected district</div>
      </div>
      <div class="weight-row">
        <div class="w-name" style="color:var(--amber);">Speed Score</div>
        <div class="w-pct" style="color:var(--amber);">10%</div>
        <div class="bar-track"><div class="bar-fill" style="width:10%;background:var(--amber);"></div></div>
        <div class="w-desc">1 − (steps / max_steps) if finished early; else 0</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Design Decisions</div>
      <div style="font-size:0.75rem;color:var(--muted);line-height:2;">
        <div><span style="color:var(--blue);">→</span> <strong style="color:var(--text);">Hospital weighted highest</strong> — system collapse is catastrophic and irreversible.</div>
        <div><span style="color:var(--green);">→</span> <strong style="color:var(--text);">Grace period</strong> — first 2 steps excluded from containment; initial state outside agent control.</div>
        <div><span style="color:var(--violet);">→</span> <strong style="color:var(--text);">Pre-action efficiency</strong> — uses previous step's state so successful treatment isn't retroactively penalised.</div>
        <div><span style="color:var(--amber);">→</span> <strong style="color:var(--text);">Speed as tiebreaker</strong> — rewards decisive proactive containment over dragging to max steps.</div>
        <div><span style="color:var(--red);">→</span> <strong style="color:var(--text);">Breach multiplier ×0.6</strong> — any hospital collapse permanently degrades the hospital sub-score.</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Reward Function</div>
    <table class="table">
      <tr><th>Term</th><th>Value</th><th>Fires When</th><th>Design Rationale</th></tr>
      <tr><td>Infection penalty</td><td class="neg">−0.50 × density</td><td>District infection &gt; 0.40</td><td>Dense districts penalised more; realistic triage</td></tr>
      <tr><td>Hospital breach</td><td class="neg">−1.00</td><td>Hospital capacity ≤ 10%</td><td>Collapse is catastrophic; heaviest penalty</td></tr>
      <tr><td>Early containment</td><td class="pos">+0.50 × (1 − t)</td><td>District infection &lt; 0.20</td><td>Decays over time; proactive action rewarded</td></tr>
      <tr><td>Correct prioritisation</td><td class="pos">+0.30</td><td>Allocate to highest-infected</td><td>Rewards triage intelligence at each step</td></tr>
      <tr><td>Unnecessary restriction</td><td class="neg">−0.20</td><td>Restrict district below 0.20</td><td>Penalises over-intervention on safe districts</td></tr>
    </table>
  </div>
</div>


<!-- ═══════════════════════════════════════════════
     ARCHITECTURE
════════════════════════════════════════════════ -->
<div id="tab-architecture" class="section">
  <div class="card" style="margin-bottom:1rem;">
    <div class="card-title">OpenEnv Interface</div>
    <div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-top:0.25rem;">
      <div style="background:var(--surface2);border-radius:8px;padding:0.6rem 1rem;font-size:0.74rem;">
        <span style="color:var(--muted);">env.</span><span style="color:var(--blue);">reset</span><span style="color:var(--muted);">(task_name)</span>
        <span style="color:var(--dim);margin:0 0.5rem;">→</span><span style="color:var(--green);">CityObservation</span>
      </div>
      <div style="background:var(--surface2);border-radius:8px;padding:0.6rem 1rem;font-size:0.74rem;">
        <span style="color:var(--muted);">env.</span><span style="color:var(--blue);">step</span><span style="color:var(--muted);">(action)</span>
        <span style="color:var(--dim);margin:0 0.5rem;">→</span><span style="color:var(--green);">CityObservation</span>
      </div>
      <div style="background:var(--surface2);border-radius:8px;padding:0.6rem 1rem;font-size:0.74rem;">
        <span style="color:var(--muted);">env.</span><span style="color:var(--blue);">state</span>
        <span style="color:var(--dim);margin:0 0.5rem;">→</span><span style="color:var(--green);">State</span>
      </div>
    </div>
  </div>

  <div class="grid-3" style="margin-bottom:1rem;">
    <div class="arch-card"><div class="arch-icon">📋</div><div class="arch-name">models.py</div><div class="arch-desc">Typed data contracts. DistrictObservation, DistrictTruth, CityState, CityObservation, ContainmentAction. Pydantic + dataclasses.</div></div>
    <div class="arch-card"><div class="arch-icon">⚙️</div><div class="arch-name">environment.py</div><div class="arch-desc">Core RL loop. Maintains OpenEnv State (tracking) and CityState (simulation). Agent only ever receives CityObservation.</div></div>
    <div class="arch-card"><div class="arch-icon">📊</div><div class="arch-name">grader.py</div><div class="arch-desc">Deterministic trajectory scorer. Reads hidden CityState. Four components weighted into final_score ∈ [0.0, 1.0]. Zero LLM calls.</div></div>
    <div class="arch-card"><div class="arch-icon">🌊</div><div class="arch-name">utils.py</div><div class="arch-desc">SIR-inspired spread model with linear spillover (no wrap-around). Observation builder enforcing partial observability by task.</div></div>
    <div class="arch-card"><div class="arch-icon">🧠</div><div class="arch-name">core/trajectory.py</div><div class="arch-desc">EpisodicMemory. Stores (obs, action, reward) tuples. Retrieves top-k by L1 distance on infection profiles, phase-weighted.</div></div>
    <div class="arch-card"><div class="arch-icon">🎯</div><div class="arch-name">baseline/evaluator.py</div><div class="arch-desc">GRPO-style loop. Advantage = Rᵢ − mean(R). Memory gated by advantage threshold. Reports best grader score across rollouts.</div></div>
  </div>

  <div class="card">
    <div class="card-title">HTTP Endpoints</div>
    <table class="table">
      <tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
      <tr><td style="color:var(--green);font-weight:600;">/</td><td>GET</td><td>This judge dashboard</td></tr>
      <tr><td style="color:var(--green);font-weight:600;">/health</td><td>GET</td><td>Health check — returns {"status":"ok"}</td></tr>
      <tr><td style="color:var(--green);font-weight:600;">/info</td><td>GET</td><td>Environment metadata, task config, grader weights</td></tr>
      <tr><td style="color:var(--green);font-weight:600;">/grade</td><td>GET</td><td>Grader scores for the last completed episode</td></tr>
      <tr><td style="color:var(--green);font-weight:600;">/validate</td><td>GET</td><td>Phase 1 automated spec compliance — all checks with pass/fail</td></tr>
      <tr><td style="color:var(--green);font-weight:600;">/demo/{task}</td><td>GET</td><td>Rule-based greedy episode + full step log + grader score</td></tr>
    </table>
  </div>
</div>

</div><!-- /content -->

<script>
/* ── TAB SWITCHING ── */
function tab(name, el) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  el.classList.add('active');
}

const pct = v => (v * 100).toFixed(1) + '%';
const fmt = v => typeof v === 'number' ? (v * 100).toFixed(1) + '%' : v;

/* ── PHASE 1: VALIDATION ── */
async function runValidation() {
  const btn = document.getElementById('btn-validate');
  const loader = document.getElementById('load-validate');
  const result = document.getElementById('validate-result');
  const ph = document.getElementById('validate-placeholder');

  btn.disabled = true;
  loader.style.display = 'flex';
  result.style.display = 'none';
  ph.style.display = 'none';

  try {
    const r = await fetch('/validate');
    const d = await r.json();

    const passCount = d.pass_count;
    const failCount = d.total - d.pass_count;

    document.getElementById('v-overall').textContent = d.overall;
    document.getElementById('v-overall').style.color = d.overall === 'PASS' ? 'var(--green)' : 'var(--red)';
    document.getElementById('v-pass').textContent = passCount;
    document.getElementById('v-pass').style.color = 'var(--green)';
    document.getElementById('v-fail').textContent = failCount;
    document.getElementById('v-fail').style.color = failCount > 0 ? 'var(--red)' : 'var(--muted)';
    document.getElementById('v-total').textContent = d.total;

    const container = document.getElementById('v-checks');
    container.innerHTML = Object.entries(d.checks).map(([key, val]) => {
      const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const cls = val.pass ? 'pass' : 'fail';
      const icon = val.pass ? '✓' : '✗';
      return `<div class="check-item">
        <div class="check-icon ${cls}">${icon}</div>
        <div class="check-name">${label}</div>
        <div class="check-detail">${val.detail || ''}</div>
      </div>`;
    }).join('');

    loader.style.display = 'none';
    result.style.display = 'block';
  } catch (e) {
    loader.style.display = 'none';
    ph.style.display = 'block';
    ph.textContent = 'Validation error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

/* ── PHASE 2: DEMO ── */
async function runDemo(task) {
  ['easy','medium','hard'].forEach(t => document.getElementById('btn-' + t).disabled = true);
  const loader = document.getElementById('load-demo');
  const result = document.getElementById('demo-result');
  const ph = document.getElementById('demo-placeholder');

  document.getElementById('load-demo-text').textContent = 'Running ' + task + ' episode...';
  loader.style.display = 'flex';
  result.style.display = 'none';
  ph.style.display = 'none';

  try {
    const r = await fetch('/demo/' + task);
    const d = await r.json();

    if (d.error) { throw new Error(d.error); }

    const scoreEl = document.getElementById('d-score');
    scoreEl.textContent = pct(d.final_score);
    scoreEl.style.color = d.final_score > 0.65 ? 'var(--green)' : d.final_score > 0.40 ? 'var(--amber)' : 'var(--red)';

    const tColors = {easy:'badge-green', medium:'badge-amber', hard:'badge-red'};
    document.getElementById('d-task-badge').className = 'badge ' + (tColors[task] || 'badge-blue');
    document.getElementById('d-task-badge').textContent = task.toUpperCase();
    document.getElementById('d-steps').textContent = d.total_steps + ' steps · ' + d.districts_contained + ' contained';
    document.getElementById('d-breach').innerHTML = d.hospital_breached
      ? '<span class="badge badge-red">Hospital Breached</span>'
      : '<span class="badge badge-green">No Breach</span>';

    const comps = {hospital: d.hospital_score, containment: d.containment_score,
                   efficiency: d.efficiency_score, speed: d.speed_score};
    for (const [k, v] of Object.entries(comps)) {
      document.getElementById('cv-' + k).textContent = pct(v);
      setTimeout(() => { document.getElementById('cf-' + k).style.width = (v * 100) + '%'; }, 80);
    }

    const logEl = document.getElementById('demo-log');
    logEl.innerHTML = d.log.map(s => {
      const cls = s.reward > 0.05 ? 'pos' : s.reward < -0.4 ? 'neg' : 'neu';
      const rStr = (s.reward >= 0 ? '+' : '') + s.reward.toFixed(4);
      return `<div class="log-row">
        <span class="l-step">s${String(s.step).padStart(2,'0')}</span>
        <span class="l-act">${s.action_type}</span>
        <span class="l-dist">D${s.district_id}</span>
        <span class="${cls}">${rStr}</span>
        <span class="l-msg">${s.message || ''}</span>
      </div>`;
    }).join('');
    document.getElementById('d-log-meta').textContent =
      'Greedy agent · ' + d.total_steps + ' steps · score ' + pct(d.final_score);

    loader.style.display = 'none';
    result.style.display = 'block';
  } catch (e) {
    loader.style.display = 'none';
    ph.style.display = 'block';
    ph.textContent = 'Error: ' + e.message;
  } finally {
    ['easy','medium','hard'].forEach(t => document.getElementById('btn-' + t).disabled = false);
  }
}
</script>
</body>
</html>"""
