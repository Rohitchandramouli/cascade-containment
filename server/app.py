# server/app.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application entry point for Cascade Containment.
# Uses a factory function so each WebSocket session gets its own isolated
# environment instance — required for concurrent session safety.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openenv.core.env_server import create_app
from server.environment import EpidemicContainmentEnv
from models import ContainmentAction, CityObservation


app = create_app(
    EpidemicContainmentEnv,
    ContainmentAction,
    CityObservation,
)

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <body style="font-family: sans-serif; padding: 2rem; background: #0f172a; color: #e2e8f0;">
        <h1>🦠 Cascade Containment</h1>
        <p>An RL benchmark for epidemic containment policy under uncertainty.</p>
        <p>Status: <strong style="color: #4ade80;">Running ✓</strong></p>
        <h3>Available Endpoints</h3>
        <ul>
            <li><a href="/health" style="color: #60a5fa;">/health</a> — Health check</li>
            <li>/reset — Start new episode</li>
            <li>/step — Take action</li>
            <li>/state — Get episode state</li>
        </ul>
        <p style="color: #94a3b8; margin-top: 2rem;">
            Tasks: easy (2 districts) · medium (4 districts) · hard (6 districts, 3-day data lag)
        </p>
    </body>
    </html>
    """