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