# server/tasks/task_medium.py
# ─────────────────────────────────────────────────────────────────────────────
# Medium task: 4 districts, 2 simultaneous outbreaks, tighter resource pool.
# Agent must prioritise between competing threats — it cannot fully
# address both outbreaks simultaneously and must learn to triage.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))    # reaches server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')) # reaches project root

from models import CityState
from server.utils import generate_districts
from server.tasks.base import BaseTask
from server.constants import TASK_CONFIG


class MediumTask(BaseTask):

    name          = "medium"
    num_districts = TASK_CONFIG["medium"]["num_districts"]  # 4
    max_steps     = TASK_CONFIG["medium"]["max_steps"]      # 15
    resource_pool = TASK_CONFIG["medium"]["resource_pool"]  # 8
    data_lag_days = TASK_CONFIG["medium"]["data_lag_days"]  # 0

    def build_initial_state(self) -> CityState:
        # Districts 0 and 2 are seeded with outbreaks (non-adjacent).
        # Districts 1 and 3 are clean but will receive spillover.
        # Agent must choose which outbreak to tackle first.
        seed_infections = [0.25, 0.06, 0.22, 0.06]

        return CityState(
            day                 = 0,
            available_resources = self.resource_pool,
            task_name           = self.name,
            data_lag_days       = self.data_lag_days,
            max_steps           = self.max_steps,
            districts           = generate_districts(
                                      num_districts    = self.num_districts,
                                      seed_infections  = seed_infections,
                                  ),
            infection_history   = [],
        )