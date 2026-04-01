# server/tasks/task_easy.py
# ─────────────────────────────────────────────────────────────────────────────
# Easy task: 2 districts, 1 outbreak, accurate real-time data.
# Agent should learn to test the infected district, restrict it,
# and allocate resources. A straightforward strategy scores 0.7–0.9.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))    # reaches server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')) # reaches project root

from models import CityState
from server.utils import generate_districts
from server.tasks.base import BaseTask
from server.constants import TASK_CONFIG


class EasyTask(BaseTask):

    name          = "easy"
    num_districts = TASK_CONFIG["easy"]["num_districts"]    # 2
    max_steps     = TASK_CONFIG["easy"]["max_steps"]        # 10
    resource_pool = TASK_CONFIG["easy"]["resource_pool"]    # 10
    data_lag_days = TASK_CONFIG["easy"]["data_lag_days"]    # 0

    def build_initial_state(self) -> CityState:
        # District 0 has a visible outbreak. District 1 is clean.
        # Agent only needs to identify and respond to one threat.
        seed_infections = [0.25, 0.05]

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