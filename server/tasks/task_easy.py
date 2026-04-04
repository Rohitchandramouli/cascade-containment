# server/tasks/task_easy.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models import CityState
from server.utils import generate_districts
from server.tasks.base import BaseTask
from server.constants import TASK_CONFIG


class EasyTask(BaseTask):

    name          = "easy"
    num_districts = TASK_CONFIG["easy"]["num_districts"]
    max_steps     = TASK_CONFIG["easy"]["max_steps"]
    resource_pool = TASK_CONFIG["easy"]["resource_pool"]
    data_lag_days = TASK_CONFIG["easy"]["data_lag_days"]

    def build_initial_state(self) -> CityState:
        # D0 has a visible outbreak in WARNING-CRITICAL boundary.
        # Seed is high enough that one allocation cannot instantly contain it —
        # agent must sustain 3+ allocations while also managing D1's growth.
        seed_infections = [0.45, 0.06]

        return CityState(
            day                 = 0,
            available_resources = self.resource_pool,
            task_name           = self.name,
            data_lag_days       = self.data_lag_days,
            max_steps           = self.max_steps,
            districts           = generate_districts(
                                      num_districts   = self.num_districts,
                                      seed_infections = seed_infections,
                                  ),
            infection_history   = [],
        )