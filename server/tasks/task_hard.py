# server/tasks/task_hard.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models import CityState
from server.utils import generate_districts
from server.tasks.base import BaseTask
from server.constants import TASK_CONFIG


class HardTask(BaseTask):

    name          = "hard"
    num_districts = TASK_CONFIG["hard"]["num_districts"]
    max_steps     = TASK_CONFIG["hard"]["max_steps"]
    resource_pool = TASK_CONFIG["hard"]["resource_pool"]
    data_lag_days = TASK_CONFIG["hard"]["data_lag_days"]

    def build_initial_state(self) -> CityState:
        # All districts start with small but growing infections.
        # The 3-day lag means agent sees these seed values while true
        # infection is already 3 days ahead — districts D0, D2, D4 will
        # be CRITICAL before the agent sees updated data.
        # 7 resources for 6 districts with delayed information is
        # the hardest possible triage scenario.
        seed_infections = [0.10, 0.06, 0.12, 0.05, 0.13, 0.07]

        initial_rates     = seed_infections[:]
        infection_history = [
            initial_rates[:],  # day -3 (what agent sees on step 1)
            initial_rates[:],  # day -2
            initial_rates[:],  # day -1
        ]

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
            infection_history   = infection_history,
        )