# server/tasks/task_medium.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models import CityState
from server.utils import generate_districts
from server.tasks.base import BaseTask
from server.constants import TASK_CONFIG


class MediumTask(BaseTask):

    name          = "medium"
    num_districts = TASK_CONFIG["medium"]["num_districts"]
    max_steps     = TASK_CONFIG["medium"]["max_steps"]
    resource_pool = TASK_CONFIG["medium"]["resource_pool"]
    data_lag_days = TASK_CONFIG["medium"]["data_lag_days"]

    def build_initial_state(self) -> CityState:
        # D0 and D2 seeded with outbreaks (non-adjacent).
        # D1 and D3 start clean but will grow into critical within 3-4 steps.
        # 8 total resources for 4 districts creates genuine triage pressure —
        # agent cannot save all districts and must choose strategically.
        seed_infections = [0.18, 0.03, 0.15, 0.03]

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