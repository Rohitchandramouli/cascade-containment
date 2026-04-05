# server/tasks/task_medium.py
import sys, os
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
        # D0 and D2 seeded with active outbreaks (non-adjacent boroughs).
        # D1 and D3 start with low infections that grow into danger within
        # 4-6 steps through spillover and their own spread rates.
        # With only 8 resources for 4 districts over 15 steps, the agent
        # cannot contain all districts simultaneously — genuine triage required.
        # A dumb agent scores ~0.35, a smart agent with good prioritisation
        # scores 0.60-0.70.
        seed_infections = [0.42, 0.10, 0.38, 0.10]

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