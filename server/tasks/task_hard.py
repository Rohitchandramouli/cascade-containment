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
        # All six districts start with small but growing infections.
        # The 3-day lag means the agent observes these seed values on step 1
        # while true infection is already 3 days ahead — districts 0, 2, and 4
        # are likely in the danger zone before updated data arrives.
        # Pre-populate infection_history so build_observation can apply the lag
        # correctly from the very first step.
        seed_infections = [0.20, 0.14, 0.23, 0.11, 0.26, 0.17]

        infection_history = [
            seed_infections[:],  # day -3 (what agent sees on step 1)
            seed_infections[:],  # day -2
            seed_infections[:],  # day -1
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
