# server/tasks/task_easy.py
import sys, os
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
        # D0 starts at the danger threshold — one district with a visible outbreak.
        # D1 is very clean, grows slowly through spillover only.
        # With TREATMENT_REDUCTION=0.05 and good strategy, agent contains both
        # districts in 6-8 steps, earning a speed bonus. Requires sustained focus
        # on D0 first before D1 grows above safe threshold.
        seed_infections = [0.06, 0.50]

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