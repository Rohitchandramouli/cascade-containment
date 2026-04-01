# server/tasks/task_hard.py
# ─────────────────────────────────────────────────────────────────────────────
# Hard task: 6 districts, 3-day data lag, scarce resources.
# All districts start with small but growing infections.
# Agent must learn to read growth_rate_hint signals and act proactively
# on districts that look manageable today but will be critical in 3 days.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))    # reaches server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')) # reaches project root

from models import CityState
from server.utils import generate_districts
from server.tasks.base import BaseTask
from server.constants import TASK_CONFIG


class HardTask(BaseTask):

    name          = "hard"
    num_districts = TASK_CONFIG["hard"]["num_districts"]    # 6
    max_steps     = TASK_CONFIG["hard"]["max_steps"]        # 20
    resource_pool = TASK_CONFIG["hard"]["resource_pool"]    # 7
    data_lag_days = TASK_CONFIG["hard"]["data_lag_days"]    # 3

    def build_initial_state(self) -> CityState:
        # All districts start with small infections that grow at different rates.
        # The 3-day lag means the agent won't see today's true rates until day 3.
        # Districts with high true_spread_rate will accelerate invisibly.
        seed_infections = [0.10, 0.08, 0.12, 0.07, 0.15, 0.09]

        # Pre-populate infection_history with 3 days of identical
        # starting values so the lag mechanic works from step 1.
        initial_rates   = seed_infections[:]
        infection_history = [
            initial_rates[:],   # day -3 (what agent sees on step 1)
            initial_rates[:],   # day -2
            initial_rates[:],   # day -1
        ]

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
            infection_history   = infection_history,
        )