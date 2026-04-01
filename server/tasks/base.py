# server/tasks/base.py
# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class that every task must implement.
# Defines the interface environment.py uses to initialise any episode.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from abc import ABC, abstractmethod
from models import CityState


class BaseTask(ABC):

    # These must be defined by every subclass
    name:           str
    num_districts:  int
    max_steps:      int
    resource_pool:  int
    data_lag_days:  int

    @abstractmethod
    def build_initial_state(self) -> CityState:
        """
        Return a freshly initialised CityState for a new episode.
        Called by environment.py at the start of every reset().
        """
        ...

    def __repr__(self) -> str:
        return f"Task(name={self.name}, districts={self.num_districts}, steps={self.max_steps})"