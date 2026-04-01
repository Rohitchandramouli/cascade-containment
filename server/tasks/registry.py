# server/tasks/registry.py
# ─────────────────────────────────────────────────────────────────────────────
# Maps task name strings to their classes.
# This is what environment.py and the evaluator use to select a task.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))    # reaches server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')) # reaches project root

from server.tasks.task_easy   import EasyTask
from server.tasks.task_medium import MediumTask
from server.tasks.task_hard   import HardTask
from server.tasks.base        import BaseTask
from typing import Dict, Type

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "easy":   EasyTask,
    "medium": MediumTask,
    "hard":   HardTask,
}


def get_task(name: str) -> BaseTask:
    """
    Return an instantiated task object by name.
    Raises ValueError for unrecognised task names.

    Usage:
        task = get_task("medium")
        initial_state = task.build_initial_state()
    """
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Valid options: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]()