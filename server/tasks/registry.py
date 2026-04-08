import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Dict, Type
from server.tasks.task_easy   import EasyTask
from server.tasks.task_medium import MediumTask
from server.tasks.task_hard   import HardTask
from server.tasks.base        import BaseTask

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "easy":   EasyTask,
    "medium": MediumTask,
    "hard":   HardTask,
}


def get_task(name: str) -> BaseTask:
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Valid options: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]()
