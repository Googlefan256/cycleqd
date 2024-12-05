from typing import Dict, List, Tuple
from .models import ExpertAndMetrics
from .sampler import BaseSampler
from .tasks import BaseTask


class Archive:
    def __init__(self, tasks: List[BaseTask]):
        self.__data: Dict[str, Dict[Tuple[int], ExpertAndMetrics]] = {
            task.name: {} for task in tasks
        }

    def sample(self, task_name: str, sampler: BaseSampler):
        dad, mom = sampler.sample(self.__data[task_name])
        return (
            self.__data[task_name][dad].get_expert(),
            self.__data[task_name][mom].get_expert(),
        )

    def update(self, task_name: str, expert: ExpertAndMetrics):
        if (
            expert.metrics.bc_ids not in self.__data[task_name]
            or self.__data[task_name][expert.metrics.bc_ids].metrics.quality
            < expert.metrics.quality
        ):
            if expert.metrics.bc_ids in self.__data[task_name]:
                self.__data[task_name][expert.metrics.bc_ids].drop()
            self.__data[task_name][expert.metrics.bc_ids] = expert
            return True
        expert.drop()
        return False

    def save(self, path: str):
        print("Saving final results")
        pass
