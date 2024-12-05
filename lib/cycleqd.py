from typing import Dict, List, Optional, Tuple
from .archive import Archive
from .models import ExpertModel, BaseModel, ExpertAndMetrics
from .mutation import BaseMutator
from .merge import BaseMerger
from .sampler import BaseSampler
from .objects import Metrics
from .tasks import BaseTask
import torch


class CycleQD:
    def __init__(
        self,
        base_model: str,
        archive: Archive,
        tasks: List[BaseTask],
        sampler: BaseSampler,
        merger: BaseMerger,
        mutator: BaseMutator,
        storage: str,
    ):
        self.base_model = BaseModel(base_model)
        self.base_model_name = base_model
        self.tasks = tasks
        self.storage = storage
        for task in tasks:
            expert = ExpertModel(task.expert_model_name, task.name)
            self.__eval_store_model(expert, archive)
        self.sampler = sampler
        self.merger = merger
        self.mutator = mutator

    def __get_bc_ids(self, task_name: str, metrics: Dict[str, Metrics]) -> Tuple[int]:
        bc_ids = ()
        for k in metrics:
            if k != task_name:
                bc_ids += metrics[k].bc_ids
        return bc_ids

    def __eval_model_for_task(self, model: ExpertModel, task: BaseTask) -> Metrics:
        acc, bc_id = task.evaluate(model)
        return Metrics(quality=acc, bc_ids=(bc_id,))

    def __eval_store_model(
        self, model: ExpertModel, archive: Archive, task_name: Optional[str] = None
    ):
        metrics = {
            task.name: self.__eval_model_for_task(model, task) for task in self.tasks
        }
        if task_name:
            perf = {
                task_name: Metrics(
                    quality=metrics[task_name].quality,
                    bc_ids=self.__get_bc_ids(task_name, metrics),
                )
            }
        else:
            perf = {
                task.name: Metrics(
                    quality=metrics[task.name].quality,
                    bc_ids=self.__get_bc_ids(task.name, metrics),
                )
                for task in self.tasks
            }
        for key, value in perf.items():
            if archive.update(key, ExpertAndMetrics(model, value, self.storage)):
                print(
                    f"Updated with metrics.quality: {value.quality} for task_name: {key} at bc_ids: {value.bc_ids}"
                )
            else:
                print("Not updated")

    def __get_task_vector(self, model: ExpertModel) -> Dict[str, torch.Tensor]:
        adv = model.model.state_dict()
        new = {}
        for k, v in self.base_model.model.state_dict().items():
            new[k] = adv[k] - v
        return new

    def __append_task_vector(
        self, vector: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        new = {}
        for k, v in self.base_model.model.state_dict().items():
            new[k] = vector[k] + v
        return new

    def step(
        self,
        archive: Archive,
        task_name: str,
    ):
        dad, mom = archive.sample(task_name, self.sampler)
        print("Sampled dad and mom")
        child = self.merger.merge(
            [self.__get_task_vector(dad), self.__get_task_vector(mom)]
        )
        print("Got a child task vector")
        child = self.mutator.mutate(child)
        print("Mutated it")
        child = self.__append_task_vector(child)
        self.__eval_store_model(
            ExpertModel(self.base_model_name, task_name, child), archive, task_name
        )

    def best(self, archive: Archive) -> Tuple[ExpertModel, float]:
        best = None
        score = 0.0
        for model in archive.iter_all():
            model_score = 0.0
            expert = model.get_expert()
            for task in self.tasks:
                model_score += (
                    self.__eval_model_for_task(expert, task).quality * task.weights
                )
            if model_score >= score:
                best = expert
                score = model_score
        return best, score
