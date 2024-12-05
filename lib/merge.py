from typing import List, Dict
import torch


class BaseMerger:
    def __init__(self, std: float):
        self.std = std

    def merge(
        self, task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return NotImplementedError


class LinearMerger(BaseMerger):
    def merge(
        self, task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        num_task_vectors = len(task_vectors)
        weights = torch.normal(
            1.0,
            abs(self.std),
            size=(num_task_vectors,),
        )
        merged_params = {}
        for k, v in task_vectors[0].items():
            merged_params[k] = torch.zeros_like(v)
            for i in range(num_task_vectors):
                merged_params[k] += weights[i] * task_vectors[i][k]
            merged_params[k] /= weights.sum()
        return merged_params
