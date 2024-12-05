from typing import Dict, Optional
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseMutator:
    default_mutation_params: np.ndarray

    def _mutate(
        self,
        task_vector: Dict[str, torch.Tensor],
        mutation_params: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def mutate(
        self,
        task_vector: Dict[str, torch.Tensor],
        mutation_params: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._mutate(
            task_vector,
            (
                mutation_params
                if mutation_params is not None
                else self.default_mutation_params
            ),
        )


class GaussianMutator(BaseMutator):

    def __init__(self, mutation_rate: float):
        self.default_mutation_params = np.array(mutation_rate)

    def _mutate(
        self,
        task_vector: Dict[str, torch.Tensor],
        mutation_params: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        for key, value in task_vector.items():
            task_vector[key] = task_vector[key].to(device)
            task_vector[key] += torch.normal(
                0,
                abs(float(mutation_params)),
                value.shape,
                device=device,
                dtype=value.dtype,
            )
            task_vector[key] = task_vector[key].to("cpu")
        return task_vector
