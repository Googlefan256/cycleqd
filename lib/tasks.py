from typing import List, Tuple
from .models import ExpertModel
import numpy as np


class BaseTask:
    def __init__(
        self,
        name: str,
        expert_model_name: str,
        bc_num_dims: int,
        bc_min_vals: List[float],
        bc_max_vals: List[float],
        bc_grid_sizes: List[int],
    ):
        self.name = name
        self.expert_model_name = expert_model_name
        self.bc_num_dims = bc_num_dims
        self.bc_min_vals = bc_min_vals
        self.bc_max_vals = bc_max_vals
        self.bc_grid_sizes = bc_grid_sizes

    def get_bin_id(self, bc_idx: int, metric: float) -> int:
        bins = np.linspace(
            self.bc_min_vals[bc_idx],
            self.bc_max_vals[bc_idx],
            self.bc_grid_sizes[bc_idx] + 1,
        )
        return min(
            max(0, np.digitize(metric, bins, right=True) - 1),
            self.bc_grid_sizes[0] - 1,
        )

    def evaluate(self, model: ExpertModel) -> Tuple[float, int]:
        raise NotImplementedError("Return 0.0 ~ 1.0 tensor")
