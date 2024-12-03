from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    base_model_path: str
    expert_model_paths: List[str]
    tasks: List[str]
    num_generations: int = 10
    population_size: int = 5
    mutation_strength: float = 0.1
    aggregation_frequency: int = 5
