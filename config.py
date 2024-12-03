from tasks import BaseTask
from models import BaseModel, ExpertModel


class CycleQDConfig:
    def __init__(
        self,
        expert_models: list[ExpertModel],
        base_model: BaseModel,
        tasks: list[BaseTask],
        population_size=100,
        generations=10,
        mutation_rate=0.1,
        crossover_rate=0.5,
        noise_std=0.01,
    ):
        self.expert_models = expert_models
        self.base_model = base_model
        self.tasks = tasks
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.noise_std = noise_std
