from tasks import BaseTask


class CycleQDConfig:
    def __init__(
        self,
        base_model: str,
        tasks: list[BaseTask],
        population_size=100,
        generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        noise_std=0.01,
        alpha_low=0.1,
        alpha_high=1.0,
        cells=2,
    ):
        self.base_model = base_model
        self.tasks = tasks
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.noise_std = noise_std
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.cells = cells
