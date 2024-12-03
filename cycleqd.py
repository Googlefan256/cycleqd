import torch
from config import CycleQDConfig
from models import ExpertModel, BaseModel
import random
import copy
from torch.nn import functional as F


class CycleQD:
    def __init__(self, config: CycleQDConfig):
        self.config = config
        self.experts = [
            ExpertModel(model_path, task.name)
            for model_path, task in zip(config.expert_models, config.tasks)
        ]
        self.base_model = BaseModel(config.base_model)
        self.tasks = config.tasks

    def linear_merge(self, experts: list[ExpertModel], coefficients: list[float]):
        base_state_dict = self.base_model.model.state_dict()
        merged_state_dict = {k: v.clone() for k, v in base_state_dict.items()}
        for expert, coeff in zip(experts, coefficients):
            expert_state_dict = expert.model.state_dict()
            for key in merged_state_dict:
                if key in expert_state_dict:
                    merged_state_dict[key] += coeff * (
                        expert_state_dict[key] - base_state_dict[key]
                    )
        merged_model = copy.deepcopy(self.base_model.model)
        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    def gaussian_noise(self, model: ExpertModel, std: float):
        for param in model.model.parameters():
            noise = torch.randn_like(param) * std
            param.data += noise

    def cyclic_optimization(self):
        population = [self.base_model] + self.experts
        for gen in range(self.config.generations):
            print(f"Generation: {gen}")
            current_task = self.tasks[gen % len(self.tasks)]
            performances = [current_task.evaluate(model) for model in population]
            elites = sorted(zip(performances, population), reverse=True)
            elites = [pop for _, pop in elites[: self.config.population_size]]
            new_generation = []
            for pop in range(self.config.population_size):
                parent1, parent2 = random.sample(elites, 2)
                if torch.rand(1) < self.config.crossover_rate:
                    coeff1, coeff2 = torch.rand(2).tolist()
                    merged_model = self.linear_merge(
                        [parent1, parent2], [coeff1, coeff2]
                    )
                    child = ExpertModel(
                        self.config.base_model, "Child", merged_model.state_dict()
                    )
                else:
                    child = parent1
                if torch.rand(1) < self.config.mutation_rate:
                    self.gaussian_noise(child, self.config.noise_std)
                print(f"Add new generation/Pop:{pop}")
                new_generation.append(child)
            population = new_generation
        final_coefficients = F.softmax(torch.rand(len(self.experts)), dim=0)
        final_model = self.linear_merge(self.experts, final_coefficients.tolist())
        return ExpertModel(self.config.base_model, "Child", final_model.state_dict())
