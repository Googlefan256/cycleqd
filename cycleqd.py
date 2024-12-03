import torch
from config import CycleQDConfig
from models import ExpertModel, BaseModel
import random
import copy
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import random


class CycleQD:
    def __init__(self, config: CycleQDConfig):
        self.config = config
        self.experts = [
            ExpertModel(model_path, task.name)
            for model_path, task in zip(config.expert_models, config.tasks)
        ]
        self.base_model = BaseModel(config.base_model)
        self.tasks = config.tasks
        self.archives = {task.name: self.init_archive() for task in config.tasks}

    def init_archive(self):
        return [None for _ in range(self.config.cells)]

    def get_task_vector(self, expert: ExpertModel):
        expert_state_dict = expert.model.state_dict()
        base_state_dict = self.base_model.model.state_dict()
        task_vector = {}
        for key in expert_state_dict:
            if key in base_state_dict:
                task_vector[key] = expert_state_dict[key] - base_state_dict[key]
        return task_vector

    def linear_merge(self, task_vectors: list[dict], coefficients: list[float]):
        merged_task_vector = {
            k: torch.zeros_like(v, device="cuda") for k, v in task_vectors[0].items()
        }
        for vector, coeff in zip(task_vectors, coefficients):
            for key in merged_task_vector:
                merged_task_vector[key] += coeff * vector[key].to("cuda")
        base_state_dict = self.base_model.model.state_dict()
        merged_state_dict = {k: v.clone() for k, v in base_state_dict.items()}
        for key in merged_task_vector:
            merged_state_dict[key] += merged_task_vector[key].to("cpu")
        merged_model = copy.deepcopy(self.base_model.model)
        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    def svd_based_mutation(self, task_vector: dict, mutation_strength=0.1):
        mutated_task_vector = {}
        for key in tqdm(task_vector, desc="Mutation"):
            tensor = task_vector[key].to("cuda")
            if len(tensor.shape) > 1:
                U, S, Vh = torch.svd(tensor.clone().to(torch.float32), some=True)
                S_rand = torch.randn_like(S, device="cuda", dtype=torch.bfloat16)
                diag_S_rand = torch.diag_embed(S_rand)
                V = Vh.transpose(-2, -1)
                perturbation = U.to(torch.bfloat16) @ diag_S_rand @ V.to(torch.bfloat16)
                perturbation = perturbation.to(tensor.dtype).to(tensor.device)
                mutated_task_vector[key] = (
                    tensor + mutation_strength * perturbation
                ).to("cpu")
            else:
                mutated_task_vector[key] = tensor.to("cpu")
        return mutated_task_vector

    def calculate_gamma(self, performances: list[list[float]]):
        gammas = []
        for j in range(len(performances)):
            gamma_j = 1.0
            for i in range(len(performances[j])):
                f_ji = performances[j][i]
                f_min = min([performances[k][i] for k in range(len(performances))])
                f_max = max([performances[k][i] for k in range(len(performances))])
                alpha = self.config.alpha_low + (f_ji - f_min) / (f_max - f_min) * (
                    self.config.alpha_high - self.config.alpha_low
                )
                gamma_j *= alpha
            gammas.append(gamma_j)
        return gammas

    def cyclic_optimization(self):
        task_vectors = [self.get_task_vector(expert) for expert in self.experts]
        population = task_vectors
        for gen in range(self.config.generations):
            task_idx = gen % len(self.tasks)
            current_task = self.tasks[task_idx]
            print(f"Task: {current_task.name} / Generation: {gen}")
            performances = []
            bcs = []
            for model in population:
                merged_model = ExpertModel(
                    self.config.base_model,
                    "Placeholder",
                    self.linear_merge([model], [1.0]).state_dict(),
                )
                performance = current_task.evaluate(merged_model)
                bc_performance = [
                    task.evaluate(merged_model)
                    for task in self.tasks
                    if task != current_task
                ]
                performances.append(performance)
                bcs.append(bc_performance)
            gammas = self.calculate_gamma(performances)
            sampling_probs = F.softmax(torch.tensor(gammas), dim=0)
            new_population = []
            for _ in range(self.config.population_size):
                parent1, parent2 = random.choices(
                    population, weights=sampling_probs, k=2
                )
                if random.random() < self.config.crossover_rate:
                    child_task_vector = self.linear_merge(
                        [parent1, parent2], [0.5, 0.5]
                    ).state_dict()
                else:
                    child_task_vector = parent1
                if random.random() < self.config.mutation_rate:
                    child_task_vector = self.svd_based_mutation(child_task_vector)
                new_population.append(child_task_vector)
            alter_tasks = [task.name for task in self.tasks if task != current_task]
            for model, performance, bc in zip(new_population, performances, bcs):
                for ix, bc_ in enumerate(bc):
                    tn = alter_tasks[ix]
                    i = (
                        int(np.digitize(bc_, np.linspace(0, 1, self.config.cells - 1)))
                        - 1
                    )
                    if (
                        self.archives[tn][i] is None
                        or performance > self.archives[tn][i]["perf"]
                    ):
                        self.archives[tn][i] = dict(model=model, perf=performance)
            population = new_population
        final_models = []
        for task in self.tasks:
            for model_record in self.archives[task.name]:
                if model_record is not None:
                    final_models.append(model_record)
        final_coefficients = F.softmax(
            torch.tensor([model["perf"] for model in final_models]), dim=0
        )
        final_model_task_vector = {}
        for key in final_models[0]["model"]:
            final_model_task_vector[key] = sum(
                coeff * model["model"][key]
                for coeff, model in zip(final_coefficients, final_models)
            )
        final_model = self.linear_merge([final_model_task_vector], [1.0])
        return ExpertModel(
            self.config.base_model, "FinalModel", final_model.state_dict()
        )
