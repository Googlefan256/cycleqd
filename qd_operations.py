import torch

from model_handling import ExpertModel


class QDOperations:
    def crossover(self, parent1: ExpertModel, parent2: ExpertModel, omega1, omega2):
        child_params = {}
        for k in parent1.get_task_vector():
            child_params[k] = (
                parent1.base_parameters[k]
                + (omega1 / (omega1 + omega2)) * parent1.get_task_vector()[k]
                + (omega2 / (omega1 + omega2)) * parent2.get_task_vector()[k]
            )
        return child_params

    def svd_mutation(self, task_vector, mutation_strength=0.1):
        mutated_params = {}
        for k in task_vector:
            param_mat = task_vector[k].detach()
            U, S, V = torch.svd(param_mat)
            perturbation = torch.randn_like(S) * mutation_strength
            S_perturbed = S + perturbation
            mutated_param = U @ (S_perturbed.unsqueeze(-1) * V.transpose(-2, -1))
            mutated_params[k] = mutated_param
        return mutated_params

    def model_aggregation(self, experts: ExpertModel, base_model, mixing_coeffs):
        aggregated_params = base_model.get_parameters()
        for coeff, expert in zip(mixing_coeffs, experts):
            for k in aggregated_params:
                aggregated_params[k] += coeff * expert.get_task_vector()[k]
        return aggregated_params
