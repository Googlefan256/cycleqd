# training_loop.py
# Simulating the training loop for CycleQD with evaluation handling

from qd_framework import QDFramework
from model_handling import BaseModel, ExpertModel
from task_evaluation import CodingEvaluation, OSEvaluation, DBEvaluation
import random
import logging
import os
import datetime
import pickle
import torch
from config import Config
import dataclasses
import json

logging.basicConfig(level=logging.INFO)


def training_loop(config: Config):
    # Initialize models and framework
    base_model = BaseModel(config.base_model_path)
    experts = [ExpertModel(path, base_model) for path in config.expert_model_paths]
    framework = QDFramework(base_model, experts, tasks=config.tasks)

    # Initialize evaluation classes
    task_evaluations = {
        "coding": CodingEvaluation(),
        "os": OSEvaluation(),
        "db": DBEvaluation(),
    }

    # Create results directory
    results_dir = os.path.join(
        "results", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(results_dir, exist_ok=True)

    for gen in range(config.num_generations):
        current_task = framework.tasks[gen % len(framework.tasks)]
        logging.info(f"Generation {gen}: Current Task - {current_task}")

        quality_task, bcs_tasks = framework.alternate_quality_bcs(current_task)

        selected_models = framework.select_models(config.population_size)
        new_models = []
        for _ in range(config.population_size):
            parent1, parent2 = random.sample(selected_models, 2)
            child_params = framework.qd_ops.crossover(
                parent1,
                parent2,
                omega1=torch.randn(1).abs().item(),
                omega2=torch.randn(1).abs().item(),
            )
            mutated_child = framework.qd_ops.svd_mutation(
                child_params, config.mutation_strength
            )
            new_models.append(mutated_child)
        new_models = []
        for _ in range(config.population_size):
            parent1, parent2 = random.sample(selected_models, 2)
            child_params = framework.qd_ops.crossover(
                parent1,
                parent2,
                omega1=torch.randn(1).abs().item(),
                omega2=torch.randn(1).abs().item(),
            )
            mutated_child = framework.qd_ops.svd_mutation(
                child_params, config.mutation_strength
            )
            # Create a new ExpertModel instance for the new model
            new_expert = ExpertModel(None, base_model)
            new_expert.set_parameters(mutated_child)
            new_models.append(new_expert)
        # Update archives
        for model in new_models:
            # Evaluate the model on the current task
            base_model.set_parameters(model)
            evaluation_score = task_evaluations[current_task].evaluate(base_model.model)
            framework.update_archive(current_task, model, evaluation_score)

        # Aggregate models periodically
        if (gen + 1) % config.aggregation_frequency == 0:
            mixing_coeffs = torch.rand(len(experts))
            mixing_coeffs = mixing_coeffs / mixing_coeffs.sum()
            aggregated_params = framework.qd_ops.model_aggregation(
                experts, base_model, mixing_coeffs
            )
            framework.update_archive("aggregated", aggregated_params, evaluation_score)

            # Save aggregated model
            aggregated_model_path = os.path.join(
                results_dir, f"aggregated_model_gen_{gen}.pth"
            )
            torch.save(aggregated_params, aggregated_model_path)
            logging.info(f"Aggregated model saved at {aggregated_model_path}")

    # Save configuration and evaluation results
    config_path = os.path.join(results_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(json.dumps(dataclasses.asdict(config)))
    logging.info(f"Configuration saved at {config_path}")

    evaluations_path = os.path.join(results_dir, "evaluations.pkl")
    with open(evaluations_path, "wb") as f:
        pickle.dump(framework.evaluations, f)
    logging.info(f"Evaluation results saved at {evaluations_path}")
