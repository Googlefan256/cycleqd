from qd_operations import QDOperations
import random


class QDFramework:
    def __init__(self, base_model, experts, tasks):
        self.base_model = base_model
        self.experts = experts
        self.tasks = tasks
        self.archives = {task: [] for task in tasks}
        self.qd_ops = QDOperations()
        self.evaluations = {task: [] for task in tasks}

        # Initialize archives with expert models
        for expert, task in zip(self.experts, self.tasks):
            self.archives[task].append(expert)

    def alternate_quality_bcs(self, current_task):
        quality_task = current_task
        bcs_tasks = [task for task in self.tasks if task != current_task]
        return quality_task, bcs_tasks

    def update_archive(self, task, model, evaluation_score):
        self.archives[task].append(model)
        self.evaluations[task].append(evaluation_score)

    def select_models(self, num_models):
        available_models = []
        for task in self.tasks:
            available_models.extend(self.archives[task])
        if len(available_models) < num_models:
            raise ValueError(
                f"Not enough models in archives to select {num_models} models. Available: {len(available_models)}"
            )
        return random.choices(available_models, k=num_models)
