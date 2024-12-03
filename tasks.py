import torch


class BaseTask:
    def __init__(self, name):
        self.name = name

    def evaluate(self, model):
        # Placeholder for evaluation logic
        pass


class OsTask(BaseTask):
    def __init__(self):
        super().__init__("OS Task")

    def evaluate(self, model):
        # Evaluation logic for OS task
        return torch.rand(1)  # Random performance metric


class DBTask(BaseTask):
    def __init__(self):
        super().__init__("DB Task")

    def evaluate(self, model):
        # Evaluation logic for DB task
        return torch.rand(1)  # Random performance metric


class CodeTask(BaseTask):
    def __init__(self):
        super().__init__("Coding Task")

    def evaluate(self, model):
        # Evaluation logic for Coding task
        return torch.rand(1)  # Random performance metric
