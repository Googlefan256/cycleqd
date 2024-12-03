import torch


class BaseTask:
    def __init__(self, name):
        self.name = name

    def evaluate(self, model):
        # Placeholder for evaluation logic
        pass
