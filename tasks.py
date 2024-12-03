from models import ExpertModel


class BaseTask:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, model: ExpertModel):
        # Placeholder for evaluation logic
        raise NotImplementedError()
