from models import ExpertModel


class BaseTask:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, model: ExpertModel):
        raise NotImplementedError("Return 0.0 ~ 1.0 tensor")
