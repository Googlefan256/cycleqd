from models import ExpertModel


class BaseTask:
    def __init__(self, name: str, expert_model_name: str):
        self.name = name
        self.expert_model_name = expert_model_name

    def evaluate(self, model: ExpertModel):
        raise NotImplementedError("Return 0.0 ~ 1.0 tensor")
