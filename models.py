from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:
    def __init__(self, model_path, state_dict=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path
        )
        if state_dict is not None:
            self.model.load_state_dict(state_dict)


class ExpertModel(BaseModel):
    def __init__(self, model_path, task_name, state_dict=None):
        super().__init__(model_path, state_dict)
        self.task_name = task_name
