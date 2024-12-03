from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:
    def __init__(self, model_path, state_dict=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)


class ExpertModel(BaseModel):
    def __init__(self, model_path, task_name, state_dict=None):
        super().__init__(model_path, state_dict)
        self.task_name = task_name
