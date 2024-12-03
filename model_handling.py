from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def get_parameters(self):
        return self.model.state_dict()

    def set_parameters(self, params):
        self.model.load_state_dict(params)


class ExpertModel(BaseModel):
    def __init__(self, model_path, base_model):
        super().__init__(model_path)
        self.base_parameters = base_model.get_parameters()

    def get_task_vector(self):
        expert_params = self.get_parameters()
        base_params = self.base_parameters
        task_vector = {k: expert_params[k] - base_params[k] for k in expert_params}
        return task_vector
