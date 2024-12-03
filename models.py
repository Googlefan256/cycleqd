from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Dict
import torch


class BaseModel:
    def __init__(
        self,
        model_path: str,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path)
        if state_dict is None:
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
        else:
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(model_path), torch_dtype=torch.bfloat16
            )
            self.model.load_state_dict(state_dict)


class ExpertModel(BaseModel):
    def __init__(
        self,
        model_path: str,
        task_name: str,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(model_path, state_dict)
        self.task_name = task_name
