from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Dict
import torch
from pathlib import Path
from uuid import uuid4
import shutil

from .objects import Metrics


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
        self.metrics: Optional[Metrics] = None


class ExpertAndMetrics:
    def __init__(self, expert: ExpertModel, metrics: Metrics, storage: str):
        self.metrics = metrics
        self.expert_path = Path(storage) / str(uuid4())
        self.task_name = expert.task_name
        print(f"Save model at: {self.expert_path}")
        expert.model.save_pretrained(self.expert_path)
        expert.tokenizer.save_pretrained(self.expert_path)
        with open(self.expert_path / "metrics.txt", "w") as w:
            w.write(
                f"Quality: {metrics.quality}\nBC Ids: {metrics.bc_ids}\nTask Name: {self.task_name}"
            )

    def drop(self):
        print(f"Delete model at: {self.expert_path}")
        shutil.rmtree(self.expert_path)

    def get_expert(self):
        print(f"Load model from: {self.expert_path}")
        return ExpertModel(self.expert_path, self.task_name)
