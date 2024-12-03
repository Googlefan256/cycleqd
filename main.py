from config import CycleQDConfig
from cycleqd import CycleQD
from tasks import BaseTask
from models import ExpertModel
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import logging

logging.disable(100)


class Task1(BaseTask):
    def __init__(self):
        super().__init__("Task 1")

    def evaluate(self, model: ExpertModel):
        lm = HFLM(
            pretrained=model.model.cuda(), backend="causal", tokenizer=model.tokenizer
        )
        res = simple_evaluate(
            lm,
            tasks=["mmlu_electrical_engineering"],
            random_seed=None,
            numpy_random_seed=None,
            torch_random_seed=None,
            fewshot_random_seed=None,
        )["results"]["mmlu_electrical_engineering"]["acc,none"]
        print(f"Task1: {res}")
        model.model.cpu()
        return torch.tensor([res])


class Task2(BaseTask):
    def __init__(self):
        super().__init__("Task  2")

    def evaluate(self, model: ExpertModel):
        lm = HFLM(
            pretrained=model.model.cuda(), backend="causal", tokenizer=model.tokenizer
        )
        res = simple_evaluate(
            lm,
            tasks=["mmlu_college_computer_science"],
            random_seed=None,
            numpy_random_seed=None,
            torch_random_seed=None,
            fewshot_random_seed=None,
        )["results"]["mmlu_college_computer_science"]["acc,none"]
        print(f"Task2: {res}")
        model.model.cpu()
        return torch.tensor([res])


expert_models = [
    "AXCXEPT/EZO-Common-T2-2B-gemma-2-it",
    "google/gemma-2-2b-it",
    "cognitivecomputations/dolphin-2.9.4-gemma2-2b",
]
base_model = "google/gemma-2-2b"

# Define tasks
tasks = [Task1(), Task2()]

# Create configuration
config = CycleQDConfig(
    expert_models=expert_models,
    base_model=base_model,
    tasks=tasks,
    population_size=5,
    generations=30,
    cells=2,
)

cycle_qd = CycleQD(config)

final_model = cycle_qd.cyclic_optimization()

for task in tasks:
    performance = task.evaluate(final_model)
    print(f"Result performance on {task.name}: {performance.item()}")

final_model.model.save_pretrained("./results")
