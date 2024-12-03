from config import CycleQDConfig
from cycleqd import CycleQD
from tasks import BaseTask
from models import ExpertModel
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


class Task1(BaseTask):
    def __init__(self):
        super().__init__("Task 1")

    def evaluate(self, model: ExpertModel):
        lm = HFLM(
            pretrained=model.model.cuda(), backend="causal", tokenizer=model.tokenizer
        )
        res = simple_evaluate(lm, tasks=["mmlu_electrical_engineering"])["results"]["mmlu_electrical_engineering"]["acc,none"]
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
        res = simple_evaluate(lm, tasks=["mmlu_college_computer_science"])["results"]["mmlu_college_computer_science"]["acc,none"]
        print(f"Task2: {res}")
        model.model.cpu()
        return torch.tensor([res])


expert_models = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
base_model = "Qwen/Qwen2.5-0.5B"

# Define tasks
tasks = [Task1(), Task2()]

# Create configuration
config = CycleQDConfig(
    expert_models=expert_models,
    base_model=base_model,
    tasks=tasks,
    population_size=5,
    generations=10,
)

# Initialize CycleQD
cycle_qd = CycleQD(config)

for task in tasks:
    performance = task.evaluate(cycle_qd.base_model)
    print(f"Base performance on {task.name}: {performance.item()}")

# Perform cyclic optimization
final_model = cycle_qd.cyclic_optimization()

# Evaluate final model on all tasks
for task in tasks:
    performance = task.evaluate(final_model)
    print(f"Base performance on {task.name}: {performance.item()}")
