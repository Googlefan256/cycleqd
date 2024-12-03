from config import CycleQDConfig
from cycleqd import CycleQD
from tasks import BaseTask
from models import BaseModel
import torch


class CodeTask(BaseTask):
    def __init__(self):
        super().__init__("Coding Task")

    def evaluate(self, model: BaseModel):
        # Evaluation logic for Coding task
        return torch.rand(1)  # Random performance metric


expert_models = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
]
base_model = "Qwen/Qwen2.5-0.5B-Instruct"

# Define tasks
tasks = [CodeTask()]

# Create configuration
config = CycleQDConfig(
    expert_models=expert_models,
    base_model=base_model,
    tasks=tasks,
    population_size=3,
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
