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
        res = simple_evaluate(
            lm,
            tasks=["mmlu_electrical_engineering"],
            random_seed=None,
            numpy_random_seed=None,
            torch_random_seed=None,
            fewshot_random_seed=None,
            verbosity="ERROR",
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
            verbosity="ERROR",
        )["results"]["mmlu_college_computer_science"]["acc,none"]
        print(f"Task2: {res}")
        model.model.cpu()
        return torch.tensor([res])


expert_models = [
    "caelancooper/Qwen2.5-0.5B-business",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "artificialguybr/Qwen2.5-0.5B-OpenHermes2.5",
]
base_model = "Qwen/Qwen2.5-0.5B"

# Define tasks
tasks = [Task1(), Task2()]

# Create configuration
config = CycleQDConfig(
    expert_models=expert_models,
    base_model=base_model,
    tasks=tasks,
    population_size=2,
    generations=10,
    cells=3,
)

cycle_qd = CycleQD(config)

begin_perfs = []
for task in tasks:
    begin_perfs.append(task.evaluate(cycle_qd.base_model))

final_model = cycle_qd.cyclic_optimization()

for task, be_performance in zip(tasks, begin_perfs):
    performance = task.evaluate(final_model)
    print(
        f"Result performance on {task.name}: {be_performance.item()}->{performance.item()}"
    )

final_model.model.save_pretrained("./results")
