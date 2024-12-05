from typing import Tuple
from lib import BaseTask, ExpertModel, EliteSampler, LinearMerger, GaussianMutator
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


class Task1(BaseTask):
    def __init__(self):
        super().__init__(
            "MMLU Electrical Engineering",
            "Qwen/Qwen2.5-0.5B-Instruct",
            1,
            [0.0],
            [1.0],
            [15],
        )

    def evaluate(self, model: ExpertModel) -> Tuple[float, int]:
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
        print(f"Task Result / Task: {self.name}: {res}")
        model.model.cpu()
        return res, self.get_bin_id(0, res)


class Task2(BaseTask):
    def __init__(self):
        super().__init__(
            "MMLU College Computer Science",
            "artificialguybr/Qwen2.5-0.5B-OpenHermes2.5",
            1,
            [0.0],
            [1.0],
            [15],
        )

    def evaluate(self, model: ExpertModel) -> Tuple[float, int]:
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
        print(f"Task Result / Task: {self.name}: {res}")
        model.model.cpu()
        return res, self.get_bin_id(0, res)


tasks = [Task1(), Task2()]
base_model = "Qwen/Qwen2.5-0.5B"
sampler = EliteSampler()
merger = LinearMerger(0.01)
mutator = GaussianMutator(0.003)
steps = 20
storage = "./store"
step_steps = 5
