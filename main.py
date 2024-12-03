import logging
from training_loop import training_loop
from config import Config

logging.basicConfig(level=logging.INFO)


def main():
    config = Config(
        base_model_path="google/gemma-2-2b-it",
        expert_model_paths=[
            "google/gemma-2-2b-it",
            "google/gemma-2-2b-it",
            "google/gemma-2-2b-it",
        ],
        tasks=["coding", "os", "db"],
        num_generations=10,
        population_size=3,
        mutation_strength=0.1,
        aggregation_frequency=5,
    )
    training_loop(config)


if __name__ == "__main__":
    main()
