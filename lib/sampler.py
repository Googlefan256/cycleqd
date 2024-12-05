from typing import Tuple, Dict
from .models import ExpertAndMetrics
import numpy as np


class BaseSampler:
    def sample(
        self, to_sample: Dict[Tuple[int], ExpertAndMetrics]
    ) -> Tuple[Tuple[int], Tuple[int]]:
        raise NotImplementedError


class EliteSampler(BaseSampler):
    def sample(
        self, to_sample: Dict[Tuple[int], ExpertAndMetrics]
    ) -> Tuple[Tuple[int], Tuple[int]]:
        min_base = 0.5
        max_base = 0.8

        keys = list(to_sample.keys())
        keys_array = np.array(keys)

        qualities = np.array([to_sample[key].metrics.quality for key in keys])
        probs = self.__normalize(qualities, min_base, max_base)
        normalized_keys = [
            self.__normalize(keys_array[:, i], min_base, max_base)
            for i in range(keys_array.shape[1])
        ]
        for normalized_key in normalized_keys:
            probs *= normalized_key
        if np.sum(probs) == 0:
            probs = np.ones(len(probs))
        probs = probs / np.sum(probs)
        parents = (
            np.random.choice(len(keys), 2, p=probs, replace=False)
            if len(keys) > 1
            else [0, 0]
        )
        dad = keys[parents[0]]
        mom = keys[parents[1]]
        # to_sample[dad].sampling_freq += 1
        # to_sample[mom].sampling_freq += 1
        return dad, mom

    def __normalize(
        self, values: np.ndarray, min_base: float, max_base: float
    ) -> np.ndarray:
        min_value = np.min(values)
        max_value = np.max(values)
        if min_value == max_value:
            return np.ones_like(values)
        return min_base + (values - min_value) * (max_base / (max_value - min_value))


class RandomSampler(BaseSampler):
    def sample(
        self, to_sample: Dict[Tuple[int], ExpertAndMetrics]
    ) -> Tuple[Tuple[int], Tuple[int]]:
        keys = list(to_sample.keys())
        probs = np.ones(len(keys)) / len(keys)
        parents = np.random.choice(len(keys), 2, p=probs, replace=False)
        dad = keys[parents[0]]
        mom = keys[parents[1]]
        return dad, mom
