from dataclasses import dataclass
from typing import Tuple


@dataclass
class Metrics:
    quality: float
    bc_ids: Tuple[int]
