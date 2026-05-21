from abc import ABC
from typing import Dict, List

import pandas as pd
import timeit


class Benchmark(ABC):
    def __init__(self) -> None:
        self.duration: List[float] = []  # in ms (float number)
        self.metadata: Dict[str, list[float]] = {}

    def start(self):
        self.duration.append(timeit.default_timer())

    def stop(self):
        self.duration[-1] = (timeit.default_timer() - self.duration[-1]) * 1000.0

    def save(self, filename: str):
        data = pd.DataFrame({"duration": self.duration})
        metadata = pd.DataFrame(self.metadata)
        data = pd.concat([data, metadata], axis=1)
        data.to_csv(
            f"{filename}_benchmark.csv",
            index=False,
        )

    def clear(self):
        self.duration.clear()
        for _, value in self.metadata.items():
            value.clear()
