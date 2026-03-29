from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from .schemas import TimeStepData, StreamingSnapshot


class BaseDataProvider(ABC):
    @abstractmethod
    def generate_day(self, num_pdns: int, seed: int = 42) -> Dict[int, List[TimeStepData]]:
        raise NotImplementedError

    @abstractmethod
    def stream(self, num_pdns: int, steps: int, seed: int = 42):
        raise NotImplementedError
