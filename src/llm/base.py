from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

from .schemas import LLMRequest, LLMResponse


class BaseLLMClient(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
