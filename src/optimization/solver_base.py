from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import OptimizationProblem, OptimizationResult


class SolverBackend(ABC):
    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        raise NotImplementedError
