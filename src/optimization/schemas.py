from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class OptimizationProblem:
    horizon: int
    num_pdns: int
    objective_weights: Dict[str, float]
    parameters: Dict[str, Any]
    forecasts: Dict[str, Any]
    constraints: Dict[str, Any]


@dataclass
class OptimizationResult:
    status: str
    objective_value: float
    variables: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
