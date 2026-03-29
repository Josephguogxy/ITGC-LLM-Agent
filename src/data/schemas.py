from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any


@dataclass
class TimeStepData:
    timestamp: str
    load: float
    pv: float
    wind: float
    trip_energy: float
    buy_price: float
    sell_price: float
    frequency: float = 50.0
    voltage: float = 1.0
    temperature: float = 25.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PDNState:
    pdn_id: int
    series: List[TimeStepData] = field(default_factory=list)


@dataclass
class StreamingSnapshot:
    step: int
    pdn_states: Dict[int, TimeStepData]
    meta: Dict[str, Any] = field(default_factory=dict)
