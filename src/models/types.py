from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LongTermPlan:
    battery_energy: Dict[int, float]
    battery_power: Dict[int, float]
    v2g_ratio: Dict[int, float]
    import_cap: Dict[int, float]
    export_cap: Dict[int, float]


@dataclass
class BridgeVariables:
    terminal_soc: Dict[int, List[float]]
    reserve_requirement: Dict[int, List[float]]
    grid_exchange_cap: Dict[int, List[float]]
    ev_energy_requirement: Dict[int, List[float]]
    risk_budget: Dict[int, List[float]]


@dataclass
class ShortTermDecision:
    battery_charge: Dict[int, List[float]] = field(default_factory=dict)
    battery_discharge: Dict[int, List[float]] = field(default_factory=dict)
    ev_charge: Dict[int, List[float]] = field(default_factory=dict)
    ev_discharge: Dict[int, List[float]] = field(default_factory=dict)
    wind_curtail: Dict[int, List[float]] = field(default_factory=dict)
    solar_curtail: Dict[int, List[float]] = field(default_factory=dict)
    grid_buy: Dict[int, List[float]] = field(default_factory=dict)
    grid_sell: Dict[int, List[float]] = field(default_factory=dict)
    load_shed: Dict[int, List[float]] = field(default_factory=dict)
    battery_soc: Dict[int, List[float]] = field(default_factory=dict)
    ev_soc: Dict[int, List[float]] = field(default_factory=dict)
    line_loading: Dict[int, List[float]] = field(default_factory=dict)
    voltage_profile: Dict[int, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationalState:
    battery_soc: float
    ev_soc: float
    risk_budget: float
    last_voltage: float = 1.0
    last_line_loading: float = 0.0


@dataclass
class PlanningScenario:
    scenario_id: str
    probability: float
    day_data: Dict[int, List[Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleFeedback:
    operating_cost: float
    curtailed_renewable: float
    supply_inadequacy: float
    peak_import: float
    v2g_intensity: float
    accepted: bool
    rollback_triggered: bool
    renewable_utilization: float = 0.0
    verification_failures: int = 0
    admm_iterations: float = 0.0
    benders_iterations: float = 0.0
