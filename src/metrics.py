from __future__ import annotations

from dataclasses import dataclass, asdict

@dataclass
class Metrics:
    total_cost: float = 0.0
    import_cost: float = 0.0
    export_revenue: float = 0.0
    curtailment: float = 0.0
    load_shedding: float = 0.0
    peak_import: float = 0.0
    battery_throughput: float = 0.0
    ev_throughput: float = 0.0
    ev_mobility_shortage: float = 0.0
    rollback_count: int = 0
    renewable_utilization: float = 0.0
    accepted_cycles: int = 0
    total_cycles: int = 0
    verification_failures: int = 0
    admm_iterations: float = 0.0
    benders_iterations: float = 0.0

    def to_dict(self):
        return asdict(self)
