from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ExecutionResult:
    executed: bool
    status: str


class ExecutionAgent:
    """Execution Agent.

    Responsibilities from paper:
    - translate validated first-step decision into executable commands
    - return execution status flag
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def execute(self, decision):
        if not getattr(decision, 'grid_buy', None):
            return ExecutionResult(executed=False, status='no_dispatch_candidate')

        action_norm = 0.0
        max_loading = 0.0
        voltage_ok = True
        for n, seq in decision.grid_buy.items():
            if seq:
                action_norm += abs(seq[0])
                action_norm += abs(decision.grid_sell[n][0])
                action_norm += abs(decision.battery_charge[n][0]) + abs(decision.battery_discharge[n][0])
                action_norm += abs(decision.ev_charge[n][0]) + abs(decision.ev_discharge[n][0])
            if getattr(decision, 'line_loading', None) and decision.line_loading.get(n):
                max_loading = max(max_loading, decision.line_loading[n][0])
            if getattr(decision, 'voltage_profile', None) and decision.voltage_profile.get(n):
                voltage = decision.voltage_profile[n][0]
                voltage_ok = voltage_ok and 0.95 <= voltage <= 1.05

        if action_norm <= 1e-9:
            return ExecutionResult(executed=True, status='hold_setpoint')
        if max_loading > 1.08 or not voltage_ok:
            return ExecutionResult(executed=False, status='actuation_blocked_by_network_guard')
        return ExecutionResult(executed=True, status='simulated_execution_ok')
