from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from src.agent.prompts import SYSTEM_PROMPT, VERIFICATION_PROMPT_TEMPLATE
from src.llm import LLMService


@dataclass
class VerificationResult:
    accepted: bool
    failed_constraints: List[str]
    risk_notes: List[str] | None = None


class VerificationAgent:
    def __init__(self, config, llm_service: LLMService | None = None):
        self.cfg = config
        self.llm = llm_service

    def verify(self, decision, bridge, day_data=None):
        failed = []
        for n, seq in decision.grid_buy.items():
            for h, buy in enumerate(seq):
                if buy > bridge.grid_exchange_cap[n][h] + 1e-9:
                    failed.append(f'grid_exchange_cap_violation_pdn_{n}_h_{h}')
        horizon = len(next(iter(decision.grid_buy.values()))) if decision.grid_buy else 0
        system_cap = self.cfg['short_term']['system_import_cap_default']
        for h in range(horizon):
            total_buy = sum(decision.grid_buy[n][h] for n in decision.grid_buy)
            if total_buy > system_cap + 1e-9:
                failed.append(f'system_import_cap_violation_h_{h}')
        for n, seq in decision.load_shed.items():
            if sum(seq) > self.cfg['long_term']['supply_inadequacy_limit']:
                failed.append(f'supply_inadequacy_limit_pdn_{n}')
        for n, seq in decision.battery_soc.items():
            for h, soc in enumerate(seq):
                if soc + 1e-9 < bridge.terminal_soc[n][h]:
                    failed.append(f'battery_terminal_soc_violation_pdn_{n}_h_{h}')
        for n, seq in decision.ev_soc.items():
            for h, soc in enumerate(seq):
                if soc + 1e-9 < bridge.ev_energy_requirement[n][h]:
                    failed.append(f'ev_service_violation_pdn_{n}_h_{h}')
        for n, seq in decision.line_loading.items():
            for h, loading in enumerate(seq):
                if loading > 1.08:
                    failed.append(f'line_loading_violation_pdn_{n}_h_{h}')
        for n, seq in decision.voltage_profile.items():
            for h, voltage in enumerate(seq):
                if voltage < 0.95 or voltage > 1.05:
                    failed.append(f'voltage_violation_pdn_{n}_h_{h}')
        if day_data is not None:
            for n, series in day_data.items():
                for h, t in enumerate(series):
                    lhs = (
                        (t.wind - decision.wind_curtail[n][h])
                        + (t.pv - decision.solar_curtail[n][h])
                        + decision.battery_discharge[n][h]
                        - decision.battery_charge[n][h]
                        + decision.ev_discharge[n][h]
                        - decision.ev_charge[n][h]
                        + decision.grid_buy[n][h]
                        - decision.grid_sell[n][h]
                        + decision.load_shed[n][h]
                    )
                    residual = abs(lhs - t.load)
                    if residual > 0.55:
                        failed.append(f'power_balance_residual_pdn_{n}_h_{h}')
        risk_notes = []
        if self.llm is not None:
            prompt = VERIFICATION_PROMPT_TEMPLATE.format(
                dispatch_summary={'grid_buy': decision.grid_buy, 'load_shed': decision.load_shed},
                bridge_summary={'grid_exchange_cap': bridge.grid_exchange_cap, 'ev_energy_requirement': bridge.ev_energy_requirement},
                measured_state_summary='structured checks already computed in runtime',
            )
            parsed = self.llm.call(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, agent_role='verification').parsed
            risk_notes = parsed.get('risk_notes', [])
            failed.extend(parsed.get('failed_constraints', []))
        failed = list(dict.fromkeys(failed))
        return VerificationResult(accepted=len(failed) == 0, failed_constraints=failed, risk_notes=risk_notes)
