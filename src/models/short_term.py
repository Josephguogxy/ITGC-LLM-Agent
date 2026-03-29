from __future__ import annotations

from typing import Dict, Any

from src.models.types import LongTermPlan, BridgeVariables, ShortTermDecision


class ShortTermDispatcher:
    """Approximate short-term rolling dispatch with paper-aligned variables."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def build_bridge_variables(self, plan: LongTermPlan, num_pdns: int, horizon: int) -> BridgeVariables:
        risk0 = self.cfg['agent']['default_risk_budget']
        reserve = self.cfg['short_term']['reserve_requirement_default']
        return BridgeVariables(
            terminal_soc={n: [0.18 * plan.battery_energy[n] if h < horizon - 4 else 0.24 * plan.battery_energy[n] for h in range(horizon)] for n in range(num_pdns)},
            reserve_requirement={n: [0.55 * reserve if h in range(10, 17) else reserve for h in range(horizon)] for n in range(num_pdns)},
            grid_exchange_cap={n: [plan.import_cap[n] for _ in range(horizon)] for n in range(num_pdns)},
            ev_energy_requirement={n: [0.45 if h not in (6, 7, 8, 17, 18, 19) else 0.8 * max(plan.v2g_ratio[n], 0.4) for h in range(horizon)] for n in range(num_pdns)},
            risk_budget={n: [risk0] * horizon for n in range(num_pdns)},
        )

    def solve_day(self, day_data, plan: LongTermPlan, bridge: BridgeVariables) -> ShortTermDecision:
        num_pdns = len(day_data)
        horizon = len(next(iter(day_data.values())))
        decision = ShortTermDecision(
            battery_charge={n: [0.0]*horizon for n in range(num_pdns)},
            battery_discharge={n: [0.0]*horizon for n in range(num_pdns)},
            ev_charge={n: [0.0]*horizon for n in range(num_pdns)},
            ev_discharge={n: [0.0]*horizon for n in range(num_pdns)},
            wind_curtail={n: [0.0]*horizon for n in range(num_pdns)},
            solar_curtail={n: [0.0]*horizon for n in range(num_pdns)},
            grid_buy={n: [0.0]*horizon for n in range(num_pdns)},
            grid_sell={n: [0.0]*horizon for n in range(num_pdns)},
            load_shed={n: [0.0]*horizon for n in range(num_pdns)},
        )
        soc_b = {n: 0.6 * plan.battery_energy[n] for n in range(num_pdns)}
        soc_ev = {n: max(1.0, 0.75 * plan.v2g_ratio[n] * 4.0) for n in range(num_pdns)}
        batt_power = plan.battery_power
        system_cap = self.cfg['short_term']['system_import_cap_default']

        for h in range(horizon):
            buy_requests = {}
            for n, series in day_data.items():
                t = series[h]
                net = t.load - t.pv - t.wind
                min_ev = bridge.ev_energy_requirement[n][h]
                bat_floor = bridge.terminal_soc[n][h]
                if net > 0:
                    pdis = min(max(0.0, soc_b[n] - bat_floor), batt_power[n], net)
                    soc_b[n] -= pdis
                    net -= pdis
                    decision.battery_discharge[n][h] = pdis
                    pevdis = min(max(0.0, soc_ev[n] - min_ev), 1.8, net)
                    soc_ev[n] -= pevdis
                    net -= pevdis
                    decision.ev_discharge[n][h] = pevdis
                    buy_requests[n] = max(0.0, net)
                else:
                    surplus = -net
                    pch = min(plan.battery_energy[n] - soc_b[n], batt_power[n], surplus)
                    soc_b[n] += pch
                    surplus -= pch
                    decision.battery_charge[n][h] = pch
                    pevch = min(max(0.0, 4.5 - soc_ev[n]), 1.8, surplus)
                    soc_ev[n] += pevch
                    surplus -= pevch
                    decision.ev_charge[n][h] = pevch
                    decision.grid_sell[n][h] = min(plan.export_cap[n], surplus * 0.35)
                    surplus -= decision.grid_sell[n][h]
                    decision.wind_curtail[n][h] = max(0.0, surplus * 0.45)
                    decision.solar_curtail[n][h] = max(0.0, surplus * 0.55)
                    buy_requests[n] = 0.0
                soc_ev[n] = max(0.0, soc_ev[n] - t.trip_energy)
                if soc_ev[n] < min_ev:
                    decision.load_shed[n][h] += 0.6 * (min_ev - soc_ev[n])
                    soc_ev[n] = min_ev
            total_buy = sum(buy_requests.values())
            scale = min(1.0, system_cap / total_buy) if total_buy > 1e-9 else 1.0
            for n, req in buy_requests.items():
                decision.grid_buy[n][h] = req * scale
                if req > decision.grid_buy[n][h]:
                    decision.load_shed[n][h] += 0.7 * (req - decision.grid_buy[n][h])
        return decision
