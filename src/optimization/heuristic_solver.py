from __future__ import annotations

from .schemas import OptimizationProblem, OptimizationResult
from .solver_base import SolverBackend


class HeuristicDispatchSolver(SolverBackend):
    """Practical fallback solver using deterministic dispatch heuristics."""

    name = 'heuristic'

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        horizon = problem.horizon
        num_pdns = problem.num_pdns
        forecasts = problem.forecasts
        limits = problem.constraints
        vars_out = {
            'battery_charge': {n: [0.0] * horizon for n in range(num_pdns)},
            'battery_discharge': {n: [0.0] * horizon for n in range(num_pdns)},
            'ev_charge': {n: [0.0] * horizon for n in range(num_pdns)},
            'ev_discharge': {n: [0.0] * horizon for n in range(num_pdns)},
            'grid_buy': {n: [0.0] * horizon for n in range(num_pdns)},
            'grid_sell': {n: [0.0] * horizon for n in range(num_pdns)},
            'load_shed': {n: [0.0] * horizon for n in range(num_pdns)},
            'wind_curtail': {n: [0.0] * horizon for n in range(num_pdns)},
            'solar_curtail': {n: [0.0] * horizon for n in range(num_pdns)},
        }
        objective = 0.0
        total_load_shed = 0.0
        total_curtailment = 0.0
        system_cap = limits.get('system_import_cap', 18.0)
        soc_b = {n: limits['initial_battery_soc'][n] for n in range(num_pdns)}
        soc_ev = {n: limits['initial_ev_soc'][n] for n in range(num_pdns)}

        for h in range(horizon):
            pending_buy = {}
            for n in range(num_pdns):
                load = forecasts['load'][n][h]
                pv = forecasts['pv'][n][h]
                wind = forecasts['wind'][n][h]
                trip = forecasts['trip_energy'][n][h]
                buy_price = forecasts['buy_price'][n][h]
                sell_price = forecasts['sell_price'][n][h]
                batt_cap = limits['battery_energy_cap'][n]
                batt_power = limits['battery_power_cap'][n]
                ev_floor = limits['ev_energy_requirement'][n][h]
                reserve = limits['terminal_soc'][n][h]
                export_cap = limits['export_cap'][n]
                net = load - pv - wind
                if net > 0:
                    pdis = min(max(0.0, soc_b[n] - reserve), batt_power, net)
                    soc_b[n] -= pdis
                    net -= pdis
                    vars_out['battery_discharge'][n][h] = pdis
                    ev_dis = min(max(0.0, soc_ev[n] - ev_floor), 1.5, net)
                    soc_ev[n] -= ev_dis
                    net -= ev_dis
                    vars_out['ev_discharge'][n][h] = ev_dis
                    pending_buy[n] = max(0.0, net)
                else:
                    surplus = -net
                    pch = min(batt_cap - soc_b[n], batt_power, surplus)
                    soc_b[n] += pch
                    surplus -= pch
                    vars_out['battery_charge'][n][h] = pch
                    ev_ch = min(max(0.0, limits['ev_energy_cap'][n] - soc_ev[n]), 1.5, surplus)
                    soc_ev[n] += ev_ch
                    surplus -= ev_ch
                    vars_out['ev_charge'][n][h] = ev_ch
                    sell = min(export_cap, surplus)
                    vars_out['grid_sell'][n][h] = sell
                    surplus -= sell
                    vars_out['wind_curtail'][n][h] = surplus * 0.5
                    vars_out['solar_curtail'][n][h] = surplus * 0.5
                    pending_buy[n] = 0.0
                    objective -= sell * sell_price
                soc_ev[n] = max(0.0, soc_ev[n] - trip)
                if soc_ev[n] < ev_floor:
                    vars_out['load_shed'][n][h] += ev_floor - soc_ev[n]
                    soc_ev[n] = ev_floor
            total_buy = sum(pending_buy.values())
            scale = min(1.0, system_cap / total_buy) if total_buy > 1e-9 else 1.0
            for n in range(num_pdns):
                buy = pending_buy[n] * scale
                vars_out['grid_buy'][n][h] = buy
                unsupplied = pending_buy[n] - buy
                if unsupplied > 0:
                    vars_out['load_shed'][n][h] += unsupplied
                objective += buy * forecasts['buy_price'][n][h]
                total_curtailment += vars_out['wind_curtail'][n][h] + vars_out['solar_curtail'][n][h]
                total_load_shed += vars_out['load_shed'][n][h]
                objective += 2.0 * (vars_out['wind_curtail'][n][h] + vars_out['solar_curtail'][n][h])
                objective += 20.0 * vars_out['load_shed'][n][h]
        return OptimizationResult(
            status='solved_heuristic',
            objective_value=objective,
            variables=vars_out,
            diagnostics={
                'backend': 'heuristic',
                'total_load_shed': total_load_shed,
                'total_curtailment': total_curtailment,
            },
        )
