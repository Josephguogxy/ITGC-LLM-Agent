from __future__ import annotations

import math
from typing import Dict, List

from src.optimization.schemas import OptimizationProblem, OptimizationResult
from src.optimization.solver_base import SolverBackend


class DistributedADMMDispatcher(SolverBackend):
    name = "distributed_admm"

    def __init__(self, rho: float = 1.0, max_iter: int = 10, tol: float = 1e-3):
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        horizon = problem.horizon
        num_pdns = problem.num_pdns
        c = problem.constraints
        rho = float(problem.parameters.get("admm_rho", self.rho))
        max_iter = int(problem.parameters.get("admm_max_iter", self.max_iter))
        tol = float(problem.parameters.get("admm_tol", self.tol))

        desired = [[0.0] * horizon for _ in range(num_pdns)]
        consensus = [[0.0] * horizon for _ in range(num_pdns)]
        dual = [[0.0] * horizon for _ in range(num_pdns)]
        residual_trace: List[dict] = []

        for iteration in range(1, max_iter + 1):
            for n in range(num_pdns):
                desired[n], _ = self._simulate_local(problem, n, fixed_grid_buy=None, dual=dual[n], consensus=consensus[n])

            updated = self._project_imports(desired, c["grid_exchange_cap"], c["system_import_cap"])
            primal = 0.0
            dual_residual = 0.0
            for n in range(num_pdns):
                for t in range(horizon):
                    primal = max(primal, abs(desired[n][t] - updated[n][t]))
                    dual_residual = max(dual_residual, abs(updated[n][t] - consensus[n][t]))
                    dual[n][t] += rho * (desired[n][t] - updated[n][t])
            consensus = updated
            residual_trace.append(
                {
                    "iteration": iteration,
                    "primal_residual": primal,
                    "dual_residual": dual_residual,
                }
            )
            if primal <= tol and dual_residual <= tol:
                break

        vars_out = {
            "battery_charge": {n: [] for n in range(num_pdns)},
            "battery_discharge": {n: [] for n in range(num_pdns)},
            "ev_charge": {n: [] for n in range(num_pdns)},
            "ev_discharge": {n: [] for n in range(num_pdns)},
            "grid_buy": {n: [] for n in range(num_pdns)},
            "grid_sell": {n: [] for n in range(num_pdns)},
            "load_shed": {n: [] for n in range(num_pdns)},
            "wind_curtail": {n: [] for n in range(num_pdns)},
            "solar_curtail": {n: [] for n in range(num_pdns)},
            "battery_soc": {n: [] for n in range(num_pdns)},
            "ev_soc": {n: [] for n in range(num_pdns)},
            "line_loading": {n: [] for n in range(num_pdns)},
            "voltage_profile": {n: [] for n in range(num_pdns)},
        }
        objective = 0.0
        for n in range(num_pdns):
            _, local = self._simulate_local(problem, n, fixed_grid_buy=consensus[n], dual=dual[n], consensus=consensus[n])
            objective += local["objective"]
            for key in vars_out:
                vars_out[key][n] = local[key]

        return OptimizationResult(
            status="feasible",
            objective_value=objective,
            variables=vars_out,
            diagnostics={
                "backend": self.name,
                "iterations": len(residual_trace),
                "rho": rho,
                "residual_trace": residual_trace,
                "system_import_profile": [sum(consensus[n][t] for n in range(num_pdns)) for t in range(horizon)],
            },
        )

    def _project_imports(self, desired, grid_caps, system_cap):
        num_pdns = len(desired)
        horizon = len(desired[0]) if desired else 0
        projected = [[0.0] * horizon for _ in range(num_pdns)]
        for t in range(horizon):
            capped = [max(0.0, min(desired[n][t], grid_caps[n][t])) for n in range(num_pdns)]
            total = sum(capped)
            cap_t = system_cap[t] if isinstance(system_cap, list) else system_cap
            if total <= cap_t + 1e-9:
                for n in range(num_pdns):
                    projected[n][t] = capped[n]
                continue
            scale = cap_t / max(total, 1e-9)
            for n in range(num_pdns):
                projected[n][t] = capped[n] * scale
        return projected

    def _simulate_local(self, problem: OptimizationProblem, n: int, fixed_grid_buy, dual, consensus):
        f = problem.forecasts
        c = problem.constraints
        p = problem.parameters
        w = problem.objective_weights
        horizon = problem.horizon
        dt = p.get("step_minutes", 15) / 60.0
        soc_b = c["initial_battery_soc"][n]
        soc_ev = c["initial_ev_soc"][n]
        batt_cap = c["battery_energy_cap"][n]
        batt_power = c["battery_power_cap"][n]
        ev_cap = c["ev_energy_cap"][n]
        ev_power = c["ev_power_cap"][n]
        export_cap = c["export_cap"][n]
        eta_bc = p.get("battery_charge_eff", 0.95)
        eta_bd = p.get("battery_discharge_eff", 0.95)
        eta_ec = p.get("ev_charge_eff", 0.93)
        eta_ed = p.get("ev_discharge_eff", 0.93)
        risk_budget = c.get("risk_budget", {}).get(n, [0.5] * horizon)

        local = {
            "battery_charge": [0.0] * horizon,
            "battery_discharge": [0.0] * horizon,
            "ev_charge": [0.0] * horizon,
            "ev_discharge": [0.0] * horizon,
            "grid_buy": [0.0] * horizon,
            "grid_sell": [0.0] * horizon,
            "load_shed": [0.0] * horizon,
            "wind_curtail": [0.0] * horizon,
            "solar_curtail": [0.0] * horizon,
            "battery_soc": [0.0] * horizon,
            "ev_soc": [0.0] * horizon,
            "line_loading": [0.0] * horizon,
            "voltage_profile": [1.0] * horizon,
            "objective": 0.0,
        }
        desired_buy = [0.0] * horizon

        avg_price = sum(f["buy_price"][n]) / max(horizon, 1)
        for t in range(horizon):
            load = f["load"][n][t]
            pv = f["pv"][n][t]
            wind = f["wind"][n][t]
            trip = f["trip_energy"][n][t]
            buy_price = f["buy_price"][n][t]
            sell_price = f["sell_price"][n][t]
            reserve_floor = max(c["terminal_soc"][n][t], c["reserve_requirement"][n][t])
            ev_floor = c["ev_energy_requirement"][n][t]
            risk = risk_budget[t] if t < len(risk_budget) else 0.5
            net = load - pv - wind

            price_bias = (buy_price - avg_price) + 0.18 * dual[t] + 0.06 * (consensus[t] if t < len(consensus) else 0.0)
            discharge_pref = min(0.95, max(0.15, 0.48 + 0.22 * math.tanh(price_bias) + 0.14 * risk))
            charge_pref = min(0.9, max(0.15, 0.52 - 0.2 * math.tanh(price_bias) + 0.1 * (1.0 - risk)))

            batt_available = max(0.0, soc_b - reserve_floor)
            ev_available = max(0.0, soc_ev - ev_floor)

            if net > 0:
                batt_dis = min(batt_power, batt_available / max(dt / eta_bd, 1e-6), net * discharge_pref)
                soc_b = max(reserve_floor, soc_b - batt_dis * dt / eta_bd)
                net -= batt_dis
                ev_dis = min(ev_power, ev_available / max(dt / eta_ed, 1e-6), max(0.0, net) * (0.35 + 0.25 * risk))
                soc_ev = max(ev_floor, soc_ev - ev_dis * dt / eta_ed)
                net -= ev_dis
                desired_buy[t] = max(0.0, net)
                local["battery_discharge"][t] = batt_dis
                local["ev_discharge"][t] = ev_dis
            else:
                surplus = -net
                batt_need = max(0.0, reserve_floor - soc_b) + 0.25 * batt_cap
                batt_ch = min(
                    batt_power,
                    max(0.0, (batt_cap - soc_b) / max(dt * eta_bc, 1e-6)),
                    surplus * max(0.25, min(0.95, charge_pref + 0.18 * batt_need / max(batt_cap, 1e-6))),
                )
                soc_b = min(batt_cap, soc_b + batt_ch * dt * eta_bc)
                surplus -= batt_ch

                ev_need = max(0.0, ev_floor + 0.12 * ev_cap - soc_ev)
                ev_ch = min(
                    ev_power,
                    max(0.0, (ev_cap - soc_ev) / max(dt * eta_ec, 1e-6)),
                    surplus * max(0.18, min(0.8, 0.25 + 0.3 * ev_need / max(ev_cap, 1e-6))),
                )
                soc_ev = min(ev_cap, soc_ev + ev_ch * dt * eta_ec)
                surplus -= ev_ch

                local["battery_charge"][t] = batt_ch
                local["ev_charge"][t] = ev_ch
                sell = min(export_cap, max(0.0, surplus) * (0.65 + 0.15 * (1.0 - risk)))
                local["grid_sell"][t] = sell
                surplus -= sell
                local["wind_curtail"][t] = max(0.0, surplus * (wind / max(pv + wind, 1e-6)))
                local["solar_curtail"][t] = max(0.0, surplus - local["wind_curtail"][t])
                desired_buy[t] = 0.0

            soc_ev = max(0.0, soc_ev - trip)
            if soc_ev < ev_floor:
                local["load_shed"][t] += ev_floor - soc_ev
                soc_ev = ev_floor
            if fixed_grid_buy is not None:
                allowed_buy = max(0.0, min(fixed_grid_buy[t], c["grid_exchange_cap"][n][t]))
                local["grid_buy"][t] = allowed_buy
                unmet = max(0.0, desired_buy[t] - allowed_buy)
                if unmet > 0.0:
                    extra_batt = min(batt_power - local["battery_discharge"][t], max(0.0, soc_b - reserve_floor) / max(dt / eta_bd, 1e-6), unmet)
                    if extra_batt > 0:
                        local["battery_discharge"][t] += extra_batt
                        soc_b = max(reserve_floor, soc_b - extra_batt * dt / eta_bd)
                        unmet -= extra_batt
                    extra_ev = min(ev_power - local["ev_discharge"][t], max(0.0, soc_ev - ev_floor) / max(dt / eta_ed, 1e-6), unmet)
                    if extra_ev > 0:
                        local["ev_discharge"][t] += extra_ev
                        soc_ev = max(ev_floor, soc_ev - extra_ev * dt / eta_ed)
                        unmet -= extra_ev
                local["load_shed"][t] = max(0.0, unmet)

            local["battery_soc"][t] = soc_b
            local["ev_soc"][t] = soc_ev
            total_exchange = local["grid_buy"][t] + local["grid_sell"][t] + 0.35 * (
                local["battery_charge"][t] + local["battery_discharge"][t]
            )
            line_loading = total_exchange / max(c["grid_exchange_cap"][n][t] + export_cap, 1.0)
            voltage = 1.01 - 0.045 * line_loading + 0.012 * (pv + wind) / max(load + 0.5, 1.0) - 0.015 * (local["load_shed"][t] > 0)
            local["line_loading"][t] = line_loading
            local["voltage_profile"][t] = max(0.92, min(1.08, voltage))

            local["objective"] += (
                buy_price * local["grid_buy"][t] * dt
                - sell_price * local["grid_sell"][t] * dt
                + w["curtailment_penalty"] * (local["wind_curtail"][t] + local["solar_curtail"][t]) * dt
                + w["load_shedding_penalty"] * local["load_shed"][t] * dt
                + w["battery_usage_penalty"] * (local["battery_charge"][t] + local["battery_discharge"][t]) * dt
                + w["v2g_usage_penalty"] * (local["ev_charge"][t] + local["ev_discharge"][t]) * dt
            )

        return desired_buy, local
