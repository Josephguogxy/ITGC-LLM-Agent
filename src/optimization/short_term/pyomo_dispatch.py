from __future__ import annotations

from src.optimization.schemas import OptimizationProblem, OptimizationResult


class PyomoShortTermDispatcher:
    def __init__(self, solver_name: str = 'glpk'):
        self.solver_name = solver_name

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        try:
            import pyomo.environ as pyo  # type: ignore
        except Exception as exc:
            return OptimizationResult(status='backend_unavailable', objective_value=0.0, variables={}, diagnostics={'backend': 'pyomo_short_term', 'error': str(exc)})

        H = problem.horizon
        N = range(problem.num_pdns)
        T = range(H)
        f = problem.forecasts
        c = problem.constraints
        w = problem.objective_weights
        window_steps = c.get('effective_window_steps', H)

        model = pyo.ConcreteModel()
        model.N = pyo.Set(initialize=list(N))
        model.T = pyo.Set(initialize=list(T))
        model.bch = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.bdis = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.ech = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.edis = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.grid_buy = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.grid_sell = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.load_shed = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.wind_curt = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.solar_curt = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.soc_b = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
        model.soc_ev = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)

        eta_bc = problem.parameters.get('battery_charge_eff', 0.95)
        eta_bd = problem.parameters.get('battery_discharge_eff', 0.95)
        eta_ec = problem.parameters.get('ev_charge_eff', 0.93)
        eta_ed = problem.parameters.get('ev_discharge_eff', 0.93)
        battery_usage_penalty = w.get('battery_usage_penalty', 0.05)
        v2g_usage_penalty = w.get('v2g_usage_penalty', 0.05)
        load_shedding_penalty = w.get('load_shedding_penalty', 20.0)
        curtailment_penalty = w.get('curtailment_penalty', 2.0)

        def decay(t):
            return 1.0 if t < window_steps else 1.6 + 0.03 * (t - window_steps)

        def obj_rule(m):
            return sum(
                decay(t) * (
                    f['buy_price'][n][t] * m.grid_buy[n, t]
                    - f['sell_price'][n][t] * m.grid_sell[n, t]
                    + load_shedding_penalty * m.load_shed[n, t]
                    + curtailment_penalty * (m.wind_curt[n, t] + m.solar_curt[n, t])
                    + battery_usage_penalty * (m.bch[n, t] + m.bdis[n, t])
                    + v2g_usage_penalty * (m.ech[n, t] + m.edis[n, t])
                ) for n in m.N for t in m.T
            )
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        model.cons = pyo.ConstraintList()
        for n in N:
            for t in T:
                model.cons.add(
                    f['load'][n][t] + model.bch[n, t] + model.ech[n, t] - model.bdis[n, t] - model.edis[n, t]
                    - model.grid_buy[n, t] + model.grid_sell[n, t] - model.load_shed[n, t]
                    - model.wind_curt[n, t] - model.solar_curt[n, t] - f['pv'][n][t] - f['wind'][n][t] == 0
                )
                model.cons.add(model.bch[n, t] <= c['battery_power_cap'][n])
                model.cons.add(model.bdis[n, t] <= c['battery_power_cap'][n])
                model.cons.add(model.ech[n, t] <= c['ev_power_cap'][n])
                model.cons.add(model.edis[n, t] <= c['ev_power_cap'][n])
                model.cons.add(model.grid_buy[n, t] <= c['grid_exchange_cap'][n][t])
                model.cons.add(model.grid_sell[n, t] <= c['export_cap'][n])
                model.cons.add(model.wind_curt[n, t] <= f['wind'][n][t])
                model.cons.add(model.solar_curt[n, t] <= f['pv'][n][t])
                if t == 0:
                    model.cons.add(model.soc_b[n, t] == c['initial_battery_soc'][n] + eta_bc * model.bch[n, t] - (1 / eta_bd) * model.bdis[n, t])
                    model.cons.add(model.soc_ev[n, t] == c['initial_ev_soc'][n] + eta_ec * model.ech[n, t] - (1 / eta_ed) * model.edis[n, t] - f['trip_energy'][n][t])
                else:
                    model.cons.add(model.soc_b[n, t] == model.soc_b[n, t-1] + eta_bc * model.bch[n, t] - (1 / eta_bd) * model.bdis[n, t])
                    model.cons.add(model.soc_ev[n, t] == model.soc_ev[n, t-1] + eta_ec * model.ech[n, t] - (1 / eta_ed) * model.edis[n, t] - f['trip_energy'][n][t])
                model.cons.add(model.soc_b[n, t] <= c['battery_energy_cap'][n])
                model.cons.add(model.soc_ev[n, t] <= c['ev_energy_cap'][n])
                if t < window_steps:
                    model.cons.add(model.soc_b[n, t] >= c['terminal_soc'][n][t])
                    model.cons.add(model.soc_ev[n, t] >= c['ev_energy_requirement'][n][t])
                    model.cons.add(model.soc_b[n, t] >= c['reserve_requirement'][n][t])
                else:
                    model.cons.add(model.soc_b[n, t] >= 0.65 * c['terminal_soc'][n][t])
                    model.cons.add(model.soc_ev[n, t] >= 0.7 * c['ev_energy_requirement'][n][t])
                    model.cons.add(model.soc_b[n, t] >= 0.7 * c['reserve_requirement'][n][t])
        for t in T:
            model.cons.add(sum(model.grid_buy[n, t] for n in N) <= c['system_import_cap'])

        solver = pyo.SolverFactory(self.solver_name)
        if solver is None or not solver.available(False):
            return OptimizationResult(status='solver_unavailable', objective_value=0.0, variables={}, diagnostics={'backend': 'pyomo_short_term', 'solver': self.solver_name, 'window_steps': window_steps})
        result = solver.solve(model, tee=False)
        term = str(getattr(result.solver, 'termination_condition', 'unknown'))
        status = 'optimal' if 'optimal' in term.lower() else 'feasible'
        vars_out = {
            'battery_charge': {n: [pyo.value(model.bch[n, t]) for t in T] for n in N},
            'battery_discharge': {n: [pyo.value(model.bdis[n, t]) for t in T] for n in N},
            'ev_charge': {n: [pyo.value(model.ech[n, t]) for t in T] for n in N},
            'ev_discharge': {n: [pyo.value(model.edis[n, t]) for t in T] for n in N},
            'grid_buy': {n: [pyo.value(model.grid_buy[n, t]) for t in T] for n in N},
            'grid_sell': {n: [pyo.value(model.grid_sell[n, t]) for t in T] for n in N},
            'load_shed': {n: [pyo.value(model.load_shed[n, t]) for t in T] for n in N},
            'wind_curtail': {n: [pyo.value(model.wind_curt[n, t]) for t in T] for n in N},
            'solar_curtail': {n: [pyo.value(model.solar_curt[n, t]) for t in T] for n in N},
        }
        return OptimizationResult(status=status, objective_value=float(pyo.value(model.obj)), variables=vars_out, diagnostics={'backend': 'pyomo_short_term', 'solver': self.solver_name, 'termination_condition': term, 'window_steps': window_steps})
