from __future__ import annotations

from typing import Dict, Any

from src.models.types import LongTermPlan


class PyomoLongTermPlanner:
    """Formal long-term capacity planning model using Pyomo when available.

    Includes:
    - storage energy / power sizing
    - V2G participation sizing
    - import / export cap sizing
    - budget constraint
    - adequacy / curtailment proxy constraints
    """

    def __init__(self, config: Dict[str, Any], solver_name: str = 'glpk'):
        self.cfg = config
        self.solver_name = solver_name

    def solve(self, num_pdns: int):
        try:
            import pyomo.environ as pyo  # type: ignore
        except Exception as exc:
            return None, {'status': 'backend_unavailable', 'error': str(exc)}

        lt = self.cfg['long_term']
        costs = self.cfg['costs']
        N = range(num_pdns)

        model = pyo.ConcreteModel()
        model.N = pyo.Set(initialize=list(N))
        e_lo, e_hi = lt['storage_energy_bounds']
        p_lo, p_hi = lt['storage_power_bounds']
        k_lo, k_hi = lt['v2g_participation_bounds']
        buy_lo, buy_hi = lt['pdn_import_cap_bounds']
        sell_lo, sell_hi = lt['pdn_export_cap_bounds']

        model.E = pyo.Var(model.N, bounds=(e_lo, e_hi))
        model.P = pyo.Var(model.N, bounds=(p_lo, p_hi))
        model.K = pyo.Var(model.N, bounds=(k_lo, k_hi))
        model.BuyCap = pyo.Var(model.N, bounds=(buy_lo, buy_hi))
        model.SellCap = pyo.Var(model.N, bounds=(sell_lo, sell_hi))
        model.CurtRatio = pyo.Var(bounds=(0.0, lt['renewable_curtailment_ratio_limit']))
        model.Inadequacy = pyo.Var(bounds=(0.0, lt['supply_inadequacy_limit']))

        def obj_rule(m):
            return sum(
                costs['capital_battery_energy'] * m.E[n]
                + costs['capital_battery_power'] * m.P[n]
                + costs['capital_v2g_participation'] * m.K[n]
                for n in m.N
            ) + 50.0 * m.CurtRatio + 100.0 * m.Inadequacy
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        model.cons = pyo.ConstraintList()
        model.cons.add(sum(
            costs['capital_battery_energy'] * model.E[n]
            + costs['capital_battery_power'] * model.P[n]
            + costs['capital_v2g_participation'] * model.K[n]
            for n in model.N
        ) <= lt['budget_total'])

        # Proxy adequacy / curtailment relationships to avoid purely free slack variables.
        model.cons.add(sum(model.BuyCap[n] for n in model.N) + sum(model.P[n] for n in model.N) >= num_pdns * 6.0 - model.Inadequacy)
        model.cons.add(sum(model.SellCap[n] for n in model.N) + sum(model.E[n] for n in model.N) >= num_pdns * 4.0 * (1.0 - model.CurtRatio))
        for n in model.N:
            model.cons.add(model.P[n] <= model.E[n])
            model.cons.add(model.SellCap[n] <= model.BuyCap[n] + 2.0)

        solver = pyo.SolverFactory(self.solver_name)
        if solver is None or not solver.available(False):
            return None, {'status': 'solver_unavailable', 'solver': self.solver_name}
        result = solver.solve(model, tee=False)
        term = str(getattr(result.solver, 'termination_condition', 'unknown'))

        plan = LongTermPlan(
            battery_energy={n: float(pyo.value(model.E[n])) for n in N},
            battery_power={n: float(pyo.value(model.P[n])) for n in N},
            v2g_ratio={n: float(pyo.value(model.K[n])) for n in N},
            import_cap={n: float(pyo.value(model.BuyCap[n])) for n in N},
            export_cap={n: float(pyo.value(model.SellCap[n])) for n in N},
        )
        summary = {
            'status': 'optimal' if 'optimal' in term.lower() else 'feasible',
            'termination_condition': term,
            'objective_value': float(pyo.value(model.obj)),
            'curtailment_ratio_proxy': float(pyo.value(model.CurtRatio)),
            'supply_inadequacy_proxy': float(pyo.value(model.Inadequacy)),
        }
        return plan, summary
