from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from src.models.types import LongTermPlan
from src.optimization.long_term.benders_planner import BendersLongTermPlanner
from src.optimization.long_term.pyomo_planner import PyomoLongTermPlanner


@dataclass
class LongTermObjectiveBreakdown:
    capital_cost: float
    expected_operating_cost: float
    total_cost: float
    benders_iterations: int = 0
    cut_count: int = 0


class LongTermPlanner:
    """Long-term planning model with formal Pyomo upgrade path."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def build_initial_plan(self, num_pdns: int) -> LongTermPlan:
        lt = self.cfg['long_term']
        e_lo, e_hi = lt['storage_energy_bounds']
        p_lo, p_hi = lt['storage_power_bounds']
        k_lo, k_hi = lt['v2g_participation_bounds']
        buy_lo, buy_hi = lt['pdn_import_cap_bounds']
        sell_lo, sell_hi = lt['pdn_export_cap_bounds']
        return LongTermPlan(
            battery_energy={n: (e_lo + e_hi) / 2 for n in range(num_pdns)},
            battery_power={n: (p_lo + p_hi) / 2 for n in range(num_pdns)},
            v2g_ratio={n: (k_lo + k_hi) / 2 for n in range(num_pdns)},
            import_cap={n: (buy_lo + buy_hi) / 2 for n in range(num_pdns)},
            export_cap={n: (sell_lo + sell_hi) / 2 for n in range(num_pdns)},
        )

    def capital_cost(self, plan: LongTermPlan) -> float:
        c = self.cfg['costs']
        total = 0.0
        for n in plan.battery_energy:
            total += c['capital_battery_energy'] * plan.battery_energy[n]
            total += c['capital_battery_power'] * plan.battery_power[n]
            total += c['capital_v2g_participation'] * plan.v2g_ratio[n]
        return total

    def enforce_budget(self, plan: LongTermPlan) -> LongTermPlan:
        budget = self.cfg['long_term'].get('budget_total_per_pdn', 0.0) * max(len(plan.battery_energy), 1)
        if budget <= 0:
            budget = self.cfg['long_term']['budget_total']
        cap = self.capital_cost(plan)
        if cap <= budget or cap == 0:
            return plan
        scale = budget / cap
        for d in (plan.battery_energy, plan.battery_power, plan.v2g_ratio, plan.import_cap, plan.export_cap):
            for k in d:
                d[k] *= scale
        return plan

    def solve(
        self,
        num_pdns: int,
        scenario_bundle=None,
        weight_vector: Dict[str, float] | None = None,
    ) -> tuple[LongTermPlan, LongTermObjectiveBreakdown]:
        solver_cfg = self.cfg.get('solver', {})
        preferred = solver_cfg.get('preferred_long_term', solver_cfg.get('preferred', 'benders'))

        if preferred == 'pyomo':
            planner = PyomoLongTermPlanner(self.cfg, solver_name=solver_cfg.get('pyomo_solver', 'glpk'))
            plan, summary = planner.solve(num_pdns)
            if plan is not None:
                total = summary.get('objective_value', 0.0)
                cap = self.capital_cost(plan)
                op = max(0.0, total - cap)
                return plan, LongTermObjectiveBreakdown(
                    capital_cost=cap,
                    expected_operating_cost=op,
                    total_cost=total,
                )

        planner = BendersLongTermPlanner(self.cfg, max_iter=solver_cfg.get('benders_max_iter', 7))
        plan, summary = planner.solve(num_pdns, scenario_bundle=scenario_bundle, weight_vector=weight_vector)
        cap = self.capital_cost(plan)
        op = max(0.0, summary.get('objective_value', cap) - cap)
        return plan, LongTermObjectiveBreakdown(
            capital_cost=cap,
            expected_operating_cost=op,
            total_cost=summary.get('objective_value', cap + op),
            benders_iterations=summary.get('iterations', 0),
            cut_count=summary.get('cut_count', 0),
        )
