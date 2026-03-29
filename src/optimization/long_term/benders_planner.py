from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

from src.models.types import LongTermPlan, PlanningScenario


class BendersLongTermPlanner:
    """Multicut Benders-style long-term planner runnable without external solvers.

    The master problem is maintained as a bundle of scenario-specific cuts.
    Each scenario subproblem is evaluated through a reduced-form operating model
    that returns both a recourse estimate and a cut gradient.
    """

    def __init__(self, config: Dict[str, Any], max_iter: int = 7):
        self.cfg = config
        self.max_iter = max_iter

    def solve(
        self,
        num_pdns: int,
        scenario_bundle: Iterable[PlanningScenario] | None = None,
        weight_vector: Dict[str, float] | None = None,
    ):
        scenarios = list(scenario_bundle or [])
        if not scenarios:
            scenarios = self._default_bundle(num_pdns)
        z = self._plan_to_vector(self._initial_plan(num_pdns))
        cuts: Dict[str, List[dict]] = {s.scenario_id: [] for s in scenarios}
        history: List[dict] = []
        best = None
        best_total = float("inf")

        for iteration in range(1, self.max_iter + 1):
            plan = self._vector_to_plan(z, num_pdns)
            cap_cost = self._capital_cost(plan)
            expected_cost = 0.0
            scenario_logs = []
            for scenario in scenarios:
                recourse, gradient, metrics = self._evaluate_scenario(plan, scenario, weight_vector)
                intercept = recourse - self._dot(gradient, z)
                cuts[scenario.scenario_id].append(
                    {
                        "intercept": intercept,
                        "gradient": gradient,
                        "recourse": recourse,
                        "probability": scenario.probability,
                    }
                )
                expected_cost += scenario.probability * recourse
                scenario_logs.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "probability": scenario.probability,
                        "recourse": recourse,
                        "metrics": metrics,
                    }
                )

            total = cap_cost + expected_cost
            history.append(
                {
                    "iteration": iteration,
                    "capital_cost": cap_cost,
                    "expected_operating_cost": expected_cost,
                    "objective": total,
                    "cuts": sum(len(v) for v in cuts.values()),
                    "scenarios": scenario_logs,
                }
            )
            if total < best_total:
                best_total = total
                best = {
                    "plan": deepcopy(plan),
                    "history": deepcopy(history),
                    "cuts": deepcopy(cuts),
                }

            gradient = self._capital_gradient(num_pdns)
            for scenario in scenarios:
                active = max(
                    cuts[scenario.scenario_id],
                    key=lambda c: c["intercept"] + self._dot(c["gradient"], z),
                )
                gradient = [g + scenario.probability * a for g, a in zip(gradient, active["gradient"])]

            step = 0.22 / (1.0 + 0.5 * (iteration - 1))
            candidate = [v - step * g for v, g in zip(z, gradient)]
            z = self._project_vector(candidate, num_pdns)

        if best is None:
            plan = self._vector_to_plan(z, num_pdns)
            best = {"plan": plan, "history": history, "cuts": cuts}
            best_total = self._capital_cost(plan)

        return best["plan"], {
            "status": "optimal_approx",
            "objective_value": best_total,
            "iterations": len(best["history"]),
            "cut_count": sum(len(v) for v in best["cuts"].values()),
            "history": best["history"],
        }

    def _default_bundle(self, num_pdns: int) -> List[PlanningScenario]:
        del num_pdns
        raise ValueError(
            "This curated release omits scenario-generation scripts. "
            "Provide an explicit scenario_bundle when calling "
            "BendersLongTermPlanner.solve()."
        )

    def _initial_plan(self, num_pdns: int) -> LongTermPlan:
        lt = self.cfg["long_term"]
        e_lo, e_hi = lt["storage_energy_bounds"]
        p_lo, p_hi = lt["storage_power_bounds"]
        k_lo, k_hi = lt["v2g_participation_bounds"]
        buy_lo, buy_hi = lt["pdn_import_cap_bounds"]
        sell_lo, sell_hi = lt["pdn_export_cap_bounds"]
        return LongTermPlan(
            battery_energy={n: 0.45 * e_lo + 0.55 * e_hi for n in range(num_pdns)},
            battery_power={n: 0.4 * p_lo + 0.6 * p_hi for n in range(num_pdns)},
            v2g_ratio={n: 0.4 * k_lo + 0.6 * k_hi for n in range(num_pdns)},
            import_cap={n: 0.4 * buy_lo + 0.6 * buy_hi for n in range(num_pdns)},
            export_cap={n: 0.35 * sell_lo + 0.65 * sell_hi for n in range(num_pdns)},
        )

    def _capital_cost(self, plan: LongTermPlan) -> float:
        costs = self.cfg["costs"]
        cap_import = costs.get("capital_import_cap", 0.25)
        cap_export = costs.get("capital_export_cap", 0.18)
        total = 0.0
        for n in plan.battery_energy:
            total += costs["capital_battery_energy"] * plan.battery_energy[n]
            total += costs["capital_battery_power"] * plan.battery_power[n]
            total += costs["capital_v2g_participation"] * plan.v2g_ratio[n]
            total += cap_import * plan.import_cap[n]
            total += cap_export * plan.export_cap[n]
        return total

    def _capital_gradient(self, num_pdns: int) -> List[float]:
        costs = self.cfg["costs"]
        return (
            [costs["capital_battery_energy"]] * num_pdns
            + [costs["capital_battery_power"]] * num_pdns
            + [costs["capital_v2g_participation"]] * num_pdns
            + [costs.get("capital_import_cap", 0.25)] * num_pdns
            + [costs.get("capital_export_cap", 0.18)] * num_pdns
        )

    def _evaluate_scenario(
        self,
        plan: LongTermPlan,
        scenario: PlanningScenario,
        weight_vector: Dict[str, float] | None,
    ):
        weights = weight_vector or {}
        w_rel = weights.get("reliability", 1.0)
        w_ren = weights.get("renewable", 1.0)
        w_usr = weights.get("user_service", 1.0)
        w_deg = weights.get("degradation", 1.0)
        costs = self.cfg["costs"]

        gradient = [0.0] * (5 * len(plan.battery_energy))
        total_cost = 0.0
        total_shortage = 0.0
        total_curtailment = 0.0
        total_import = 0.0
        total_renewable = 0.0
        total_used_renewable = 0.0

        for n, series in scenario.day_data.items():
            deficit_energy = 0.0
            peak_deficit = 0.0
            renewable_surplus = 0.0
            trip_energy = 0.0
            buy_price_weight = 0.0
            sell_price_weight = 0.0
            net_ramp = 0.0
            prev_net = None

            for step in series:
                renewable = step.pv + step.wind
                total_renewable += renewable
                total_used_renewable += min(step.load, renewable)
                net = step.load - renewable
                if net > 0:
                    deficit_energy += net
                    peak_deficit = max(peak_deficit, net)
                    buy_price_weight += net * step.buy_price
                else:
                    renewable_surplus += -net
                    sell_price_weight += (-net) * step.sell_price
                trip_energy += step.trip_energy
                if prev_net is not None:
                    net_ramp = max(net_ramp, abs(net - prev_net))
                prev_net = net

            avg_buy = buy_price_weight / max(deficit_energy, 1e-6)
            avg_sell = sell_price_weight / max(renewable_surplus, 1e-6)
            batt_energy = plan.battery_energy[n]
            batt_power = plan.battery_power[n]
            v2g = plan.v2g_ratio[n]
            import_cap = plan.import_cap[n]
            export_cap = plan.export_cap[n]

            storage_shift = min(0.72 * batt_energy, 5.0 * batt_power)
            power_support = batt_power + 1.35 * v2g
            mobility_support = 1.9 * v2g + 0.15 * batt_energy
            import_support = 5.5 * import_cap
            export_support = 4.5 * export_cap

            shortage = max(0.0, deficit_energy - import_support - 0.85 * storage_shift - 0.45 * mobility_support)
            peak_violation = max(0.0, peak_deficit - import_cap - power_support)
            mobility_shortage = max(0.0, 0.38 * trip_energy - mobility_support)
            curtailment = max(0.0, renewable_surplus - export_support - 0.7 * storage_shift)
            degradation = 0.16 * storage_shift + 0.12 * v2g * max(trip_energy, 1.0) + 0.05 * net_ramp
            import_energy = max(0.0, deficit_energy - 0.55 * storage_shift - 0.3 * mobility_support)
            export_energy = min(renewable_surplus - curtailment, export_support)

            total_shortage += shortage + peak_violation
            total_curtailment += curtailment
            total_import += import_energy

            local_cost = (
                avg_buy * import_energy
                - avg_sell * export_energy
                + costs["load_shedding_penalty"] * w_rel * (shortage + 1.15 * peak_violation)
                + costs["curtailment_penalty"] * w_ren * curtailment
                + costs["mobility_shortage_penalty"] * w_usr * mobility_shortage
                + (costs["battery_usage_penalty"] + costs["v2g_usage_penalty"]) * w_deg * degradation
            )
            total_cost += local_cost

            shortage_signal = shortage + 1.2 * peak_violation
            curtail_signal = curtailment
            mobility_signal = mobility_shortage
            underuse_signal = max(0.0, 0.18 * batt_energy - shortage_signal) + max(0.0, 0.1 * export_cap - curtail_signal)

            num_pdns = len(plan.battery_energy)
            gradient[n] += -0.22 * costs["load_shedding_penalty"] * w_rel * min(shortage_signal, 12.0)
            gradient[num_pdns + n] += -0.34 * costs["load_shedding_penalty"] * w_rel * min(peak_violation + 0.4 * shortage_signal, 10.0)
            gradient[2 * num_pdns + n] += -0.28 * costs["mobility_shortage_penalty"] * w_usr * min(mobility_signal + 0.15 * shortage_signal, 8.0)
            gradient[3 * num_pdns + n] += -0.38 * costs["load_shedding_penalty"] * w_rel * min(peak_violation + 0.2 * shortage_signal, 8.0)
            gradient[4 * num_pdns + n] += -0.24 * costs["curtailment_penalty"] * w_ren * min(curtail_signal, 10.0)

            if underuse_signal > 0.0 and shortage_signal < 1e-6 and curtail_signal < 1e-6:
                gradient[n] += 0.08 * underuse_signal
                gradient[num_pdns + n] += 0.06 * underuse_signal
                gradient[2 * num_pdns + n] += 0.04 * underuse_signal
                gradient[3 * num_pdns + n] += 0.05 * underuse_signal
                gradient[4 * num_pdns + n] += 0.03 * underuse_signal

        metrics = {
            "shortage": total_shortage,
            "curtailment": total_curtailment,
            "import_energy": total_import,
            "renewable_utilization": total_used_renewable / max(total_renewable, 1e-6),
        }
        return total_cost, gradient, metrics

    def _plan_to_vector(self, plan: LongTermPlan) -> List[float]:
        order = list(plan.battery_energy.keys())
        return (
            [plan.battery_energy[n] for n in order]
            + [plan.battery_power[n] for n in order]
            + [plan.v2g_ratio[n] for n in order]
            + [plan.import_cap[n] for n in order]
            + [plan.export_cap[n] for n in order]
        )

    def _vector_to_plan(self, vector: List[float], num_pdns: int) -> LongTermPlan:
        return LongTermPlan(
            battery_energy={n: vector[n] for n in range(num_pdns)},
            battery_power={n: vector[num_pdns + n] for n in range(num_pdns)},
            v2g_ratio={n: vector[2 * num_pdns + n] for n in range(num_pdns)},
            import_cap={n: vector[3 * num_pdns + n] for n in range(num_pdns)},
            export_cap={n: vector[4 * num_pdns + n] for n in range(num_pdns)},
        )

    def _project_vector(self, vector: List[float], num_pdns: int) -> List[float]:
        lt = self.cfg["long_term"]
        bounds = (
            [tuple(lt["storage_energy_bounds"])] * num_pdns
            + [tuple(lt["storage_power_bounds"])] * num_pdns
            + [tuple(lt["v2g_participation_bounds"])] * num_pdns
            + [tuple(lt["pdn_import_cap_bounds"])] * num_pdns
            + [tuple(lt["pdn_export_cap_bounds"])] * num_pdns
        )
        projected = [min(max(v, lo), hi) for v, (lo, hi) in zip(vector, bounds)]
        plan = self._vector_to_plan(projected, num_pdns)
        budget = self.cfg["long_term"]["budget_total"]
        storage_budget = 0.0
        costs = self.cfg["costs"]
        for n in range(num_pdns):
            storage_budget += (
                costs["capital_battery_energy"] * plan.battery_energy[n]
                + costs["capital_battery_power"] * plan.battery_power[n]
                + costs["capital_v2g_participation"] * plan.v2g_ratio[n]
            )
        budget = self._budget_total(num_pdns)
        if storage_budget <= budget:
            return projected

        scale = budget / max(storage_budget, 1e-6)
        for n in range(num_pdns):
            projected[n] *= scale
            projected[num_pdns + n] *= scale
            projected[2 * num_pdns + n] *= scale
        projected = [min(max(v, lo), hi) for v, (lo, hi) in zip(projected, bounds)]
        return projected

    def _budget_total(self, num_pdns: int) -> float:
        lt = self.cfg["long_term"]
        if "budget_total_per_pdn" in lt:
            return lt["budget_total_per_pdn"] * num_pdns
        return lt["budget_total"]

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
