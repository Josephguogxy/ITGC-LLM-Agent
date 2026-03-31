from __future__ import annotations

from typing import Dict, Any

from src.agent.itgc_agent import ITGCAgent
from src.agent.orchestrator_agent import OrchestratorAgent
from src.agent.dse_agent import DSEAgent
from src.agent.optimization_agent import OptimizationAgent
from src.agent.verification_agent import VerificationAgent
from src.agent.hcii_agent import HCIIAgent
from src.agent.memory import AgentMemorySystem
from src.agent.execution_agent import ExecutionAgent
from src.models.types import BridgeVariables, CycleFeedback, OperationalState, ShortTermDecision
from src.models.short_term import ShortTermDispatcher
from src.llm import LLMService
from src.metrics import Metrics


class ITGCAgentSystem:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.llm = LLMService(config.get('llm', {'provider': 'mock', 'mode': 'mock'}))
        self.itgc = ITGCAgent(config, self.llm)
        self.orchestrator = OrchestratorAgent(config, self.llm)
        self.dse = DSEAgent(config)
        self.optimizer = OptimizationAgent(config, self.llm)
        self.verifier = VerificationAgent(config, self.llm)
        self.hcii = HCIIAgent(config, self.llm)
        self.executor = ExecutionAgent(config)
        self.dispatcher = ShortTermDispatcher(config)
        self.memory = AgentMemorySystem(config, initial_weights=self.itgc.state.weight_vector)

    def run_cycle(
        self,
        day_data,
        operator_message: str = 'Maintain reliable and economical operation.',
        runtime_mode: str = 'batch',
        initial_states=None,
        planning_bundle=None,
        plan_override=None,
        plan_obj_override=None,
    ) -> dict:
        cycle_id = self.memory.begin_cycle()
        num_pdns = len(day_data)
        horizon = len(next(iter(day_data.values())))
        planning_memory = self.memory.retrieve_context()
        if plan_override is None:
            plan, plan_obj = self.itgc.plan(num_pdns, scenario_bundle=planning_bundle, memory_context=planning_memory)
        else:
            plan = plan_override
            plan_obj = plan_obj_override
        states = initial_states or self._initial_states(plan, num_pdns)
        local_states = {n: self.dse.update(n, series, current_state=states[n]) for n, series in day_data.items()}
        self.memory.observe_state_context(
            cycle_id=cycle_id,
            dse_outputs=local_states,
            operator_message=operator_message,
            runtime_mode=runtime_mode,
        )
        memory_context = self.memory.retrieve_context()
        self.orchestrator.log_task('long_term_planning', {'num_pdns': num_pdns})
        orchestration_plan = self.orchestrator.build_plan(
            {'plan_objective': getattr(plan_obj, 'total_cost', None)},
            num_pdns,
            horizon,
            runtime_mode=runtime_mode,
            memory_context=memory_context,
        )
        bridge = self._build_bridge(plan, states, local_states, horizon, memory_context=memory_context)
        self.orchestrator.log_task('state_estimation', {'pdns': list(local_states.keys())})

        decision = self.optimizer.solve(
            day_data,
            plan,
            bridge,
            initial_state=states,
            dse_outputs=local_states,
            memory_context=memory_context,
        )
        verification = self.verifier.verify(decision, bridge, day_data=day_data, memory_context=memory_context)
        review = self.hcii.review(
            verification,
            operator_message=operator_message,
            dispatch_summary={'grid_buy': decision.grid_buy, 'grid_sell': decision.grid_sell},
            memory_context=memory_context,
        )
        execution = self.executor.execute(
            decision if verification.accepted and review.approved else self._hold_decision(num_pdns, horizon, states)
        )
        feedback = self.evaluate_cycle(day_data, decision, verification.accepted and review.approved, execution.executed)
        feedback_record = self._feedback_to_dict(feedback)
        workflow_summary = self._workflow_summary(orchestration_plan)
        solver_result = self.optimizer.last_solver_result
        warm_start = {'grid_buy': decision.grid_buy} if feedback.accepted else {}
        solver_trace = {} if solver_result is None else dict(solver_result.diagnostics)
        self.memory.record_workflow_checkpoint(
            cycle_id=cycle_id,
            checkpoints=workflow_summary['checkpoints'],
            rollback_policy=workflow_summary['rollback_policy'],
            warm_start=warm_start,
            solver_trace=solver_trace,
            accepted=feedback.accepted,
        )
        if feedback.accepted:
            self.memory.record_accepted_case(
                cycle_id=cycle_id,
                context_tags=memory_context.get('latest_context', {}).get('event_tags', []),
                plan_summary=self._plan_summary(plan, plan_obj),
                bridge_summary=self._bridge_summary(bridge),
                feedback=feedback_record,
                workflow_summary=workflow_summary,
            )
        else:
            self.memory.record_failure_case(
                cycle_id=cycle_id,
                context_tags=memory_context.get('latest_context', {}).get('event_tags', []),
                failure_tags=list(verification.failed_constraints),
                rollback_reason='verification_or_review_rejection',
                repair_actions=list(review.requested_changes or []),
                feedback=feedback_record,
                workflow_summary=workflow_summary,
            )
        self.itgc.update_after_cycle(feedback_record)
        self.memory.update_policy_after_feedback(
            self.itgc.state.weight_vector,
            feedback_record,
            accepted=feedback.accepted,
            verification_failures=len(verification.failed_constraints),
        )
        return {
            'cycle_id': cycle_id,
            'plan': plan,
            'plan_objective': plan_obj,
            'orchestration_plan': orchestration_plan,
            'bridge': bridge,
            'local_states': local_states,
            'decision': decision,
            'verification': verification,
            'review': review,
            'execution': execution,
            'feedback': feedback,
            'orchestrator_trace': self.orchestrator.trace,
            'memory_context': memory_context,
            'optimization_notes': self.optimizer.last_semantic_notes,
            'solver_result': solver_result,
            'states_after': self._apply_first_step(states, decision, feedback.accepted),
        }

    def run_day(
        self,
        day_data,
        rolling_window_min: int,
        operator_message: str = 'Maintain reliable and economical operation.',
        planning_bundle=None,
        runtime_mode: str = 'rolling',
        plan_override=None,
        plan_obj_override=None,
        initial_states=None,
    ):
        num_pdns = len(day_data)
        horizon = len(next(iter(day_data.values())))
        dt = self.cfg.get('runtime', {}).get('step_minutes', 15) / 60.0
        if plan_override is None:
            plan, plan_obj = self.itgc.plan(num_pdns, scenario_bundle=planning_bundle, memory_context=self.memory.retrieve_context())
        else:
            plan, plan_obj = plan_override, plan_obj_override
        states = initial_states or self._initial_states(plan, num_pdns)
        full_decision = self._empty_decision(num_pdns, horizon)
        rollback_count = 0
        accepted_cycles = 0
        verification_failures = 0
        total_admm_iterations = 0.0
        total_benders_iterations = getattr(plan_obj, 'benders_iterations', 0.0)
        total_renewable = 0.0
        used_renewable = 0.0

        step_minutes = self.cfg.get('runtime', {}).get('step_minutes', 15)
        window_steps = max(1, int(round(rolling_window_min / max(step_minutes, 1))))
        rejection_streak = 0

        for start in range(horizon):
            end = min(horizon, start + window_steps)
            window_data = {n: series[start:end] for n, series in day_data.items()}
            result = self.run_cycle(
                window_data,
                operator_message=operator_message,
                runtime_mode=runtime_mode,
                initial_states=states,
                planning_bundle=None,
                plan_override=plan,
                plan_obj_override=plan_obj,
            )
            decision = result['decision']
            verification = result['verification']
            execution = result['execution']
            total_admm_iterations += float(result['solver_result'].diagnostics.get('iterations', 0.0))
            verification_failures += len(verification.failed_constraints)
            accepted = verification.accepted and execution.executed
            if accepted:
                accepted_cycles += 1
                rejection_streak = 0
            else:
                rollback_count += 1
                rejection_streak += 1
                self.orchestrator.trigger_rollback(
                    'verification_or_execution_failure',
                    {'step': start, 'constraints': verification.failed_constraints},
                )
                if rejection_streak >= 2:
                    plan, plan_obj = self.itgc.plan(
                        num_pdns,
                        scenario_bundle=planning_bundle,
                        memory_context=self.memory.retrieve_context(),
                    )
                    total_benders_iterations += getattr(plan_obj, 'benders_iterations', 0.0)
                    rejection_streak = 0

            for n in range(num_pdns):
                full_decision.battery_charge[n][start] = decision.battery_charge[n][0]
                full_decision.battery_discharge[n][start] = decision.battery_discharge[n][0]
                full_decision.ev_charge[n][start] = decision.ev_charge[n][0]
                full_decision.ev_discharge[n][start] = decision.ev_discharge[n][0]
                full_decision.wind_curtail[n][start] = decision.wind_curtail[n][0]
                full_decision.solar_curtail[n][start] = decision.solar_curtail[n][0]
                full_decision.grid_buy[n][start] = decision.grid_buy[n][0]
                full_decision.grid_sell[n][start] = decision.grid_sell[n][0]
                full_decision.load_shed[n][start] = decision.load_shed[n][0]
                full_decision.battery_soc[n][start] = decision.battery_soc[n][0]
                full_decision.ev_soc[n][start] = decision.ev_soc[n][0]
                full_decision.line_loading[n][start] = decision.line_loading[n][0]
                full_decision.voltage_profile[n][start] = decision.voltage_profile[n][0]
                total_renewable += day_data[n][start].pv + day_data[n][start].wind
                used_renewable += max(
                    0.0,
                    day_data[n][start].pv + day_data[n][start].wind
                    - decision.wind_curtail[n][0]
                    - decision.solar_curtail[n][0]
                )

            states = result['states_after']
            self.itgc.update_after_cycle(
                {
                    'supply_inadequacy': sum(decision.load_shed[n][0] for n in range(num_pdns)) * dt,
                    'curtailed_renewable': sum(
                        decision.wind_curtail[n][0] + decision.solar_curtail[n][0] for n in range(num_pdns)
                    ) * dt,
                    'v2g_intensity': sum(
                        decision.ev_charge[n][0] + decision.ev_discharge[n][0] for n in range(num_pdns)
                    ) * dt,
                    'renewable_utilization': used_renewable / max(total_renewable, 1e-6),
                }
            )

        metrics = Metrics()
        for n, series in day_data.items():
            for h, t in enumerate(series):
                metrics.import_cost += full_decision.grid_buy[n][h] * dt * t.buy_price
                metrics.export_revenue += full_decision.grid_sell[n][h] * dt * t.sell_price
                metrics.curtailment += (full_decision.wind_curtail[n][h] + full_decision.solar_curtail[n][h]) * dt
                metrics.load_shedding += full_decision.load_shed[n][h] * dt
                metrics.peak_import = max(metrics.peak_import, full_decision.grid_buy[n][h])
                metrics.battery_throughput += (full_decision.battery_charge[n][h] + full_decision.battery_discharge[n][h]) * dt
                metrics.ev_throughput += (full_decision.ev_charge[n][h] + full_decision.ev_discharge[n][h]) * dt
                metrics.ev_mobility_shortage += max(0.0, 0.22 - full_decision.ev_soc[n][h]) * dt

        metrics.rollback_count = rollback_count
        metrics.accepted_cycles = accepted_cycles
        metrics.total_cycles = horizon
        metrics.verification_failures = verification_failures
        metrics.admm_iterations = total_admm_iterations
        metrics.benders_iterations = total_benders_iterations
        metrics.renewable_utilization = used_renewable / max(total_renewable, 1e-6)
        metrics.total_cost = (
            metrics.import_cost
            - metrics.export_revenue
            + self.cfg['costs']['curtailment_penalty'] * metrics.curtailment
            + self.cfg['costs']['load_shedding_penalty'] * metrics.load_shedding
            + self.cfg['costs']['mobility_shortage_penalty'] * metrics.ev_mobility_shortage
            + self.cfg['costs']['battery_usage_penalty'] * metrics.battery_throughput
            + self.cfg['costs']['v2g_usage_penalty'] * metrics.ev_throughput
        )
        return {
            'plan': plan,
            'plan_objective': plan_obj,
            'decision': full_decision,
            'metrics': metrics,
            'final_states': states,
        }

    def run_week(self, week_days, rolling_window_min: int, planning_bundle=None, operator_message: str | None = None):
        total = Metrics()
        plan = None
        plan_obj = None
        states = None
        for day in week_days:
            out = self.run_day(
                day,
                rolling_window_min=rolling_window_min,
                operator_message=operator_message or self.cfg['runtime']['operator_default_message'],
                planning_bundle=planning_bundle,
                plan_override=plan,
                plan_obj_override=plan_obj,
                initial_states=states,
            )
            plan = out['plan']
            plan_obj = out['plan_objective']
            states = out['final_states']
            m = out['metrics']
            total.total_cost += m.total_cost
            total.import_cost += m.import_cost
            total.export_revenue += m.export_revenue
            total.curtailment += m.curtailment
            total.load_shedding += m.load_shedding
            total.peak_import = max(total.peak_import, m.peak_import)
            total.battery_throughput += m.battery_throughput
            total.ev_throughput += m.ev_throughput
            total.ev_mobility_shortage += m.ev_mobility_shortage
            total.rollback_count += m.rollback_count
            total.accepted_cycles += m.accepted_cycles
            total.total_cycles += m.total_cycles
            total.verification_failures += m.verification_failures
            total.admm_iterations += m.admm_iterations
            total.benders_iterations += m.benders_iterations
            total.renewable_utilization += m.renewable_utilization
        if week_days:
            total.renewable_utilization /= len(week_days)
        return total

    def evaluate_cycle(self, day_data, decision, verified: bool, executed: bool) -> CycleFeedback:
        dt = self.cfg.get('runtime', {}).get('step_minutes', 15) / 60.0
        cost = curtail = shed = peak = v2g = renewable = used_renewable = 0.0
        for n, series in day_data.items():
            for h, t in enumerate(series):
                buy = decision.grid_buy[n][h]
                sell = decision.grid_sell[n][h]
                wc = decision.wind_curtail[n][h]
                sc = decision.solar_curtail[n][h]
                ls = decision.load_shed[n][h]
                cost += (buy * t.buy_price - sell * t.sell_price) * dt
                curtail += (wc + sc) * dt
                shed += ls * dt
                peak = max(peak, buy)
                v2g += (decision.ev_charge[n][h] + decision.ev_discharge[n][h]) * dt
                renewable += (t.pv + t.wind) * dt
                used_renewable += max(0.0, (t.pv + t.wind - wc - sc) * dt)
        accepted_limit = self.cfg['long_term']['supply_inadequacy_limit'] * self.memory.policy_memory.acceptance_threshold
        accepted = verified and executed and shed <= accepted_limit
        rollback = not accepted
        return CycleFeedback(
            operating_cost=cost,
            curtailed_renewable=curtail,
            supply_inadequacy=shed,
            peak_import=peak,
            v2g_intensity=v2g,
            accepted=accepted,
            rollback_triggered=rollback,
            renewable_utilization=used_renewable / max(renewable, 1e-6),
            verification_failures=0 if verified else 1,
            admm_iterations=float(decision.metadata.get('iterations', 0.0)),
            benders_iterations=0.0,
        )

    def _initial_states(self, plan, num_pdns: int):
        return {
            n: OperationalState(
                battery_soc=0.58 * plan.battery_energy[n],
                ev_soc=max(0.9, 0.68 * max(2.0, plan.v2g_ratio[n] * 4.0)),
                risk_budget=self.cfg['agent']['default_risk_budget'],
            )
            for n in range(num_pdns)
        }

    def _build_bridge(self, plan, states, dse_outputs, horizon: int, memory_context: Dict[str, Any] | None = None) -> BridgeVariables:
        terminal_soc = {}
        reserve_requirement = {}
        grid_exchange_cap = {}
        ev_energy_requirement = {}
        risk_budget = {}
        system_cap = self.cfg['short_term']['system_import_cap_default']
        policy = (memory_context or {}).get('policy', {})
        failure_pressure = (memory_context or {}).get('failure_pressure', 0.0)
        success_credit = (memory_context or {}).get('success_credit', 0.0)
        risk_bias = policy.get('risk_bias', 0.0)
        reserve_bias = policy.get('reserve_bias', 0.0)

        for n, state in states.items():
            net_forecast = dse_outputs[n].context['net_load_forecast']
            volatility = dse_outputs[n].context['volatility']
            mobility_pressure = dse_outputs[n].context['mobility_pressure']
            risk_label = dse_outputs[n].context['risk_label']
            risk_shift = 0.14 if risk_label in ('elevated', 'critical') else 0.05 if risk_label == 'normal' else 0.0
            terminal_soc[n] = []
            reserve_requirement[n] = []
            grid_exchange_cap[n] = []
            ev_energy_requirement[n] = []
            risk_budget[n] = []
            for h in range(horizon):
                predicted_peak = max(0.0, net_forecast[h]) if h < len(net_forecast) else 0.0
                risk = min(1.0, max(0.0, state.risk_budget + risk_shift + 0.03 * volatility + risk_bias + failure_pressure - 0.5 * success_credit))
                terminal_floor = 0.16 * plan.battery_energy[n] + 0.08 * predicted_peak + 0.04 * reserve_bias * plan.battery_energy[n]
                terminal_soc[n].append(max(0.18 * plan.battery_energy[n], terminal_floor))
                reserve_requirement[n].append(max(0.45, 0.35 + 0.08 * predicted_peak + 0.18 * risk + reserve_bias))
                envelope = min(plan.import_cap[n], 0.9 * plan.import_cap[n] + 0.05 * system_cap - 0.03 * volatility - failure_pressure + 0.4 * success_credit)
                grid_exchange_cap[n].append(max(1.0, envelope))
                ev_energy_requirement[n].append(min(max(0.35, 0.25 + 0.55 * mobility_pressure + 0.25 * risk), max(2.0, plan.v2g_ratio[n] * 4.0)))
                risk_budget[n].append(risk)
        return BridgeVariables(
            terminal_soc=terminal_soc,
            reserve_requirement=reserve_requirement,
            grid_exchange_cap=grid_exchange_cap,
            ev_energy_requirement=ev_energy_requirement,
            risk_budget=risk_budget,
        )

    def _apply_first_step(self, states, decision, accepted: bool):
        updated = {}
        for n, state in states.items():
            if accepted and decision.battery_soc.get(n):
                next_batt = decision.battery_soc[n][0]
                next_ev = decision.ev_soc[n][0]
            else:
                next_batt = state.battery_soc
                next_ev = state.ev_soc
            risk = min(
                1.0,
                max(
                    0.0,
                    state.risk_budget
                    + 0.06 * (decision.load_shed[n][0] if decision.load_shed.get(n) else 0.0)
                    + 0.02 * (
                        (decision.wind_curtail[n][0] if decision.wind_curtail.get(n) else 0.0)
                        + (decision.solar_curtail[n][0] if decision.solar_curtail.get(n) else 0.0)
                    )
                    - (0.03 if accepted else 0.0),
                ),
            )
            updated[n] = OperationalState(
                battery_soc=next_batt,
                ev_soc=next_ev,
                risk_budget=risk,
                last_voltage=decision.voltage_profile.get(n, [state.last_voltage])[0],
                last_line_loading=decision.line_loading.get(n, [state.last_line_loading])[0],
            )
        return updated

    @staticmethod
    def _feedback_to_dict(feedback: CycleFeedback) -> Dict[str, Any]:
        return {
            'operating_cost': feedback.operating_cost,
            'curtailed_renewable': feedback.curtailed_renewable,
            'supply_inadequacy': feedback.supply_inadequacy,
            'peak_import': feedback.peak_import,
            'v2g_intensity': feedback.v2g_intensity,
            'accepted': feedback.accepted,
            'rollback_triggered': feedback.rollback_triggered,
            'renewable_utilization': feedback.renewable_utilization,
            'verification_failures': feedback.verification_failures,
            'admm_iterations': feedback.admm_iterations,
            'benders_iterations': feedback.benders_iterations,
        }

    @staticmethod
    def _plan_summary(plan, plan_obj) -> Dict[str, Any]:
        return {
            'avg_battery_energy': sum(plan.battery_energy.values()) / max(len(plan.battery_energy), 1),
            'avg_battery_power': sum(plan.battery_power.values()) / max(len(plan.battery_power), 1),
            'avg_v2g_ratio': sum(plan.v2g_ratio.values()) / max(len(plan.v2g_ratio), 1),
            'avg_import_cap': sum(plan.import_cap.values()) / max(len(plan.import_cap), 1),
            'objective': getattr(plan_obj, 'total_cost', None),
            'benders_iterations': getattr(plan_obj, 'benders_iterations', 0),
            'cut_count': getattr(plan_obj, 'cut_count', 0),
        }

    @staticmethod
    def _bridge_summary(bridge: BridgeVariables) -> Dict[str, Any]:
        pdns = list(bridge.risk_budget.keys())
        if not pdns:
            return {}
        avg_risk = sum(sum(bridge.risk_budget[n]) / max(len(bridge.risk_budget[n]), 1) for n in pdns) / len(pdns)
        avg_exchange = sum(sum(bridge.grid_exchange_cap[n]) / max(len(bridge.grid_exchange_cap[n]), 1) for n in pdns) / len(pdns)
        avg_reserve = sum(sum(bridge.reserve_requirement[n]) / max(len(bridge.reserve_requirement[n]), 1) for n in pdns) / len(pdns)
        return {
            'avg_risk_budget': avg_risk,
            'avg_grid_exchange_cap': avg_exchange,
            'avg_reserve_requirement': avg_reserve,
        }

    @staticmethod
    def _workflow_summary(orchestration_plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'checkpoints': list(orchestration_plan.get('checkpoints', [])),
            'rollback_policy': orchestration_plan.get('rollback_policy', 'default_rollback_on_failure'),
            'parallel_tasks': list(orchestration_plan.get('parallel_tasks', [])),
            'sequential_tasks': list(orchestration_plan.get('sequential_tasks', [])),
            'warm_start_available': bool(orchestration_plan.get('warm_start_available', False)),
        }

    def _hold_decision(self, num_pdns: int, horizon: int, states):
        decision = self._empty_decision(num_pdns, horizon)
        for n in range(num_pdns):
            decision.battery_soc[n] = [states[n].battery_soc for _ in range(horizon)]
            decision.ev_soc[n] = [states[n].ev_soc for _ in range(horizon)]
            decision.line_loading[n] = [states[n].last_line_loading for _ in range(horizon)]
            decision.voltage_profile[n] = [states[n].last_voltage for _ in range(horizon)]
        return decision

    @staticmethod
    def _empty_decision(num_pdns: int, horizon: int):
        return ShortTermDecision(
            battery_charge={n: [0.0] * horizon for n in range(num_pdns)},
            battery_discharge={n: [0.0] * horizon for n in range(num_pdns)},
            ev_charge={n: [0.0] * horizon for n in range(num_pdns)},
            ev_discharge={n: [0.0] * horizon for n in range(num_pdns)},
            wind_curtail={n: [0.0] * horizon for n in range(num_pdns)},
            solar_curtail={n: [0.0] * horizon for n in range(num_pdns)},
            grid_buy={n: [0.0] * horizon for n in range(num_pdns)},
            grid_sell={n: [0.0] * horizon for n in range(num_pdns)},
            load_shed={n: [0.0] * horizon for n in range(num_pdns)},
            battery_soc={n: [0.0] * horizon for n in range(num_pdns)},
            ev_soc={n: [0.0] * horizon for n in range(num_pdns)},
            line_loading={n: [0.0] * horizon for n in range(num_pdns)},
            voltage_profile={n: [1.0] * horizon for n in range(num_pdns)},
            metadata={},
        )
