from __future__ import annotations

from typing import Dict, Any

from src.optimization import OptimizationProblemBuilder
from src.optimization.backend_factory import build_solver_backend
from src.models.types import ShortTermDecision
from src.agent.prompts import SYSTEM_PROMPT, OPTIMIZATION_PROMPT_TEMPLATE
from src.llm import LLMService


class OptimizationAgent:
    """Optimization Agent.

    LLM handles semantic formulation/explanation.
    Solver backend handles numerical decision generation.
    """

    def __init__(self, config: Dict[str, Any], llm_service: LLMService | None = None):
        self.cfg = config
        self.llm = llm_service
        self.builder = OptimizationProblemBuilder(config)
        self.solver = build_solver_backend(config)
        self.last_semantic_notes = {}
        self.last_solver_result = None

    def solve(self, day_data, plan, bridge, initial_state=None, dse_outputs=None):
        problem = self.builder.build(day_data, plan, bridge, initial_state=initial_state, dse_outputs=dse_outputs)
        if self.llm is not None:
            prompt = OPTIMIZATION_PROMPT_TEMPLATE.format(
                problem_summary={
                    'horizon': problem.horizon,
                    'num_pdns': problem.num_pdns,
                    'constraint_keys': list(problem.constraints.keys()),
                },
                bridge_summary={
                    'reserve_requirement': bridge.reserve_requirement,
                    'grid_exchange_cap': bridge.grid_exchange_cap,
                    'risk_budget': bridge.risk_budget,
                },
                objective_weights=problem.objective_weights,
                solver_backend=getattr(self.solver, 'name', self.solver.__class__.__name__),
            )
            resp = self.llm.call(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, agent_role='optimization')
            self.last_semantic_notes = resp.parsed if resp.success else {'error': resp.error}
        result = self.solver.solve(problem)
        self.last_solver_result = result
        if result.status not in ('solved_heuristic', 'placeholder_solved', 'optimal', 'feasible'):
            # fallback to heuristic if a formal backend is selected but unavailable
            from src.optimization.heuristic_solver import HeuristicDispatchSolver
            fallback = HeuristicDispatchSolver().solve(problem)
            self.last_solver_result = fallback
            result = fallback
        return ShortTermDecision(
            battery_charge=result.variables['battery_charge'],
            battery_discharge=result.variables['battery_discharge'],
            ev_charge=result.variables['ev_charge'],
            ev_discharge=result.variables['ev_discharge'],
            wind_curtail=result.variables['wind_curtail'],
            solar_curtail=result.variables['solar_curtail'],
            grid_buy=result.variables['grid_buy'],
            grid_sell=result.variables['grid_sell'],
            load_shed=result.variables['load_shed'],
            battery_soc=result.variables.get('battery_soc', {}),
            ev_soc=result.variables.get('ev_soc', {}),
            line_loading=result.variables.get('line_loading', {}),
            voltage_profile=result.variables.get('voltage_profile', {}),
            metadata=result.diagnostics,
        )
