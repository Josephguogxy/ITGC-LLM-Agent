from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from src.models.long_term import LongTermPlanner
from src.agent.prompts import SYSTEM_PROMPT, ITGC_PROMPT_TEMPLATE
from src.llm import LLMService


@dataclass
class ITGCState:
    weight_vector: Dict[str, float]
    knowledge_state: Dict[str, Any]


class ITGCAgent:
    """Infinite Time Grid Controller.

    Strategic layer: updates long-term policy and invokes LLM for semantic analysis.
    Numerical planning remains solver-centered via LongTermPlanner.
    """

    def __init__(self, config: Dict[str, Any], llm_service: LLMService | None = None):
        self.cfg = config
        self.planner = LongTermPlanner(config)
        self.llm = llm_service
        self.state = ITGCState(
            weight_vector={
                'economic': 1.0,
                'reliability': 1.0,
                'renewable': 1.0,
                'user_service': 1.0,
                'degradation': 0.5,
            },
            knowledge_state={},
        )

    def plan(self, num_pdns: int, scenario_bundle=None):
        plan, obj = self.planner.solve(num_pdns, scenario_bundle=scenario_bundle, weight_vector=self.state.weight_vector)
        analysis = self._semantic_analysis(plan)
        self.state.knowledge_state['last_plan_analysis'] = analysis
        return plan, obj

    def _semantic_analysis(self, plan):
        if self.llm is None:
            return {'summary': 'LLM disabled', 'recommendations': []}
        prompt = ITGC_PROMPT_TEMPLATE.format(
            plan=plan,
            feedback=self.state.knowledge_state.get('last_feedback', {}),
            weight_vector=self.state.weight_vector,
            config_summary={
                'budget_total': self.cfg['long_term']['budget_total'],
                'supply_limit': self.cfg['long_term']['supply_inadequacy_limit'],
                'curtailment_limit': self.cfg['long_term']['renewable_curtailment_ratio_limit'],
            },
        )
        resp = self.llm.call(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, agent_role='itgc')
        return resp.parsed

    def update_after_cycle(self, feedback: Dict[str, float]):
        self.state.knowledge_state['last_feedback'] = feedback
        if feedback.get('supply_inadequacy', 0.0) > 0:
            self.state.weight_vector['reliability'] += 0.12
        if feedback.get('curtailed_renewable', 0.0) > 0:
            self.state.weight_vector['renewable'] += 0.07
        if feedback.get('v2g_intensity', 0.0) > 4.0:
            self.state.weight_vector['degradation'] += 0.03
        if feedback.get('renewable_utilization', 1.0) < 0.75:
            self.state.weight_vector['renewable'] += 0.05
