from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List

from src.agent.prompts import SYSTEM_PROMPT, ORCHESTRATOR_PROMPT_TEMPLATE
from src.llm import LLMService


@dataclass
class OrchestratorTrace:
    task_log: List[dict] = field(default_factory=list)
    rollback_events: List[dict] = field(default_factory=list)
    llm_summaries: List[dict] = field(default_factory=list)


class OrchestratorAgent:
    """Workflow orchestrator with optional LLM planning assistance."""

    def __init__(self, config: Dict[str, Any], llm_service: LLMService | None = None):
        self.cfg = config
        self.llm = llm_service
        self.trace = OrchestratorTrace()

    def build_plan(
        self,
        planning_summary: Dict[str, Any],
        pdn_count: int,
        horizon: int,
        runtime_mode: str = 'batch',
        memory_context: Dict[str, Any] | None = None,
    ):
        checkpoint = (memory_context or {}).get('checkpoint', {})
        retrieved_checkpoints = list(checkpoint.get('checkpoints', []))
        if self.llm is None:
            result = {
                'summary': 'LLM disabled, using default workflow plan.',
                'parallel_tasks': ['dse_update'],
                'sequential_tasks': ['planning', 'optimization', 'verification', 'execution'],
                'checkpoints': list(dict.fromkeys(['post_planning', 'post_verification', *retrieved_checkpoints])),
                'rollback_policy': 'default_rollback_on_failure',
                'warm_start_available': bool(checkpoint.get('warm_start_available', False)),
                'bottlenecks': [],
                'assumptions': ['deterministic default workflow'],
            }
        else:
            prompt = ORCHESTRATOR_PROMPT_TEMPLATE.format(
                planning_summary=planning_summary,
                pdn_count=pdn_count,
                horizon=horizon,
                verification_enabled=self.cfg['agent']['verification_enabled'],
                human_gate_enabled=self.cfg['agent']['human_gate_enabled'],
                runtime_mode=runtime_mode,
                memory_summary={
                    'recent_failures': (memory_context or {}).get('similar_failures', []),
                    'recent_successes': (memory_context or {}).get('similar_successes', []),
                    'checkpoint_memory': checkpoint,
                },
            )
            result = self.llm.call(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, agent_role='orchestrator').parsed
        result.setdefault('checkpoints', list(dict.fromkeys(['post_planning', 'post_verification', *retrieved_checkpoints])))
        result.setdefault('warm_start_available', bool(checkpoint.get('warm_start_available', False)))
        self.trace.llm_summaries.append(result)
        return result

    def log_task(self, name: str, payload: Dict[str, Any]):
        self.trace.task_log.append({'name': name, 'payload': payload})

    def trigger_rollback(self, reason: str, context: Dict[str, Any]):
        self.trace.rollback_events.append({'reason': reason, 'context': context})
        return {'rollback': True, 'reason': reason}
