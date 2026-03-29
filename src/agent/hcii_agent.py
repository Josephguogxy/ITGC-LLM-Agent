from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.agent.prompts import SYSTEM_PROMPT, HAII_PROMPT_TEMPLATE
from src.llm import LLMService


@dataclass
class HAIIResult:
    approved: bool
    notes: str
    interpreted_intent: list[str] | None = None
    requested_changes: list[str] | None = None


class HAIIAgent:
    """Human-AI Interaction Interface Agent."""

    def __init__(self, config: Dict[str, Any], llm_service: LLMService | None = None):
        self.cfg = config
        self.llm = llm_service

    def review(self, verification_result, operator_message: str = 'Maintain reliable and economical operation.', dispatch_summary: Dict[str, Any] | None = None):
        if not self.cfg['agent']['human_gate_enabled'] and self.llm is None:
            return HAIIResult(approved=True, notes='human gate disabled -> auto approved', interpreted_intent=['default safe economical operation'], requested_changes=[])
        if self.llm is None:
            return HAIIResult(approved=verification_result.accepted, notes='LLM unavailable; mirrored verification result', interpreted_intent=['fallback review'], requested_changes=[])
        prompt = HAII_PROMPT_TEMPLATE.format(
            operator_message=operator_message,
            verification_summary=verification_result,
            dispatch_summary=dispatch_summary or {},
            policy_context={'human_gate_enabled': self.cfg['agent']['human_gate_enabled']},
        )
        parsed = self.llm.call(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, agent_role='haii').parsed
        return HAIIResult(
            approved=bool(parsed.get('approved', verification_result.accepted)),
            notes=parsed.get('comments', ''),
            interpreted_intent=parsed.get('interpreted_intent', []),
            requested_changes=parsed.get('requested_changes', []),
        )
