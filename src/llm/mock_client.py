from __future__ import annotations

import json
import time
from typing import Any, Dict

from .base import BaseLLMClient
from .schemas import LLMRequest, LLMResponse


class MockLLMClient(BaseLLMClient):
    """Deterministic fallback client for offline experiments."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.time()
        role = request.metadata.get("agent_role", "generic")
        payload = self._mock_payload(role, request)
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        return LLMResponse(
            raw_text=raw,
            parsed=payload,
            model_name=self.config.get("model", "mock-llm"),
            provider="mock",
            latency_ms=(time.time() - start) * 1000,
            success=True,
        )

    def _mock_payload(self, role: str, request: LLMRequest) -> Dict[str, Any]:
        if role == "itgc":
            return {
                "summary": "Mock ITGC planning response",
                "policy_updates": {"reliability": 1.05, "renewable": 1.02},
                "risk_posture": "balanced",
                "recommendations": ["preserve reserve margin", "limit curtailment"],
            }
        if role == "orchestrator":
            return {
                "summary": "Mock orchestration plan",
                "parallel_tasks": ["dse_update", "local_dispatch_prepare"],
                "checkpoints": ["post_planning", "post_verification"],
                "rollback_policy": "rollback_on_constraint_violation_or_execution_failure",
            }
        if role == "haii":
            return {
                "approved": True,
                "comments": "Mock human review approved under nominal risk.",
                "requested_changes": [],
            }
        if role == "verification":
            return {
                "accepted": True,
                "failed_constraints": [],
                "risk_notes": ["mock verification used"],
            }
        return {
            "summary": f"Mock response for {role}",
            "notes": ["offline deterministic mode"],
        }
