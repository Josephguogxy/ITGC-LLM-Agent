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
                "planning_directives": ["preserve reserve margin", "reuse successful bridge posture"],
                "bridge_preferences": ["tighten risk budget slightly under elevated volatility"],
                "memory_use": ["reused recent successful cases", "checked recent rollback tags"],
                "recommendations": ["preserve reserve margin", "limit curtailment"],
                "concerns": [],
                "assumptions": ["mock mode"],
            }
        if role == "orchestrator":
            return {
                "summary": "Mock orchestration plan",
                "parallel_tasks": ["dse_update", "local_dispatch_prepare"],
                "sequential_tasks": ["planning", "optimization", "verification", "hcii", "execution"],
                "checkpoints": ["post_planning", "post_verification"],
                "rollback_policy": "rollback_on_constraint_violation_or_execution_failure",
                "warm_start_plan": ["reuse last accepted checkpoint as ADMM initialization"],
                "bottlenecks": [],
                "memory_actions": ["read workflow checkpoint memory"],
                "assumptions": ["mock mode"],
            }
        if role == "optimization":
            return {
                "summary": "Mock optimization interpretation",
                "dominant_constraints": ["grid_exchange_cap", "ev_service", "reserve_requirement"],
                "dispatch_strategy": "balanced",
                "solver_guidance": ["reuse warm start when available"],
                "tradeoff_notes": ["prioritize reliability over marginal arbitrage"],
                "memory_reuse": ["checkpoint warm start available"],
                "assumptions": ["mock mode"],
            }
        if role == "hcii":
            return {
                "approved": True,
                "comments": "Mock human review approved under nominal risk.",
                "interpreted_intent": ["maintain safe and economical service"],
                "requested_changes": [],
                "safety_warnings": [],
                "coordination_advice": ["continue with validated first-step action"],
                "assumptions": ["mock mode"],
            }
        if role == "verification":
            return {
                "accepted": True,
                "failed_constraints": [],
                "risk_notes": ["mock verification used"],
                "revise_actions": [],
                "safety_rank": "nominal",
                "memory_hits": [],
                "assumptions": ["mock mode"],
            }
        if role == "execution":
            return {
                "action_summary": "Mock execution bundle prepared.",
                "execution_ready": True,
                "monitoring_focus": ["voltage", "grid import", "EV service"],
                "rollback_watchpoints": ["constraint violation", "execution failure"],
                "operator_notes": ["mock mode"],
            }
        return {
            "summary": f"Mock response for {role}",
            "notes": ["offline deterministic mode"],
        }
