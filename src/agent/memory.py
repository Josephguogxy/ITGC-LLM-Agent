from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class StateContextRecord:
    cycle_id: int
    runtime_mode: str
    operator_message: str
    pdn_context: Dict[int, Dict[str, Any]]
    event_tags: List[str] = field(default_factory=list)
    average_risk: float = 0.0


@dataclass
class PolicyGoalMemory:
    weight_vector: Dict[str, float]
    risk_bias: float = 0.0
    reserve_bias: float = 0.0
    acceptance_threshold: float = 1.0
    preferred_posture: str = "balanced"
    last_feedback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AcceptedCaseRecord:
    cycle_id: int
    context_tags: List[str]
    plan_summary: Dict[str, Any]
    bridge_summary: Dict[str, Any]
    feedback: Dict[str, Any]
    workflow_summary: Dict[str, Any]


@dataclass
class FailureRollbackRecord:
    cycle_id: int
    context_tags: List[str]
    failure_tags: List[str]
    rollback_reason: str
    repair_actions: List[str]
    feedback: Dict[str, Any]
    workflow_summary: Dict[str, Any]


@dataclass
class WorkflowCheckpointRecord:
    cycle_id: int
    checkpoints: List[str]
    rollback_policy: str
    warm_start: Dict[str, Any]
    solver_trace: Dict[str, Any]
    accepted: bool


class AgentMemorySystem:
    """Memory-centered closed loop for iterative agent improvement.

    The memory layer is organized into:
    - state/context memory
    - policy/goal memory
    - accepted case memory
    - failure/rollback memory
    - workflow/checkpoint memory
    """

    def __init__(self, config: Dict[str, Any], initial_weights: Dict[str, float]):
        memory_cfg = config.get("memory", {})
        self.max_context = int(memory_cfg.get("max_state_context", 32))
        self.max_success = int(memory_cfg.get("max_accepted_cases", 32))
        self.max_failure = int(memory_cfg.get("max_failure_cases", 32))
        self.max_checkpoint = int(memory_cfg.get("max_checkpoints", 32))
        self.storage_path = self._resolve_storage_path(memory_cfg.get("storage_path", "memory_store/agent_memory.json"))
        self.cycle_index = 0
        self.accepted_cycles = 0
        self.rejected_cycles = 0
        self.state_context_memory: List[StateContextRecord] = []
        self.policy_memory = PolicyGoalMemory(weight_vector=dict(initial_weights))
        self.accepted_case_memory: List[AcceptedCaseRecord] = []
        self.failure_memory: List[FailureRollbackRecord] = []
        self.workflow_memory: List[WorkflowCheckpointRecord] = []
        self._load_from_disk()
        self._save_to_disk()

    def begin_cycle(self) -> int:
        self.cycle_index += 1
        self._save_to_disk()
        return self.cycle_index

    def observe_state_context(
        self,
        cycle_id: int,
        dse_outputs: Dict[int, Any],
        operator_message: str,
        runtime_mode: str,
    ) -> StateContextRecord:
        pdn_context = {}
        event_tags: List[str] = []
        risk_values = []
        for pdn_id, output in dse_outputs.items():
            ctx = output.context
            pdn_context[pdn_id] = {
                "risk_label": ctx.get("risk_label", "unknown"),
                "anomalies": list(ctx.get("anomalies", [])),
                "volatility": float(ctx.get("volatility", 0.0)),
                "mobility_pressure": float(ctx.get("mobility_pressure", 0.0)),
                "renewable_share": float(ctx.get("renewable_share", 0.0)),
            }
            risk_values.append(self._risk_to_float(ctx.get("risk_label", "unknown")))
            event_tags.append(f"risk:{ctx.get('risk_label', 'unknown')}")
            event_tags.extend(f"anomaly:{item}" for item in ctx.get("anomalies", []))
            if ctx.get("volatility", 0.0) > 3.0:
                event_tags.append("stress:volatile_net_load")
            if ctx.get("mobility_pressure", 0.0) > 0.35:
                event_tags.append("stress:mobility")
            if ctx.get("renewable_share", 0.0) > 0.7:
                event_tags.append("stress:renewable_heavy")
        record = StateContextRecord(
            cycle_id=cycle_id,
            runtime_mode=runtime_mode,
            operator_message=operator_message,
            pdn_context=pdn_context,
            event_tags=sorted(set(event_tags)),
            average_risk=sum(risk_values) / max(len(risk_values), 1),
        )
        self.state_context_memory.append(record)
        self.state_context_memory = self.state_context_memory[-self.max_context :]
        self._save_to_disk()
        return record

    def retrieve_context(self) -> Dict[str, Any]:
        latest = self.state_context_memory[-1] if self.state_context_memory else None
        latest_tags = latest.event_tags if latest else []
        similar_successes = self._select_similar(latest_tags, self.accepted_case_memory)
        similar_failures = self._select_similar(latest_tags, self.failure_memory)
        checkpoint = self.workflow_memory[-1] if self.workflow_memory else None
        failure_pressure = min(0.18, 0.04 * len(similar_failures))
        success_credit = min(0.12, 0.03 * len(similar_successes))
        return {
            "cycle_id": self.cycle_index,
            "latest_context": {
                "event_tags": latest_tags,
                "average_risk": 0.0 if latest is None else latest.average_risk,
                "pdn_context": {} if latest is None else latest.pdn_context,
            },
            "policy": {
                "weight_vector": dict(self.policy_memory.weight_vector),
                "risk_bias": self.policy_memory.risk_bias,
                "reserve_bias": self.policy_memory.reserve_bias,
                "acceptance_threshold": self.policy_memory.acceptance_threshold,
                "preferred_posture": self.policy_memory.preferred_posture,
            },
            "similar_successes": [self._case_summary(item) for item in similar_successes],
            "similar_failures": [self._failure_summary(item) for item in similar_failures],
            "failure_pressure": failure_pressure,
            "success_credit": success_credit,
            "checkpoint": {
                "checkpoints": [] if checkpoint is None else checkpoint.checkpoints,
                "rollback_policy": "default" if checkpoint is None else checkpoint.rollback_policy,
                "warm_start": {} if checkpoint is None else checkpoint.warm_start,
                "warm_start_available": bool(checkpoint and checkpoint.warm_start),
                "solver_trace": {} if checkpoint is None else checkpoint.solver_trace,
            },
        }

    def update_policy_after_feedback(
        self,
        weight_vector: Dict[str, float],
        feedback: Dict[str, Any],
        accepted: bool,
        verification_failures: int,
    ) -> None:
        self.policy_memory.weight_vector = dict(weight_vector)
        self.policy_memory.last_feedback = dict(feedback)
        if accepted:
            self.policy_memory.risk_bias = max(0.0, self.policy_memory.risk_bias - 0.03)
            self.policy_memory.reserve_bias = max(0.0, self.policy_memory.reserve_bias - 0.02)
            self.policy_memory.acceptance_threshold = min(1.05, self.policy_memory.acceptance_threshold + 0.01)
            self.policy_memory.preferred_posture = "balanced"
            self.accepted_cycles += 1
        else:
            self.policy_memory.risk_bias = min(0.35, self.policy_memory.risk_bias + 0.08 + 0.02 * verification_failures)
            self.policy_memory.reserve_bias = min(0.25, self.policy_memory.reserve_bias + 0.05)
            self.policy_memory.acceptance_threshold = max(0.85, self.policy_memory.acceptance_threshold - 0.03)
            self.policy_memory.preferred_posture = "conservative"
            self.rejected_cycles += 1
        self._save_to_disk()

    def record_accepted_case(
        self,
        cycle_id: int,
        context_tags: List[str],
        plan_summary: Dict[str, Any],
        bridge_summary: Dict[str, Any],
        feedback: Dict[str, Any],
        workflow_summary: Dict[str, Any],
    ) -> None:
        self.accepted_case_memory.append(
            AcceptedCaseRecord(
                cycle_id=cycle_id,
                context_tags=list(context_tags),
                plan_summary=plan_summary,
                bridge_summary=bridge_summary,
                feedback=feedback,
                workflow_summary=workflow_summary,
            )
        )
        self.accepted_case_memory = self.accepted_case_memory[-self.max_success :]
        self._save_to_disk()

    def record_failure_case(
        self,
        cycle_id: int,
        context_tags: List[str],
        failure_tags: List[str],
        rollback_reason: str,
        repair_actions: List[str],
        feedback: Dict[str, Any],
        workflow_summary: Dict[str, Any],
    ) -> None:
        self.failure_memory.append(
            FailureRollbackRecord(
                cycle_id=cycle_id,
                context_tags=list(context_tags),
                failure_tags=list(failure_tags),
                rollback_reason=rollback_reason,
                repair_actions=list(repair_actions),
                feedback=feedback,
                workflow_summary=workflow_summary,
            )
        )
        self.failure_memory = self.failure_memory[-self.max_failure :]
        self._save_to_disk()

    def record_workflow_checkpoint(
        self,
        cycle_id: int,
        checkpoints: List[str],
        rollback_policy: str,
        warm_start: Dict[str, Any],
        solver_trace: Dict[str, Any],
        accepted: bool,
    ) -> None:
        self.workflow_memory.append(
            WorkflowCheckpointRecord(
                cycle_id=cycle_id,
                checkpoints=list(checkpoints),
                rollback_policy=rollback_policy,
                warm_start=warm_start,
                solver_trace=solver_trace,
                accepted=accepted,
            )
        )
        self.workflow_memory = self.workflow_memory[-self.max_checkpoint :]
        self._save_to_disk()

    @staticmethod
    def _risk_to_float(label: str) -> float:
        return {
            "low": 0.1,
            "normal": 0.35,
            "elevated": 0.7,
            "critical": 1.0,
        }.get(label, 0.25)

    @staticmethod
    def _tag_overlap(lhs: List[str], rhs: List[str]) -> int:
        return len(set(lhs).intersection(rhs))

    def _select_similar(self, tags: List[str], records: List[Any]) -> List[Any]:
        ranked = sorted(records, key=lambda item: (self._tag_overlap(tags, item.context_tags), item.cycle_id), reverse=True)
        return [item for item in ranked if self._tag_overlap(tags, item.context_tags) > 0][:3]

    @staticmethod
    def _case_summary(item: AcceptedCaseRecord) -> Dict[str, Any]:
        return {
            "cycle_id": item.cycle_id,
            "context_tags": item.context_tags,
            "feedback": item.feedback,
            "workflow_summary": item.workflow_summary,
        }

    @staticmethod
    def _failure_summary(item: FailureRollbackRecord) -> Dict[str, Any]:
        return {
            "cycle_id": item.cycle_id,
            "context_tags": item.context_tags,
            "failure_tags": item.failure_tags,
            "rollback_reason": item.rollback_reason,
            "repair_actions": item.repair_actions,
        }

    @staticmethod
    def _resolve_storage_path(path_value: str) -> Path:
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / candidate

    def _load_from_disk(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.cycle_index = int(payload.get("cycle_index", self.cycle_index))
        self.accepted_cycles = int(payload.get("accepted_cycles", self.accepted_cycles))
        self.rejected_cycles = int(payload.get("rejected_cycles", self.rejected_cycles))
        policy_payload = payload.get("policy_memory")
        if isinstance(policy_payload, dict):
            self.policy_memory = PolicyGoalMemory(**policy_payload)
        self.state_context_memory = [StateContextRecord(**item) for item in payload.get("state_context_memory", [])]
        self.accepted_case_memory = [AcceptedCaseRecord(**item) for item in payload.get("accepted_case_memory", [])]
        self.failure_memory = [FailureRollbackRecord(**item) for item in payload.get("failure_memory", [])]
        self.workflow_memory = [WorkflowCheckpointRecord(**item) for item in payload.get("workflow_memory", [])]

    def _save_to_disk(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cycle_index": self.cycle_index,
            "accepted_cycles": self.accepted_cycles,
            "rejected_cycles": self.rejected_cycles,
            "policy_memory": asdict(self.policy_memory),
            "state_context_memory": [asdict(item) for item in self.state_context_memory],
            "accepted_case_memory": [asdict(item) for item in self.accepted_case_memory],
            "failure_memory": [asdict(item) for item in self.failure_memory],
            "workflow_memory": [asdict(item) for item in self.workflow_memory],
        }
        temp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.storage_path)
