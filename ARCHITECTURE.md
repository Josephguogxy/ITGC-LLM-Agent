# Architecture Guide

This repository contains the paper's agent system stripped down to the modules that explain the core logic.

## Closed-Loop Flow

The main orchestration entrypoint is `src/agent/framework.py`.

1. `ITGCAgent` updates long-term priorities and produces a strategic resource plan.
2. `OrchestratorAgent` records task traces and builds the cycle-level coordination intent.
3. `DSEAgent` prepares local estimated states and forecasts for each PDN.
4. `OptimizationAgent` builds a short-term optimization problem and delegates numerical solving to the selected backend.
5. `VerificationAgent` checks whether the dispatch is safe with respect to bridge and network constraints.
6. `HCIIAgent` injects operator-facing review logic through the LLM layer.
7. `ExecutionAgent` applies the accepted decision and feeds the next operational state back to the framework.

Across this loop, `src/agent/memory.py` continuously writes and retrieves five memory types: state/context memory, policy/goal memory, accepted case memory, failure/rollback memory, and workflow/checkpoint memory. The memory layer is file-backed through `memory_store/agent_memory.json`, so experience persists across runs instead of disappearing at process exit. This makes the curated repository reflect memory-driven iterative evolution instead of a one-shot workflow.

## Core Data Contracts

The system is readable because the cross-layer data contracts are explicit.

- `LongTermPlan`: long-horizon investment and capacity decisions
- `BridgeVariables`: the contract between long-term planning and short-term dispatch
- `ShortTermDecision`: the dispatch result returned by the optimization layer
- `OperationalState`: the rolling state carried across cycles
- `CycleFeedback`: the feedback signal used to update the long-term controller

These types live in `src/models/types.py`.

## Module Responsibilities

### `src/agent/`

- Houses the agent roles defined by the paper.
- Keeps the control flow readable through small single-purpose classes.
- Uses prompts only for semantic interpretation; numerical decisions remain solver-centered.
- Contains the memory-centered evolution layer that writes back successful cases, failure tags, checkpoint traces, and policy updates.

### `src/models/`

- Encodes cross-layer objects and lightweight wrappers around the planning and dispatch logic.
- Keeps the formal interfaces separate from the agent wrappers.

### `src/optimization/`

- Defines optimization problem schemas and solver backends.
- Contains the multicut Benders-style long-term planner and the distributed ADMM dispatcher.
- Makes it possible to compare heuristic and formal backends behind one interface.

### `src/llm/`

- Provides provider-agnostic request and response contracts.
- Supports `mock`, `openai`, and `anthropic` modes.
- Keeps the LLM layer optional so the rest of the system remains understandable without remote services.

### `src/data/`

- Only keeps the input contracts needed by the core system.
- The experiment-specific generators and runners were removed from this export.

## What Was Removed

To keep the public repository focused and easy to navigate, the following were excluded on purpose:

- Experiment entrypoints
- Streaming demo scripts
- Plotting utilities
- Report generation code
- Output folders and generated figures
- Paper-specific experiment redesign packages

## Reading Order

If someone opens the repository for the first time, this reading order works best:

1. `src/agent/framework.py`
2. `src/models/types.py`
3. `src/agent/itgc_agent.py`
4. `src/models/long_term.py`
5. `src/optimization/long_term/benders_planner.py`
6. `src/agent/optimization_agent.py`
7. `src/optimization/builder.py`
8. `src/optimization/short_term/admm_dispatch.py`
9. `src/agent/verification_agent.py`
10. `src/agent/hcii_agent.py`

## Important Limitation In This Curated Export

The Benders planner no longer auto-builds synthetic planning scenarios. In this public code export, callers must pass an explicit `scenario_bundle` into `BendersLongTermPlanner.solve()` or `LongTermPlanner.solve()`. This keeps the repository centered on architecture instead of experiment scaffolding.
