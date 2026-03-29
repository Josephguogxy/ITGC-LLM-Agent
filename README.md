# ITGC Agent System

This folder is a curated GitHub-ready export of the paper's agent system. It keeps the strategy, coordination, optimization, and LLM-integration layers, while intentionally excluding experiment runners, plotting utilities, generated figures, and result files.

## What Is Included

- `src/agent/`: the multi-agent stack and the main closed-loop framework
- `src/models/`: shared planning, bridge, and decision data structures
- `src/optimization/`: long-term planning, short-term dispatch, and backend abstractions
- `src/llm/`: pluggable mock, OpenAI, and Anthropic interfaces
- `src/data/`: only the input contracts used by the core system
- `configs/`: the minimal configuration files needed to understand the system setup
- `docs/`: a static GitHub Pages site that explains the architecture and code map

## What Is Intentionally Excluded

- Experiment entrypoints such as `run_experiments.py`
- Demos and runtime runners such as `main_agent_demo.py` and `run_streaming_demo.py`
- Plotting and report generation code
- Generated outputs, figures, notes, and paper-specific experiment packages

## Repository Layout

```text
itgc-agent-system/
+-- configs/
|   +-- base.yaml
|   +-- llm_config.yaml
|   +-- model_params.yaml
|   `-- runtime.yaml
+-- docs/
|   +-- index.html
|   +-- script.js
|   `-- styles.css
+-- src/
|   +-- agent/
|   +-- data/
|   +-- llm/
|   +-- models/
|   +-- optimization/
|   +-- metrics.py
|   `-- __init__.py
+-- ARCHITECTURE.md
+-- requirements.txt
`-- .gitignore
```

## How To Read The Code

1. Start with `src/agent/framework.py` to see the full cycle: ITGC -> Orchestrator -> DSE -> Optimization -> Verification -> HCII -> Execution.
2. Read `src/models/types.py` to understand the long-term plan, bridge variables, short-term decision, and operational state contracts.
3. Move to `src/models/long_term.py` and `src/optimization/long_term/benders_planner.py` for the long-horizon planning layer.
4. Then read `src/models/short_term.py`, `src/optimization/builder.py`, and `src/optimization/short_term/admm_dispatch.py` for the short-horizon dispatch logic.
5. Use `src/llm/` and `src/agent/prompts.py` to see how semantic reasoning is attached to solver-centered decision making.

## Configuration Notes

- `configs/llm_config.yaml` controls the LLM provider mode.
- `configs/model_params.yaml` contains long-term, short-term, cost, and agent parameters.
- `configs/runtime.yaml` stores rolling-window and trace-related runtime settings that the framework reads.

## Curated Release Notes

- The long-term Benders planner in this export requires an explicit `scenario_bundle`. The synthetic scenario generation helpers were intentionally left out to keep the repository focused on the agent architecture.
- This export is designed for code reading, repository publication, and architecture explanation. It is not the full experiment repository.

## GitHub Pages

https://josephguogxy.github.io/ITGC-LLM-Agent/

The resulting site will present the architecture, module map, and curated repository scope.
