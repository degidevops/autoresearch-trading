# AGENTS.md – Overview of Agents in the AMT Auto‑Research Workflow

This repository implements a **Full Autonomous ML Evolution** loop following the **Karpathy Auto‑Research** methodology.

---

## 1. Agent Roles
| Role | Responsibility | Typical Implementation |
|------|----------------|------------------------|
| **Orchestrator** | Automates the research cycle (Audit → Rewrite → Train → Verify → Commit). | `src/orchestrator.py` |
| **Researcher** | Performs backtesting, audit, and architecture design via LLM. | `src/backtest.py` (API/Hermes/Local) |
| **Architect** | Designs the ML pipeline (features, model choice, training). | `src/model_factory.py` (LLM-optimized) |
| **Validator** | Validates the strategy against quantitative performance metrics. | `src/backtest.py` (compute_metrics) |
| **Logger** | Records all experimental results for historical audit. | `logs/trade_log.csv` + `audit_report.md` |
| **Auditor** | Analyzes failure modes and reports to the Architect. | `src/audit.py` |

---

## 2. Supported LLM Providers
The `LLM_PROVIDER` environment variable controls **how the research/evolution agent is served**.
The project has **3 operational modes**:

| Mode | Value | Behavior |
|------|-------|----------|
| Local LLM | `local` | Uses `llama-cpp-python` against a local GGUF model. |
| External API LLM | `openai` or `anthropic` | Uses REST API with `LLM_API_KEY`. |
| Hermes Agent | Hermes as agent/driver | Run the orchestrator/workflow from inside Hermes. Do NOT set `LLM_PROVIDER=hermes` in code. |

Notes:
- `LLM_PROVIDER=auto` prefers `local`, then `openai`.
- When running under Hermes: Hermes is the agent, so execution is not `hermes_tools` importable inside `_call_llm`.
- `local` or API modes remain defaults for Colab/standalone execution.
- Hermes implements `src/model_factory.py` features and verifies via backtest; no module-side Hermes import is required.
- `LLM_PROVIDER=hermes` is intentionally not implemented inside `_call_llm`; use Hermes as the agent driver instead.

---

## 3. Workflow Pattern
1.  **Orchestrator** initializes the evolution loop.
2.  **Architect** trains/updates the ML model in `model_factory.py`.
3.  **Backtest engine** evaluates performance.
4.  **Auditor** analyzes failures, feeds data back to the **Architect** (LLM).
5.  **Commit** if Sharpe improves, else `git checkout` (rollback).
