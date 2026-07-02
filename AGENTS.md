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
The `LLM_PROVIDER` environment variable controls which engine drives the evolution:
- `hermes`: Uses internal Hermes agent tools (context-aware).
- `openai` / `anthropic`: Uses external API (stable).
- `local`: Uses GGUF model via `llama-cpp-python` (private, free).

---

## 3. Workflow Pattern
1.  **Orchestrator** initializes the evolution loop.
2.  **Architect** trains/updates the ML model in `model_factory.py`.
3.  **Backtest engine** evaluates performance.
4.  **Auditor** analyzes failures, feeds data back to the **Architect** (LLM).
5.  **Commit** if Sharpe improves, else `git checkout` (rollback).
