# AGENTS.md – Overview of Agents in the AMT Auto‑Research Workflow

This repository follows the **Karpathy Auto‑Research** methodology, which relies on a small set of well‑defined agents that interact through **Hermes Kanban** (when used) or directly via function calls.  The agents are deliberately lightweight so they can be reproduced in a Colab notebook, a local environment, or a production server.

---

## 1. Agent Roles
| Role | Responsibility | Typical Implementation |
|------|----------------|------------------------|
| **Orchestrator** | Creates the high‑level research **Epic** (e.g. *"Implement AMT with Virgin Levels"*), decomposes it into atomic units, and monitors progress. | In this repo the orchestrator is a **human‑in‑the‑loop** (you) who creates the Kanban Epic or runs the notebook. When Kanban is enabled, the built‑in `kanban_decomposer` performs automatic decomposition.
| **Researcher** | Executes the **Audit → Hypothesis → Implementation → Verification** cycle for a single atomic unit. | Implemented as the **backtest runner** (`src/backtest.py`) and the **strategy generator** (`src/strategy.py`). The researcher may call an LLM via the `use_api` flag to generate signals.
| **Validator** | Checks that the backtest output meets success criteria (Sharpe, max‑DD, trade count, etc.) and writes the result back to the board or notebook. | In the notebook the validation step is the `print(json.dumps(result["metrics"], …))` block. In a Kanban setup you would use `kanban_complete` with a structured result.
| **Logger** | Persists trade‑log CSV and any auxiliary artefacts (model checkpoints, plots). | `LOGS_DIR` (`logs/`) is created by `run_backtest`; the CSV is written there. |
| **Auditor** | Analyzes trade logs to identify failure modes and propose new hypotheses. | `src/audit.py` reads `trade_log.csv` and uses an LLM to generate `audit_report.md`. |

---

## 2. Interaction Pattern
```
Orchestrator (human) ──► creates Epic on Kanban (or runs notebook)
          │
          ▼
Researcher (backtest runner) ──► generates signals (local or via LLM)
          │
          ▼
Validator (same process) ──► computes metrics, writes logs
          │
          ▼
Result is displayed to the user (notebook output) or posted back to Kanban via `kanban_complete`.
```

When the **Kanban toolset** is active, the flow becomes fully asynchronous:
- The orchestrator creates a task (`kanban_create`).
- The dispatcher spawns a worker that runs `run_backtest`.
- The worker reports progress with `kanban_heartbeat` and finishes with `kanban_complete`.
- The human checks the board or the notebook for the final metrics.

---

## 3. Extending the Agent Set
If you want to add more specialized agents (e.g. a **Feature Engineer** that pre‑processes the parquet data, or a **Model Trainer** that fits an XGBoost model), follow these guidelines:
1. **Create a new Python module** under `src/` (e.g. `feature_engineer.py`).
2. **Expose a single callable** that accepts a DataFrame and returns the transformed DataFrame.
3. **Wrap it as a Kanban worker** by adding a small wrapper script in `src/` that calls the function and then uses `kanban_complete` to return a JSON payload.
4. **Document the agent** in this `AGENTS.md` file under a new section.

---

## 4. Running in Google Colab
In Colab the agents are *implicit* – the notebook itself acts as the orchestrator, the backtest function is the researcher/validator, and the notebook cells serve as the logger.  No explicit Kanban calls are required, but you can still enable them by installing the Hermes CLI inside the notebook and using the `/kanban` slash commands.

---

## 5. Quick Reference
- **Orchestrator** – human / Kanban Epic creation (`kanban_create`).
- **Researcher** – `src/backtest.py` (`run_backtest`).
- **Strategy Generator** – `src/strategy.py` (`generate_signal`).
- **Validator** – metric calculation inside `run_backtest` (`compute_metrics`).
- **Logger** – CSV written to `logs/trade_log.csv`.

---

*Keep this file up‑to‑date as you add new agents or change responsibilities.*
