# 📋 PROJECT KANBAN: AMT Auto-Research ML (Source of Truth)

## 🛠 Status: `PHASE 1 - INFRASTRUCTURE & BASELINE`

| 📥 BACKLOG | ⏳ TODO | 🚀 IN PROGRESS | ✅ DONE |
| :--- | :--- | :--- | :--- |
| [ ] ML-01: Feature Engineering | [ ] AUD-01: Cluster Loss Analysis | [ ] BSL-01: Run Virgin-Level Backtest | [x] INF-01: Project Structure |
| [ ] ML-02: Model Training | [ ] AUD-02: TF Correlation Audit | | [x] INF-02: 12y Data Conversion |
| [ ] ML-03: Probability Filter | [ ] AUD-03: Root Cause ID | | [x] INF-03: Timezone Sync (UTC) |
| [ ] VER-01: Verification Run | [ ] HYP-01: Formal Hypothesis | | [x] INF-04: Triple-TF TPO Engine |
| [ ] VER-02: Result Comparison | [ ] HYP-02: Success Metric Def | | [x] INF-05: VWAP StdDev Bands |

---

## 🧬 ATOMIC-UNIT DEPENDENCIES (DAG)
`BSL-01` $\rightarrow$ (`AUD-01` & `AUD-02`) $\rightarrow$ `AUD-03` $\rightarrow$ (`HYP-01` & `HYP-02`) $\rightarrow$ `ML-01` $\rightarrow$ `VER-01` $\rightarrow$ `VER-02`

## 📝 CURRENT LOG
- **Current Focus:** BSL-01 (Running baseline backtest with Virgin Levels logic).
- **Last Update:** Fixed "Sustain" logic and implemented Virgin Level Registry.
- **Blockers:** None.
