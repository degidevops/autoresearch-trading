# 📓 Research Log: AMT Auto-Research

## Hypothesis #1: Time-Stop Discipline & Directional Efficiency
**Date:** 2026-07-02
**Status:** Defining Success Metrics

### Root Cause (from t_9bc7881a)
Baseline failure is driven by a lack of time-stop discipline (52-60 day cluster) and poor performance of SHORT trades, independent of the trading timeframe.

### Success Metrics (The 'Win' Condition)
A "win" for this hypothesis is defined as:
1. **Cluster Reduction:** Loss rate in the 52-60 day duration window reduced by $\ge 30\%$ or cluster eliminated via hard time-stop.
2. **Directional Parity:** SHORT win rate brought within 10% of LONG win rate.
3. **Net Improvement:** Increase in total PNL specifically attributable to the reduction of these two failure modes.

### Formalized Hypothesis
**Hypothesis:** The baseline failure rate is primarily driven by structural inefficiencies in trade duration (the 52-60 day loss cluster) and a systemic directional bias (SHORT trade underperformance). By introducing a dynamic risk-filtering layer—utilizing duration-based exit signals and regime-based entry filters for SHORT positions—we can reduce the failure rate in these specific segments by $\ge 30\%$ and achieve win-rate parity between LONG and SHORT trades, thereby increasing overall portfolio PNL.

### Proposed Implementation
Leverage a Machine Learning model (e.g., XGBoost/Random Forest) to generate a "Risk Score" based on:
- **Duration Risk:** `trade_duration_days`, `duration_cluster_risk`, `signal_decay_rate`.
- **Directional Bias:** `market_regime_trend`, `relative_strength_index`, `volatility_spike_entry`.
- **MTF Confluence:** `tf_confluence_score`, `tf_divergence_flag`.

Trades with a Risk Score exceeding a defined threshold will be either filtered at entry (for SHORTs) or closed early (for trades entering the 52-60 day window).

---
