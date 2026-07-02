# src/audit_logs.py
# ------------------------------------------------------------------
#  AMT Trade Log Auditor
#  Goal: Identify "Clusters of Failure" to inform ML Hypothesis.
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path

# Use absolute paths to avoid cwd issues
BASE_DIR = Path("/home/degi/autoresearch-trading")
DATA_PATH = BASE_DIR / "data/XAUUSD_1m_20140114_20260626.parquet"
LOG_PATH = BASE_DIR / "logs/trade_log.csv"

def analyze_failures():
    if not LOG_PATH.is_file():
        print(f"Trade log not found at {LOG_PATH}")
        return

    trades = pd.read_csv(LOG_PATH)
    # We only load data if needed for deep analysis, but let's start with log analysis
    
    # Convert times to datetime
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    # Separate Wins and Losses
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] <= 0]
    
    win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
    
    print(f"--- General Stats ---")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Win: {wins['pnl'].mean():.4f}")
    print(f"Avg Loss: {losses['pnl'].mean():.4f}")
    print(f"Profit Factor: {abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 else float('inf'):.2f}")
    
    # 1. Time of Day Analysis
    trades['entry_hour'] = trades['entry_time'].dt.hour
    hour_dist = trades.groupby('entry_hour')['pnl'].mean()
    print("\n--- PnL by Entry Hour (UTC) ---")
    print(hour_dist)
    
    # 2. Trade Duration Analysis
    trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600
    print("\n--- Duration Analysis ---")
    print(f"Avg Duration: {trades['duration'].mean():.2f} hours")
    print(f"Max Duration: {trades['duration'].max():.2f} hours")
    
    # 3. Cluster Analysis: Loss vs Win Duration
    avg_dur_win = wins['duration'].mean() if not wins.empty else 0
    avg_dur_loss = losses['duration'].mean() if not wins.empty else 0
    print(f"\nAvg Duration Win: {avg_dur_win:.2f}h | Avg Duration Loss: {avg_dur_loss:.2f}h")
    
    # 4. Failure Correlation with TPO Reason
    reason_stats = trades.groupby('reason')['pnl'].agg(['mean', 'count'])
    print("\n--- PnL by Reason ---")
    print(reason_stats)

if __name__ == "__main__":
    analyze_failures()
