import sys
import pandas as pd
from datetime import datetime

# Load the trade log
log_path = sys.argv[1] if len(sys.argv) > 1 else '/home/degi/autoresearch-trading/logs/trade_log.csv'
df = pd.read_csv(log_path)

# Convert times to datetime
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

# Calculate duration in days
df['duration_days'] = (df['exit_time'] - df['entry_time']).dt.days

# Basic Stats
total_trades = len(df)
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] <= 0]
win_rate = len(wins) / total_trades if total_trades > 0 else 0

# Directional Stats
longs = df[df['type'] == 'LONG']
shorts = df[df['type'] == 'SHORT']

long_win_rate = len(longs[longs['pnl'] > 0]) / len(longs) if len(longs) > 0 else 0
short_win_rate = len(shorts[shorts['pnl'] > 0]) / len(shorts) if len(shorts) > 0 else 0

# Duration Analysis for Losses
# We filter the main df to get only losses first, then analyze that subset
losses_df = df[df['pnl'] <= 0].copy()
loss_durations = losses_df['duration_days']

# Check for the 52-60 day cluster
cluster_mask = (loss_durations >= 52) & (loss_durations <= 60)
cluster_count = cluster_mask.sum()
cluster_percentage = (cluster_count / len(losses_df)) * 100 if len(losses_df) > 0 else 0

print(f"Total Trades: {total_trades}")
print(f"Overall Win Rate: {win_rate:.2%}")
print(f"LONG Win Rate: {long_win_rate:.2%}")
print(f"SHORT Win Rate: {short_win_rate:.2%}")
print(f"Losses in 52-60 day window: {cluster_count} ({cluster_percentage:.2%})")
print("\nTrades in 52-60 day loss cluster:")
print(losses_df[cluster_mask][['entry_time', 'exit_time', 'type', 'pnl', 'duration_days']])
