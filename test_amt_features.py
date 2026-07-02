"""
Test script to validate AMT feature engineering and time-block boundaries.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
DATA_PATH = Path("/home/degi/autoresearch-trading/data/XAUUSD_1m_20140114_20260626.parquet")
df = pd.read_parquet(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Use a recent slice for visual inspection
sample = df.loc['2026-03-01':'2026-03-05'].copy()

# Session labels per 6h block aligned to 05:00 WIB start
def get_session_label(ts):
    hour = ts.hour
    if 5 <= hour < 11:
        return 'S1_05-11'
    elif 11 <= hour < 17:
        return 'S2_11-17'
    elif 17 <= hour < 23:
        return 'S3_17-23'
    else:
        return 'S4_23-05'

sample['session_label'] = sample.index.map(get_session_label)

# Daily session key: reset at 05:00. Any time before 05:00 belongs to previous day's session.
sample['daily_key'] = (sample.index - pd.Timedelta(hours=5)).date
sample['session_key'] = sample['daily_key'].astype(str) + '_' + sample['session_label']

# Weekly session key: reset Tuesday 05:00. Shift back 2 days so Tuesday aligns to ISO Monday start.
weekly_shifted = sample.index - pd.Timedelta(days=2)
sample['week_id'] = weekly_shifted.isocalendar().week

# Sessional VWAP
sample['pv'] = sample['close'] * sample['volume']
sample['cum_pv'] = sample.groupby('session_key')['pv'].cumsum()
sample['cum_vol'] = sample.groupby('session_key')['volume'].cumsum()
sample['vwap_sessional'] = sample['cum_pv'] / sample['cum_vol']

# Daily VWAP
sample['daily_pv'] = sample['close'] * sample['volume']
sample['daily_cum_pv'] = sample.groupby('daily_key')['daily_pv'].cumsum()
sample['daily_cum_vol'] = sample.groupby('daily_key')['volume'].cumsum()
sample['vwap_daily'] = sample['daily_cum_pv'] / sample['daily_cum_vol']

# Weekly VWAP
sample['weekly_pv'] = sample['close'] * sample['volume']
sample['weekly_cum_pv'] = sample.groupby('week_id')['weekly_pv'].cumsum()
sample['weekly_cum_vol'] = sample.groupby('week_id')['volume'].cumsum()
sample['vwap_weekly'] = sample['weekly_cum_pv'] / sample['weekly_cum_vol']

summary = sample[['open','high','low','close','volume','session_label','daily_key','week_id','vwap_sessional','vwap_daily','vwap_weekly']].copy()

print("=== Session Boundary Check (2026-03-01) ===")
day1 = summary.loc['2026-03-01']
print(day1.to_string())

print("\n=== Daily VWAP Reset Check ===")
print(sample[['close','daily_key','vwap_daily']].head(20).to_string())

print("\n=== Weekly VWAP Reset Check ===")
print(sample[['close','week_id','vwap_weekly']].head(20).to_string())

out_path = Path('/home/degi/autoresearch-trading/logs/amt_feature_test.csv')
out_path.parent.mkdir(exist_ok=True)
summary.to_csv(out_path)
print(f"\nSaved test features to {out_path}")
