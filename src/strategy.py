# src/strategy.py
# ------------------------------------------------------------------
#  AMT Adaptive Timeframe Observation - FULL TRIPLE TIMEFRAME
#  Implements: Weekly, Daily, and Sessional TPO + VWAP.
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import timedelta

# ------------------------------------------------------------------
#  CONSTANTS
# ------------------------------------------------------------------
SESSION_START_HOUR = 5  # 05:00 WIB
VALUE_AREA_PCT = 0.70
OBS_WINDOW_MIN = 90
TICK_SIZE = 0.01        
SUSTAIN_BARS = 3        

# ------------------------------------------------------------------
#  TPO & VWAP ENGINE
# ------------------------------------------------------------------
def calculate_tpo_levels(df):
    """Calculates POC, VAH, and VAL based on TPO distribution."""
    if df.empty or len(df) < 2:
        return None, None, None

    low_bin = np.floor(df['low'].min() / TICK_SIZE) * TICK_SIZE
    high_bin = np.ceil(df['high'].max() / TICK_SIZE) * TICK_SIZE
    bins = np.arange(low_bin, high_bin + TICK_SIZE, TICK_SIZE)
    
    tpos = np.zeros_like(bins, dtype=int)
    for _, row in df.iterrows():
        mask = (bins >= row['low']) & (bins <= row['high'])
        tpos += mask.astype(int)
    
    poc_idx = np.argmax(tpos)
    poc = bins[poc_idx]
    
    total_tpos = np.sum(tpos)
    target_tpos = total_tpos * VALUE_AREA_PCT
    
    curr_tpos = tpos[poc_idx]
    upper_idx = poc_idx
    lower_idx = poc_idx
    
    while curr_tpos < target_tpos:
        up_val = tpos[upper_idx + 1] if upper_idx + 1 < len(tpos) else -1
        down_val = tpos[lower_idx - 1] if lower_idx - 1 >= 0 else -1
        if up_val > down_val:
            upper_idx += 1
            curr_tpos += up_val
        elif down_val > up_val:
            lower_idx -= 1
            curr_tpos += down_val
        else:
            if upper_idx + 1 < len(tpos):
                upper_idx += 1
                curr_tpos += up_val
            elif lower_idx - 1 >= 0:
                lower_idx -= 1
                curr_tpos += down_val
            else:
                break
    
    return poc, bins[upper_idx], bins[lower_idx]

def compute_vwap(df):
    """Standard VWAP calculation."""
    typical = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume']
    return (typical * vol).sum() / vol.sum() if vol.sum() != 0 else np.nan

# ------------------------------------------------------------------
#  STRATEGY LOGIC
# ------------------------------------------------------------------
def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # --- 1. Triple Timeframe Resampling ---
    
    # A. Weekly (Starting Tuesday as per docs)
    # 'W-TUE' resamples to weeks ending on Tuesday
    weekly_resampler = df.resample('W-TUE')
    w_data = weekly_resampler.apply(lambda x: calculate_tpo_levels(x) if not x.empty else (None, None, None))
    w_vwap = weekly_resampler.apply(lambda x: compute_vwap(x) if not x.empty else np.nan)
    
    # B. Daily (05:00 - 05:00 WIB)
    daily_resampler = df.resample('1D', origin=f'T{SESSION_START_HOUR}:00:00')
    d_data = daily_resampler.apply(lambda x: calculate_tpo_levels(x) if not x.empty else (None, None, None))
    d_vwap = daily_resampler.apply(lambda x: compute_vwap(x) if not x.empty else np.nan)
    
    # C. Sessional (6h blocks)
    sess_resampler = df.resample('6h', origin=f'T{SESSION_START_HOUR}:00:00')
    s_data = sess_resampler.apply(lambda x: calculate_tpo_levels(x) if not x.empty else (None, None, None))
    s_vwap = sess_resampler.apply(lambda x: compute_vwap(x) if not x.empty else np.nan)
    
    # --- 2. Level Alignment (Avoid Look-ahead) ---
    # Shift and ffill so we use the levels from the COMPLETED previous period
    
    # Weekly
    w_val = w_data.apply(lambda x: x[2]).shift(1).reindex(df.index, method='ffill')
    w_vah = w_data.apply(lambda x: x[1]).shift(1).reindex(df.index, method='ffill')
    w_vwap_lvl = w_vwap.shift(1).reindex(df.index, method='ffill')
    
    # Daily
    d_val = d_data.apply(lambda x: x[2]).shift(1).reindex(df.index, method='ffill')
    d_vah = d_data.apply(lambda x: x[1]).shift(1).reindex(df.index, method='ffill')
    d_vwap_lvl = d_vwap.shift(1).reindex(df.index, method='ffill')
    
    # Sessional
    s_val = s_data.apply(lambda x: x[2]).shift(1).reindex(df.index, method='ffill')
    s_vah = s_data.apply(lambda x: x[1]).shift(1).reindex(df.index, method='ffill')
    s_vwap_lvl = s_vwap.shift(1).reindex(df.index, method='ffill')
    
    price = df['close']
    low = df['low']
    high = df['high']
    
    # --- 3. Signal Identification (Multi-TF) ---
    
    # Long Conditions: Touched VAL of any TF AND Price > concurrent VWAP
    # We create a combined mask for VAL touches
    touched_val_any = (low <= w_val) | (low <= d_val) | (low <= s_val)
    
    # Acceptance: Price must close above a VAL level for SUSTAIN_BARS
    # For simplicity in baseline, we check if price is above ANY of the VALs
    above_val_any = (price > w_val) | (price > d_val) | (price > s_val)
    sustained_above = above_val_any.rolling(SUSTAIN_BARS).min().astype(bool)
    
    # VWAP Filter: Price must be above the VWAP of the TF that provided the VAL
    # This is a simplification: if price > all 3 VWAPs, it's strongly bullish.
    # If it's above the specific VWAP of the TF it touched, it's a valid signal.
    long_vwap_filter = (
        ((low <= w_val) & (price > w_vwap_lvl)) |
        ((low <= d_val) & (price > d_vwap_lvl)) |
        ((low <= s_val) & (price > s_vwap_lvl))
    )
    
    # Short Conditions: Touched VAH of any TF AND Price < concurrent VWAP
    touched_vah_any = (high >= w_vah) | (high >= d_vah) | (high >= s_vah)
    below_vah_any = (price < w_vah) | (price < d_vah) | (price < s_vah)
    sustained_below = below_vah_any.rolling(SUSTAIN_BARS).min().astype(bool)
    
    short_vwap_filter = (
        ((high >= w_vah) & (price < w_vwap_lvl)) |
        ((high >= d_vah) & (price < d_vwap_lvl)) |
        ((high >= s_vah) & (price < s_vwap_lvl))
    )
    
    long_trigger = touched_val_any & sustained_above & long_vwap_filter
    short_trigger = touched_vah_any & sustained_below & short_vwap_filter
    
    # --- 4. 90-Minute Block Logic ---
    timestamps = df.index.astype(np.int64) // 10**9
    block_id = timestamps // (OBS_WINDOW_MIN * 60)
    
    final_signal = np.zeros(len(df), dtype=int)
    final_reason = ["None"] * len(df)
    current_sig = 0
    
    for i in range(len(df)):
        if i > 0 and block_id[i] != block_id[i-1]:
            current_sig = 0
        
        if current_sig == 0:
            if long_trigger.iloc[i]:
                current_sig = 1
                final_reason[i] = "Acceptance at VAL (Multi-TF) + VWAP Filter"
            elif short_trigger.iloc[i]:
                current_sig = -1
                final_reason[i] = "Rejection at VAH (Multi-TF) + VWAP Filter"
        
        final_signal[i] = current_sig
        if current_sig != 0 and i > 0 and final_reason[i] == "None":
            final_reason[i] = final_reason[i-1] if i > 0 else "None"
        
    return pd.DataFrame({'signal': final_signal, 'reason': final_reason}, index=df.index)
