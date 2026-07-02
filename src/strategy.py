# src/strategy.py
# ------------------------------------------------------------------
#  AMT Adaptive Timeframe Observation - Baseline Strategy
#  Implements the core philosophy: TPO levels + VWAP filter + 90m blocks.
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import timedelta

# ------------------------------------------------------------------
#  CONSTANTS (Baseline)
# ------------------------------------------------------------------
SESSION_START_HOUR = 5  # XAUUSD specific open (05:00 WIB)
VALUE_AREA_PCT = 0.70
OBS_WINDOW_MIN = 90
TICK_SIZE = 0.01        # Precision for TPO binning
SUSTAIN_BARS = 3        # Bars to confirm acceptance/rejection

# ------------------------------------------------------------------
#  TPO ENGINE
# ------------------------------------------------------------------
def calculate_tpo_levels(df):
    """
    Calculates POC, VAH, and VAL based on TPO distribution.
    """
    if df.empty or len(df) < 2:
        return None, None, None

    # 1. Binning
    low_bin = np.floor(df['low'].min() / TICK_SIZE) * TICK_SIZE
    high_bin = np.ceil(df['high'].max() / TICK_SIZE) * TICK_SIZE
    bins = np.arange(low_bin, high_bin + TICK_SIZE, TICK_SIZE)
    
    tpos = np.zeros_like(bins, dtype=int)
    # For each bar, increment all bins it touched
    for _, row in df.iterrows():
        mask = (bins >= row['low']) & (bins <= row['high'])
        tpos += mask.astype(int)
    
    # 2. Point of Control (POC)
    poc_idx = np.argmax(tpos)
    poc = bins[poc_idx]
    
    # 3. Value Area (VAH/VAL)
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
    """
    Generates signals based on AMT Adaptive Timeframe Observation.
    Returns a DataFrame with 'signal' and 'reason'.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 1. Session-based Level Calculation (Sessional 6h blocks)
    # We use 6h blocks as the primary observation driver as per docs
    resampler = df.resample('6h', origin=f'T{SESSION_START_HOUR}:00:00')
    
    # Calculate TPO and VWAP for each 6h block
    session_data = resampler.apply(lambda x: calculate_tpo_levels(x) if not x.empty else (None, None, None))
    session_vwaps = resampler.apply(lambda x: compute_vwap(x) if not x.empty else np.nan)
    
    # Shift levels forward by one session to avoid look-ahead bias
    # We use the levels from the PREVIOUS session to trade the CURRENT session
    val_levels = session_data.apply(lambda x: x[2]).shift(1).reindex(df.index, method='ffill')
    vah_levels = session_data.apply(lambda x: x[1]).shift(1).reindex(df.index, method='ffill')
    vwap_levels = session_vwaps.shift(1).reindex(df.index, method='ffill')
    
    price = df['close']
    low = df['low']
    high = df['high']
    
    # 2. Signal identification
    # Long: Price touches VAL and then closes above VAL for SUSTAIN_BARS
    # Short: Price touches VAH and then closes below VAH for SUSTAIN_BARS
    
    # Touch detection
    touched_val = (low <= val_levels)
    touched_vah = (high >= vah_levels)
    
    # Acceptance/Rejection confirmation (sustained close)
    above_val = (price > val_levels).rolling(SUSTAIN_BARS).min().astype(bool)
    below_vah = (price < vah_levels).rolling(SUSTAIN_BARS).min().astype(bool)
    
    # Combine with VWAP momentum filter
    long_trigger = touched_val & above_val & (price > vwap_levels)
    short_trigger = touched_vah & below_vah & (price < vwap_levels)
    
    # 3. 90-Minute Observation Block Logic
    # Decisions are made ONLY within the current 90-min window
    timestamps = df.index.astype(np.int64) // 10**9
    block_id = timestamps // (OBS_WINDOW_MIN * 60)
    
    final_signal = np.zeros(len(df), dtype=int)
    final_reason = ["None"] * len(df)
    current_sig = 0
    
    for i in range(len(df)):
        # Reset signal at the start of a new 90-min block
        if i > 0 and block_id[i] != block_id[i-1]:
            current_sig = 0
        
        if current_sig == 0:
            if long_trigger.iloc[i]:
                current_sig = 1
                final_reason[i] = "Acceptance at VAL + Bullish VWAP"
            elif short_trigger.iloc[i]:
                current_sig = -1
                final_reason[i] = "Rejection at VAH + Bearish VWAP"
        
        final_signal[i] = current_sig
        if current_sig != 0 and i > 0 and final_reason[i] == "None":
            # Maintain the reason for the duration of the trade in the block
            final_reason[i] = final_reason[i-1] if i > 0 else "None"
        
    return pd.DataFrame({'signal': final_signal, 'reason': final_reason}, index=df.index)
