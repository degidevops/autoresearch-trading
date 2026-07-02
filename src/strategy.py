# src/strategy.py
import pandas as pd
import numpy as np

def get_strategy_func():
    """Returns the current strategy function."""
    def strategy(df: pd.DataFrame, api: bool = False):
        if api:
            # When called with api=True we only need to return a prompt string.
            prompt = (
                "Berikan sinyal trading dalam format JSON untuk data OHLCV berikut. "
                "Setiap elemen harus berisi 'index' (int), 'signal' (1=LONG, -1=SHORT, 0=FLAT) dan 'reason' (string). "
                "Gunakan aturan AMT-Adaptive-Timeframe-Observation: "
                "Jika close > open, sinyal LONG, jika close < open, sinyal SHORT, else FLAT. "
                f"Data: {df.head(5).to_json(orient='records')}"
            )
            return prompt
        # Local mode – return DataFrame directly
        signals = np.where(df['close'] > df['open'], 1, -1)
        reasons = np.where(df['close'] > df['open'], "Close > Open", "Close < Open")
        return pd.DataFrame({"signal": signals, "reason": reasons}, index=df.index)
    return strategy
