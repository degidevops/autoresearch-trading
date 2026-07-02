# src/backtest.py
# ------------------------------------------------------------------
#  Advanced Backtester for Auto-research
#  Goal: Provide detailed trade logs for agent analysis, not just metrics.
# ------------------------------------------------------------------
import json
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuration
COMMISSION_PER_SIDE = 0.0004
SLIPPAGE_PER_TRADE = 0.0002
DATA_PATH = Path("data/XAUUSD_1m_20260504_20260515.parquet")
LOGS_DIR = Path("logs")

def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values('timestamp').set_index('timestamp')
    return df

def compute_metrics(equity_curve):
    returns = equity_curve.pct_change().fillna(0)
    # Annualize to 252 trading days (approx 252 * 24 * 60 mins for crypto/gold)
    # For M1 data, we resample to daily first for a stable Sharpe
    daily_ret = returns.resample('1D').sum()
    mean_daily = daily_ret.mean()
    std_daily = daily_ret.std(ddof=0)
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily != 0 else -999
    
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd = drawdown.min()
    
    return {
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "total_return": float(equity_curve.iloc[-1] - 1)
    }

def run_backtest(strategy_func):
    df = load_data()
    
    # Get signals from strategy
    # strategy_func should return a DataFrame with:
    # 'signal': 1 (long), -1 (short), 0 (flat)
    # 'reason': string explaining the signal
    results_df = strategy_func(df)
    
    # Merge results back to main df
    df = df.join(results_df)
    
    equity = [1.0]
    trade_log = []
    current_pos = 0 # 1: long, -1: short, 0: flat
    entry_price = 0
    entry_time = None
    entry_reason = ""
    
    cost_per_trade = (COMMISSION_PER_SIDE * 2) + SLIPPAGE_PER_TRADE
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        time = df.index[i]
        sig = df['signal'].iloc[i]
        reason = df['reason'].iloc[i]
        
        # Check for position close/switch
        if current_pos != 0 and (sig == 0 or sig == -current_pos):
            # Close position
            exit_price = price
            exit_time = time
            
            # PnL calculation
            trade_ret = (exit_price - entry_price) / entry_price if current_pos == 1 else (entry_price - exit_price) / entry_price
            net_ret = trade_ret - cost_per_trade
            
            trade_log.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "type": "LONG" if current_pos == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": net_ret,
                "reason": entry_reason
            })
            
            current_pos = 0
            
        # Check for new entry
        if current_pos == 0 and sig != 0:
            current_pos = sig
            entry_price = price
            entry_time = time
            entry_reason = reason
            
        # Equity tracking (simplified)
        if current_pos != 0:
            # Daily return contribution
            ret = (price - df['close'].iloc[i-1]) / df['close'].iloc[i-1] if i > 0 else 0
            equity.append(equity[-1] * (1 + ret * current_pos))
        else:
            equity.append(equity[-1])

    equity_series = pd.Series(equity)
    metrics = compute_metrics(equity_series)
    
    # Save Trade Log
    LOGS_DIR.mkdir(exist_ok=True)
    log_df = pd.DataFrame(trade_log)
    log_path = LOGS_DIR / "trade_log.csv"
    log_df.to_csv(log_path, index=False)
    
    return metrics, log_path

if __name__ == "__main__":
    # Simple dummy strategy for testing
    def dummy_strategy(df):
        # Long if close > open, Short if close < open
        signals = np.where(df['close'] > df['open'], 1, -1)
        reasons = np.where(df['close'] > df['open'], "Close > Open", "Close < Open")
        return pd.DataFrame({'signal': signals, 'reason': reasons}, index=df.index)
    
    try:
        metrics, log_file = run_backtest(dummy_strategy)
        print(json.dumps(metrics))
        print(f"Trade log saved to: {log_file}")
    except Exception as e:
        print(f"Error: {e}")
