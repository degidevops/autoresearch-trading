# src/backtest.py
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Callable

# Configuration
COMMISSION_PER_SIDE = float(os.getenv("COMMISSION_PER_SIDE", "0.0004"))
SLIPPAGE_PER_TRADE = float(os.getenv("SLIPPAGE_PER_TRADE", "0.0002"))
DATA_PATH = Path(os.getenv("DATA_PATH", "/home/degi/autoresearch-trading/data/XAUUSD_1m_20140114_20260626.parquet"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", "/home/degi/autoresearch-trading/logs"))

def _call_llm(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "hermes").lower()
    if provider == "hermes":
        try:
            from hermes_tools import delegate_task
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER=hermes requires running inside Hermes Agent. "
                "Set LLM_PROVIDER=local or openai to run standalone."
            ) from e
        result = delegate_task(
            goal="Analisis riset trading dan berikan output yang diminta.",
            context=prompt
        )
        return result[0]
    elif provider == "local":
        from llama_cpp import Llama
        model_path = os.getenv("LOCAL_MODEL_PATH", "/content/model.gguf")
        llm = Llama(model_path=model_path, n_ctx=4096, n_threads=os.cpu_count() or 1)
        resp = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp['choices'][0]['message']['content']
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set")
    if provider == "openai":
        import openai
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))
    raise ValueError(f"Unsupported LLM provider: {provider}")

def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    else:
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def compute_metrics(equity_curve: pd.Series, timestamps: pd.DatetimeIndex) -> dict:
    returns = equity_curve.pct_change().fillna(0)
    ret_series = pd.Series(returns.values[1:], index=timestamps)
    daily_ret = ret_series.resample('1D').sum()
    mean_daily = daily_ret.mean()
    std_daily = daily_ret.std(ddof=0)
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily != 0 else -999
    max_dd = ((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min()
    return {"sharpe": float(sharpe), "max_dd": float(max_dd), "total_return": float(equity_curve.iloc[-1] - 1)}

def run_backtest(strategy_func: Callable, **kwargs) -> dict:
    df = load_data()
    results_df = strategy_func(df)
    if not isinstance(results_df, pd.DataFrame):
        raise ValueError("strategy_func must return a DataFrame with 'signal' and 'reason'")
    df = df.join(results_df)
    equity = [1.0]
    trade_log = []
    current_pos = 0
    entry_price = 0.0
    entry_time = None
    entry_reason = ""
    for i in range(len(df)):
        price = df['close'].iloc[i]
        sig = df['signal'].iloc[i]
        reason = df['reason'].iloc[i] if 'reason' in df.columns else ""
        if current_pos != 0 and (sig == 0 or sig == -current_pos):
            trade_ret = (price - entry_price) / entry_price if current_pos == 1 else (entry_price - price) / entry_price
            net_ret = trade_ret - (COMMISSION_PER_SIDE * 2 + SLIPPAGE_PER_TRADE)
            trade_log.append({
                "entry_time": entry_time,
                "exit_time": df.index[i],
                "type": "LONG" if current_pos == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": price,
                "pnl": net_ret,
                "reason": entry_reason,
            })
            current_pos = 0
        if current_pos == 0 and sig != 0:
            current_pos = sig
            entry_price = price
            entry_time = df.index[i]
            entry_reason = reason
        if i > 0:
            ret = (price - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
            equity.append(equity[-1] * (1 + ret * current_pos))
        else:
            equity.append(equity[-1])
    equity_series = pd.Series(equity, index=[df.index[0]] + list(df.index))
    metrics = compute_metrics(equity_series, pd.DatetimeIndex(df.index))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(trade_log)
    log_path = LOGS_DIR / "trade_log.csv"
    log_df.to_csv(log_path, index=False)
    return {"metrics": metrics, "log_path": str(log_path)}
