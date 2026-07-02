# src/backtest.py
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Callable, Any

# Configuration
COMMISSION_PER_SIDE = float(os.getenv("COMMISSION_PER_SIDE", "0.0004"))
SLIPPAGE_PER_TRADE = float(os.getenv("SLIPPAGE_PER_TRADE", "0.0002"))
DATA_PATH = Path(os.getenv("DATA_PATH", "/home/degi/autoresearch-trading/data/XAUUSD_1m_20140114_20260626.parquet"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", "/home/degi/autoresearch-trading/logs"))
CPU_COUNT = os.cpu_count() or 1

def _call_llm(prompt: str) -> str:
    """Delegasikan ke LLM via API, Hermes Agent, atau Local GGUF."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "hermes":
        from hermes_tools import delegate_task
        result = delegate_task(
            goal="Analisis riset trading dan berikan output yang diminta.",
            context=prompt
        )
        return result[0]
        
    elif provider == "local":
        from llama_cpp import Llama
        model_path = os.getenv("LOCAL_MODEL_PATH", "/content/model.gguf")
        llm = Llama(model_path=model_path, n_ctx=4096, n_threads=CPU_COUNT)
        resp = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp['choices'][0]['message']['content']
    
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set for API provider")

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
    return df.sort_values('timestamp').set_index('timestamp')

def compute_metrics(equity_curve: pd.Series, timestamps: pd.DatetimeIndex) -> dict:
    returns = equity_curve.pct_change().fillna(0)
    ret_series = pd.Series(returns.values[1:], index=timestamps)
    daily_ret = ret_series.resample('1D').sum()
    mean_daily = daily_ret.mean()
    std_daily = daily_ret.std(ddof=0)
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily != 0 else -999
    max_dd = ((equity_curve - equity_curve.cummax()) / equity_curve.cummax()).min()
    return {"sharpe": float(sharpe), "max_dd": float(max_dd), "total_return": float(equity_curve.iloc[-1] - 1)}

def run_backtest(strategy_func: Callable, use_api: bool = False, **kwargs) -> dict:
    df = load_data()
    if use_api:
        prompt = strategy_func(df, api=True, **kwargs)
        raw = _call_llm(prompt)
        try:
            signals = json.loads(raw)
        except:
            signals = []
        sig_df = pd.DataFrame(signals).set_index('index')
        sig_df.index = df.index[: len(sig_df)]
        results_df = sig_df[['signal', 'reason']]
    else:
        results_df = strategy_func(df, api=False, **kwargs)

    df = df.join(results_df)
    equity = [1.0]
    current_pos = 0
    entry_price = 0.0
    for i in range(len(df)):
        price = df['close'].iloc[i]
        sig = df['signal'].iloc[i]
        if current_pos != 0 and (sig == 0 or sig == -current_pos):
            current_pos = 0
        if current_pos == 0 and sig != 0:
            current_pos = sig
            entry_price = price
        if i > 0:
            ret = (price - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
            equity.append(equity[-1] * (1 + ret * current_pos))
        else:
            equity.append(equity[-1])
            
    metrics = compute_metrics(pd.Series(equity, index=[df.index[0]] + list(df.index)), df.index)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return {"metrics": metrics, "log_path": str(LOGS_DIR / "trade_log.csv")}
