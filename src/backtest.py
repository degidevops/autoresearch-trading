# src/backtest.py
# ------------------------------------------------------------------
#  Advanced Backtester for Auto‑research (Karpathy‑style)
#  Goal: Provide detailed trade logs for agent analysis, not just metrics.
# ------------------------------------------------------------------
import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Callable, Any

# ------------------------------------------------------------------
#  Configuration (adjustable via env vars if needed)
# ------------------------------------------------------------------
COMMISSION_PER_SIDE = float(os.getenv("COMMISSION_PER_SIDE", "0.0004"))
SLIPPAGE_PER_TRADE = float(os.getenv("SLIPPAGE_PER_TRADE", "0.0002"))
DATA_PATH = Path(os.getenv("DATA_PATH", "/home/degi/autoresearch-trading/data/XAUUSD_1m_20140114_20260626.parquet"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", "/home/degi/autoresearch-trading/logs"))

# ------------------------------------------------------------------
#  Helper – LLM API wrapper (OpenAI or Anthropic)
# ------------------------------------------------------------------
def _call_llm(prompt: str) -> str:
    """Call the configured LLM provider and return the raw response text.

    The provider is selected via the ``LLM_PROVIDER`` environment variable:
    - ``openai``  (default) – uses ``openai.ChatCompletion`` with model ``gpt‑4o‑mini``.
    - ``anthropic`` – uses ``anthropic.Anthropic`` with model ``claude‑3‑5‑sonnet‑20240620``.
    The API key is read from ``LLM_API_KEY``.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set in environment")

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
        # anthropic returns a list of content blocks; we join them
        return "".join(block.text for block in resp.content if hasattr(block, "text"))
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# ------------------------------------------------------------------
#  Data loading
# ------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load OHLCV data from ``DATA_PATH``.
    The function raises ``FileNotFoundError`` if the parquet file is missing.
    """
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values('timestamp').set_index('timestamp')
    return df

# ------------------------------------------------------------------
#  Metric computation (Sharpe, Max DD, Total Return)
# ------------------------------------------------------------------
def compute_metrics(equity_curve: pd.Series, timestamps: pd.DatetimeIndex) -> dict:
    """Return a dict with ``sharpe``, ``max_dd`` and ``total_return``.
    ``equity_curve`` must include the initial capital (1.0) as the first element.
    """
    returns = equity_curve.pct_change().fillna(0)
    # Align returns with timestamps (skip the first element which is the seed)
    ret_series = pd.Series(returns.values[1:], index=timestamps)
    daily_ret = ret_series.resample('1D').sum()
    mean_daily = daily_ret.mean()
    std_daily = daily_ret.std(ddof=0)
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily != 0 else -999
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd = drawdown.min()
    return {
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "total_return": float(equity_curve.iloc[-1] - 1),
    }

# ------------------------------------------------------------------
#  Core back‑test runner
# ------------------------------------------------------------------
def run_backtest(
    strategy_func: Callable[[pd.DataFrame, bool], pd.DataFrame],
    use_api: bool = False,
    **kwargs,
) -> dict:
    """Execute a back‑test.

    Parameters
    ----------
    strategy_func:
        Callable that receives the OHLCV ``DataFrame`` and a boolean ``api`` flag.
        It must return a ``DataFrame`` with at least two columns:
        ``signal`` (1 = LONG, -1 = SHORT, 0 = FLAT) and ``reason`` (string).
        When ``api=True`` the function should **return a prompt string** that
        the LLM will answer with a JSON array of the same schema.
    use_api:
        If ``True`` the runner will call the LLM via ``_call_llm`` and parse
        the JSON response into a DataFrame before proceeding.
    kwargs:
        Additional arguments are passed unchanged to ``strategy_func``.
    """
    df = load_data()

    # ------------------------------------------------------------------
    #  Obtain signals – either locally or via LLM API
    # ------------------------------------------------------------------
    if use_api:
        # strategy_func should return a *prompt* when called with api=True
        prompt = strategy_func(df, api=True, **kwargs)
        raw = _call_llm(prompt)
        # Expect raw to be a JSON array like [{"index": 0, "signal": 1, "reason": "..."}, ...]
        try:
            signals = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {exc}\nRaw output: {raw}")
        # Build a DataFrame indexed by the original timestamps
        sig_df = pd.DataFrame(signals).set_index('index')
        # Align to the original df – we assume the index list matches row order
        sig_df.index = df.index[: len(sig_df)]
        results_df = sig_df[['signal', 'reason']]
    else:
        # Local mode – strategy_func receives the df and returns a DataFrame directly
        results_df = strategy_func(df, api=False, **kwargs)

    # ------------------------------------------------------------------
    #  Merge signals back into price data
    # ------------------------------------------------------------------
    df = df.join(results_df)

    # ------------------------------------------------------------------
    #  Simulation loop
    # ------------------------------------------------------------------
    equity = [1.0]
    trade_log = []
    current_pos = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    entry_time = None
    entry_reason = ""
    cost_per_trade = (COMMISSION_PER_SIDE * 2) + SLIPPAGE_PER_TRADE

    for i in range(len(df)):
        price = df['close'].iloc[i]
        time = df.index[i]
        sig = df['signal'].iloc[i]
        reason = df['reason'].iloc[i]

        # Close or flip position
        if current_pos != 0 and (sig == 0 or sig == -current_pos):
            exit_price = price
            exit_time = time
            trade_ret = (exit_price - entry_price) / entry_price if current_pos == 1 else (entry_price - exit_price) / entry_price
            net_ret = trade_ret - cost_per_trade
            trade_log.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "type": "LONG" if current_pos == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": net_ret,
                "reason": entry_reason,
            })
            current_pos = 0

        # Open new position
        if current_pos == 0 and sig != 0:
            current_pos = sig
            entry_price = price
            entry_time = time
            entry_reason = reason

        # Equity tracking – simple mark‑to‑market on each bar
        if i > 0:
            prev_price = df['close'].iloc[i - 1]
            ret = (price - prev_price) / prev_price
            equity.append(equity[-1] * (1 + ret * current_pos))
        else:
            equity.append(equity[-1])

    equity_series = pd.Series(equity, index=[df.index[0]] + list(df.index))
    metrics = compute_metrics(equity_series, df.index)

    # ------------------------------------------------------------------
    #  Persist trade log
    # ------------------------------------------------------------------
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(trade_log)
    log_path = LOGS_DIR / "trade_log.csv"
    log_df.to_csv(log_path, index=False)

    return {"metrics": metrics, "log_path": str(log_path)}

# ------------------------------------------------------------------
#  Simple dummy strategy for quick local testing (used when __main__)
# ------------------------------------------------------------------
if __name__ == "__main__":
    def dummy_strategy(df: pd.DataFrame, api: bool = False):
        if api:
            # When called with api=True we only need to return a prompt string.
            # The LLM will be asked to produce a JSON array of signals.
            prompt = (
                "Berikan sinyal trading dalam format JSON untuk data OHLCV berikut. "
                "Setiap elemen harus berisi 'index' (int), 'signal' (1=LONG, -1=SHORT, 0=FLAT) dan 'reason' (string). "
                "Gunakan aturan AMT‑Adaptive‑Timeframe‑Observation: "
                "Jika close > open, sinyal LONG, jika close < open, sinyal SHORT, else FLAT. "
                f"Data: {df.head(5).to_json(orient='records')}"
            )
            return prompt
        # Local mode – return DataFrame directly
        signals = np.where(df['close'] > df['open'], 1, -1)
        reasons = np.where(df['close'] > df['open'], "Close > Open", "Close < Open")
        return pd.DataFrame({"signal": signals, "reason": reasons}, index=df.index)

    # Run with local dummy (no LLM) – useful for quick sanity check
    result = run_backtest(dummy_strategy, use_api=False)
    print(json.dumps(result["metrics"], indent=2))
    print(f"Trade log saved to: {result['log_path']}")
