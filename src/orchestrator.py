# src/orchestrator.py
import os
import subprocess
import pandas as pd
from pathlib import Path
from src.backtest import run_backtest, _call_llm
from src.model_factory import MLStrategy

MODEL_FACTORY_PATH = Path("src/model_factory.py")

def get_strategy_wrapper(model_factory):
    """Wraps the ML model to conform to the backtester interface."""
    def strategy(df: pd.DataFrame, api: bool = False):
        if api:
            return pd.DataFrame() # Return empty df to satisfy type checker
        
        # Local model prediction
        signals = model_factory.predict(df)
        reasons = ["ML-Model Prediction"] * len(signals)
        return pd.DataFrame({"signal": signals, "reason": reasons}, index=df.index)
    return strategy

def orchestrate():
    print("--- Starting Full Autonomous ML Evolution Loop ---")
    
    # 1. Initialize / Train Model
    print("Training ML model...")
    model = MLStrategy()
    df = pd.read_parquet(os.getenv("DATA_PATH", "data/XAUUSD_1m_20140114_20260626.parquet"))
    model.train(df)
    
    # 2. Backtest
    print("Running backtest...")
    result = run_backtest(get_strategy_wrapper(model), use_api=False)
    print(f"Metrics: {result['metrics']}")
    
    # 3. Audit & Evolve
    print("Auditing architecture...")
    with open(MODEL_FACTORY_PATH, "r") as f:
        current_code = f.read()
        
    prompt = (
        "Anda adalah Arsitek ML Trading. Analisis performa model trading berikut:\n"
        f"Metrics: {result['metrics']}\n"
        "Tugas Anda: Tulis ulang file src/model_factory.py untuk meningkatkan performa. "
        "Anda bisa menambahkan fitur (RSI, Bollinger, dll), mengganti model (ke PyTorch/LightGBM), "
        "atau mengubah arsitektur. BATASAN: Gunakan CPU only (nthread/omp_num_threads).\n"
        f"Kode saat ini:\n{current_code}\n"
        "Berikan hanya kode Python dalam blok markdown."
    )
    
    new_code_raw = _call_llm(prompt)
    if "```python" in new_code_raw:
        new_code = new_code_raw.split("```python")[1].split("```")[0]
    else:
        new_code = new_code_raw
        
    # 4. Apply & Verify
    with open(MODEL_FACTORY_PATH, "w") as f:
        f.write(new_code)
    
    # Reload model to test
    try:
        from importlib import reload
        import src.model_factory
        reload(src.model_factory)
        from src.model_factory import MLStrategy
        
        new_model = MLStrategy()
        new_model.train(df)
        new_result = run_backtest(get_strategy_wrapper(new_model), use_api=False)
        
        if new_result['metrics']['sharpe'] > result['metrics']['sharpe']:
            print("Model evolution successful! Committing.")
            subprocess.run(["git", "add", "src/model_factory.py"])
            subprocess.run(["git", "commit", "-m", "Auto-Research: ML Architecture Evolution"])
        else:
            print("Model regressed. Rolling back.")
            subprocess.run(["git", "checkout", "src/model_factory.py"])
    except Exception as e:
        print(f"Verification failed: {e}. Rolling back.")
        subprocess.run(["git", "checkout", "src/model_factory.py"])

if __name__ == "__main__":
    orchestrate()
