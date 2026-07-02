# src/orchestrator.py
import os
import subprocess
import pandas as pd
from pathlib import Path
from src.backtest import run_backtest, _call_llm
from src.model_factory import MLStrategy

DATA_PATH = Path(os.getenv("DATA_PATH", "/home/degi/autoresearch-trading/data/XAUUSD_1m_20140114_20260626.parquet"))
MODEL_FACTORY_PATH = Path("src/model_factory.py")

def get_strategy_wrapper(model: MLStrategy):
    def strategy(df: pd.DataFrame, api: bool = False):
        if api:
            return pd.DataFrame()
        predictions = model.predict(df)
        reasons = ["ML-Model Prediction"] * len(predictions)
        return pd.DataFrame({"signal": predictions, "reason": reasons}, index=df.index)
    return strategy

def run_git_commit(message: str):
    try:
        subprocess.run(["git", "add", "src/model_factory.py"], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        print(f"[GIT] Committed: {message}")
    except subprocess.CalledProcessError as e:
        print(f"[GIT] Error: {e}")

def run_git_checkout():
    try:
        subprocess.run(["git", "checkout", "src/model_factory.py"], check=True)
        print("[GIT] Rolled back changes")
    except subprocess.CalledProcessError as e:
        print(f"[GIT] Rollback error: {e}")

def orchestrate():
    print("=" * 60)
    print("AUTO-RESEARCH: Full Autonomous ML Evolution Loop")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH).sort_index()

    # Step 1: Train baseline model
    print("\n[1/5] Training ML model...")
    model = MLStrategy()
    model.train(df)

    # Step 2: Backtest baseline
    print("[2/5] Running backtest on baseline model...")
    baseline_result = run_backtest(get_strategy_wrapper(model))
    print(f"      Sharpe: {baseline_result['metrics']['sharpe']:.4f}, MaxDD: {baseline_result['metrics']['max_dd']:.4f}")

    # Step 3: Audit & Evolve
    print("[3/5] Auditing and evolving architecture...")
    with open(MODEL_FACTORY_PATH, "r") as f:
        current_code = f.read()

    prompt = (
        "Anda adalah Arsitek ML Trading Auto-Research. Analisis performa model berikut:\n"
        f"Sharpe: {baseline_result['metrics']['sharpe']:.4f}, MaxDD: {baseline_result['metrics']['max_dd']:.4f}\n\n"
        "Tugas Anda: Tulis ulang src/model_factory.py untuk meningkatkan Sharpe ratio.\n"
        "Fokus pada:\n"
        "- Fitur AMT (VWAP sessional/daily/weekly, TPO VAH/VAL/POC)\n"
        "- Feature engineering yang lebih baik (interaksi antar level)\n"
        "- Model ML yang cocok untuk tabular time-series\n"
        "BATASAN: CPU only, gunakan library yang sudah terinstal (xgboost, pandas, numpy)\n"
        "Jangan ubah interface class MLStrategy (train/predict).\n\n"
        f"Kode saat ini:\n{current_code}\n\n"
        "Berikan HANYA kode Python lengkap dalam blok markdown."
    )

    new_code_raw = _call_llm(prompt)
    if "```python" in new_code_raw:
        new_code = new_code_raw.split("```python")[1].split("```")[0]
    elif "```" in new_code_raw:
        new_code = new_code_raw.split("```")[1].split("```")[0]
    else:
        new_code = new_code_raw

    # Step 4: Apply changes
    print("[4/5] Applying new architecture...")
    with open(MODEL_FACTORY_PATH, "w") as f:
        f.write(new_code)

    try:
        import importlib
        import src.model_factory as mf_module
        mf_module = importlib.reload(mf_module)
        NewMLStrategy = mf_module.MLStrategy

        new_model = NewMLStrategy()
        new_model.train(df)
        new_result = run_backtest(get_strategy_wrapper(new_model))

        print(f"      New Sharpe: {new_result['metrics']['sharpe']:.4f}, MaxDD: {new_result['metrics']['max_dd']:.4f}")

        # Step 5: Verify & Commit
        print("[5/5] Verifying improvement...")
        if new_result['metrics']['sharpe'] > baseline_result['metrics']['sharpe']:
            print("      ✅ Improvement detected!")
            run_git_commit(f"Auto-Research: Architecture evolution (Sharpe {baseline_result['metrics']['sharpe']:.4f} → {new_result['metrics']['sharpe']:.4f})")
        else:
            print("      ⚠️  No improvement. Rolling back.")
            run_git_checkout()
    except Exception as e:
        print(f"      ❌ Verification failed: {e}")
        run_git_checkout()

    print("\n" + "=" * 60)
    print("Evolution loop complete.")
    print("=" * 60)

if __name__ == "__main__":
    orchestrate()
