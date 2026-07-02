# src/audit.py
# ------------------------------------------------------------------
#  Auto-Research Audit Agent
#  Goal: Analyze trade_log.csv + metrics to identify failure modes
#  and propose hypothesis improvements.
# ------------------------------------------------------------------
import pandas as pd
import json
import os
from pathlib import Path
from src.backtest import _call_llm

LOGS_DIR = Path(os.getenv("LOGS_DIR", "/home/degi/autoresearch-trading/logs"))

def run_audit():
    log_path = LOGS_DIR / "trade_log.csv"
    if not log_path.exists():
        print(f"No log file found at {log_path}. Run a backtest first.")
        return

    # Load logs
    df = pd.read_csv(log_path)
    
    # Identify top 5 worst trades by PnL
    worst_trades = df.nsmallest(5, 'pnl')
    
    # Prepare prompt for LLM
    prompt = (
        "Anda adalah Auditor Strategi Trading yang kritis. "
        "Tugas Anda adalah menganalisis trade yang gagal dan memperbaiki strategi.\n\n"
        "Data 5 Trade dengan kerugian terbesar:\n"
        f"{worst_trades.to_json(orient='records')}\n\n"
        "Tugas Anda:\n"
        "1. Identifikasi pola kegagalan (mis. apakah karena salah timing, slippage, atau logika sinyal yang buruk).\n"
        "2. Berikan usulan perbaikan (Hipotesis baru) untuk aturan trading berikutnya.\n"
        "3. Berikan alasan mengapa strategi ini gagal di pasar saat ini."
    )
    
    print("Sending trade logs to LLM for audit...")
    analysis = _call_llm(prompt)
    
    # Save analysis
    audit_path = LOGS_DIR / "audit_report.md"
    with open(audit_path, "w") as f:
        f.write("# Auto-Research Audit Report\n\n")
        f.write(analysis)
    
    print(f"Audit complete! Report saved to: {audit_path}")
    print("\n--- AUDIT ANALYSIS ---\n")
    print(analysis)

if __name__ == "__main__":
    run_audit()
