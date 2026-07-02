import sys
from pathlib import Path
sys.path.append(str(Path('src').absolute()))

from backtest import run_backtest
from strategy import generate_signal

if __name__ == "__main__":
    try:
        metrics, log_path = run_backtest(generate_signal)
        print(f"METRICS: {metrics}")
        print(f"LOG_PATH: {log_path}")
    except Exception as e:
        print(f"ERROR: {e}")
