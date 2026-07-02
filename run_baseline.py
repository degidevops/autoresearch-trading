import json
from src.backtest import run_backtest
from src.strategy import generate_signal

if __name__ == "__main__":
    print("Starting baseline backtest on 12y data...")
    try:
        result = run_backtest(generate_signal)
        metrics = result["metrics"]
        log_file = result["log_path"]
        print("\n--- Backtest Metrics ---")
        print(json.dumps(metrics, indent=4))
        print(f"\nTrade log saved to: {log_file}")
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
