# Auto‑Research AMT (Karpathy‑style + Full ML Evolution)

## 🎯 Tujuan
Implementasi **Adaptive‑Timeframe Observation (AMT)** dengan metodologi **Karpathy Auto‑Research** yang berevolusi secara otonom.

## 📂 Struktur folder
```
autoresearch-trading/
├─ src/
│   ├─ backtest.py       # Engine backtest (supports Hermes, API, Local LLM)
│   ├─ model_factory.py  # Model ML (Auto-Architected by LLM)
│   ├─ orchestrator.py   # Autonomous evolution loop
│   └─ audit.py          # Auditor agent
├─ logs/                 # Trade logs & audit reports
├─ src/setup_colab.py    # One-click Colab setup
└─ ...
```

## 🚀 Cara Menjalankan

### Lokal
1. Install `llama-cpp-python` jika ingin mode lokal.
2. Set provider:
   ```bash
   export LLM_PROVIDER=local
   export LOCAL_MODEL_PATH=/path/to/model.gguf
   python -m src.orchestrator
   ```

### Google Colab
1. Gunakan `src/setup_colab.py` untuk menginstal dependensi.
2. Set environment variables.
3. Jalankan `python -m src.orchestrator`.

---
*Repo ini adalah mesin riset trading otonom yang terus meng-optimalkan model ML-nya sendiri.*
