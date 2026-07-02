# Auto‑Research AMT (Karpathy‑style + Full ML Evolution)

## 🎯 Tujuan
Implementasi **Adaptive‑Timeframe Observation (AMT)** dengan metodologi **Karpathy Auto‑Research** yang berevolusi secara otonom.

## 📂 Struktur folder
```
autoresearch-trading/
├─ src/
│   ├─ backtest.py       # Engine backtest + _call_llm(...)
│  │                        # LLM modes: local | openai | anthropic
│  │                        # Hermes mode: use Hermes/agent to drive/run it
│   ├─ model_factory.py  # Model ML (Auto-Architected by LLM)
│   ├─ orchestrator.py   # Autonomous evolution loop
│   └─ audit.py          # Auditor agent
├─ logs/                 # Trade logs & audit reports
├─ src/setup_colab.py    # One-click Colab setup
└─ ...
```

## 🧠 3 Opsi LLM
| Opsi | Cara pakai |
|------|-----------|
| `local` | Isi `.env` → jalankan `python -m src.orchestrator` |
| API: `openai` / `anthropic` | Isi `.env` → jalankan `python -m src.orchestrator` |
| **Hermes sebagai agent/driver** | Jalankan dari dalam Hermes/Telegram/DM/agent; Hermes menjalankan orchestrator sebagai agent, bukan sebagai modul impor |

## 🚀 Cara Menjalankan

### Lokal / Colab
1. Install dependensi.
2. Salin `.env.example` ke `.env`.
3. Set `LLM_PROVIDER=local` atau `openai`/`anthropic`.
4. Jalankan:
   ```bash
   python -m src.orchestrator
   ```

### Hermes sebagai Agent
Jalankan dari dalam Hermes:
- Gunakan Hermes untuk mengeksekusi langkah orchestrator sebagai tool/terminal,
  atau kirim melalui alat terminal terintegrasi Hermes.
- File yang berubah adalah `src/model_factory.py` saja, lalu verifikasi + git commit.

---
*Repo ini adalah mesin riset trading otonom yang terus meng-optimalkan model ML-nya sendiri.*
