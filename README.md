# Auto‑Research AMT (Karpathy‑style)

## 🎯 Tujuan
Implementasi **Adaptive‑Timeframe Observation (AMT)** dengan metodologi **Karpathy Auto‑Research** (Audit → Hipotesis → Implementasi → Verifikasi).  
Repo ini sudah **bersih** dan **siap dijalankan di Google Colab** (atau secara lokal).

## 📂 Struktur folder
```
autoresearch-trading/
├─ data/                # contoh data parquet (XAUUSD 1‑menit)
├─ src/                 # kode inti
│   ├─ backtest.py      # back‑tester yang kini dapat memakai LLM via API
│   └─ strategy.py     # logika sinyal (local + LLM‑prompt mode)
├─ logs/                # trade_log.csv akan ditulis di sini
├─ requirements.txt      # dependensi Python
├─ AutoResearch‑Colab.ipynb  # notebook untuk menjalankan di Google Colab
├─ .env.example        # contoh file env untuk LLM API
└─ README.md
```

## 🚀 Cara menjalankan **lokal** (tanpa LLM)
```bash
# 1️⃣ Install dependensi
pip install -r requirements.txt

# 2️⃣ Jalankan back‑test dengan strategi dummy (tanpa LLM)
python -m src.backtest
```
Output contoh:
```
{
  "sharpe": 0.12,
  "max_dd": -0.45,
  "total_return": 0.03
}
Trade log saved to: /home/degi/autoresearch-trading/logs/trade_log.csv
```

## 📊 Cara menjalankan **di Google Colab** (dengan LLM)
1. **Upload notebook** `AutoResearch‑Colab.ipynb` ke Colab (File → Upload notebook).  
2. Pada sel **3️⃣ Set API key**, ganti `YOUR_API_KEY_HERE` dengan token LLM Anda (OpenAI, Anthropic, atau provider lain yang didukung).  
3. Jalankan semua sel. Notebook akan:
   - Meng‑install dependensi.
   - Membaca data contoh.
   - Memanggil LLM untuk menghasilkan sinyal (menggunakan prompt yang disediakan di `src/strategy.py`).
   - Menjalankan back‑test dan menampilkan metrik serta trade‑log.

### 🔑 Environment variables (opsional)
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | *none* | API key untuk provider LLM. |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, atau provider lain yang Anda tambahkan ke `requirements.txt`. |
| `COMMISSION_PER_SIDE` | `0.0004` | Komisi per sisi trade. |
| `SLIPPAGE_PER_TRADE` | `0.0002` | Slippage per trade. |
| `DATA_PATH` | `data/XAUUSD_1m_20140114_20260626.parquet` | Path ke file data OHLCV. |
| `LOGS_DIR` | `logs` | Direktori tempat trade‑log disimpan. |

## 🛠️ Pengembangan selanjutnya
- **Tambah unit‑test** di folder `tests/` (pytest).  
- **Integrasi Kanban**: buat satu *Epic* di kolom `triage` (mis. “Implement AMT dengan Virgin Levels”).  
- **Eksperimen ML**: gunakan `src/strategy.py` untuk men‑train model XGBoost pada fitur‑fitur AMT dan gunakan hasilnya sebagai sinyal.
- **CI/CD**: jalankan notebook di GitHub Actions untuk verifikasi otomatis.

---
*Repo ini dibuat oleh Anda (degi) untuk riset trading AMT dengan metodologi Karpathy.  
Semoga membantu! 🚀
