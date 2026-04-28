# 📈 TFT Stock Analytics — Top 5 Bank Indonesia

Aplikasi web analytics prediksi harga saham menggunakan **Temporal Fusion Transformer (TFT)**
dengan strategi **MIMO (Multi-Input Multiple Output)**.

---

## 🏗️ Struktur Proyek

```
tft_stock_app/
├── app.py                  # Streamlit UI utama
├── data_acquisition.py     # Pengambilan data via yfinance
├── feature_engineering.py  # RSI, MA, Volatilitas, fitur temporal
├── model.py                # TFT model (pytorch-forecasting / fallback LSTM)
├── evaluation.py           # MAE, RMSE, MAPE, R², Directional Accuracy
├── utils.py                # Cache, logging, environment check
├── requirements.txt        # Dependensi Python
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependensi

```bash
# Buat virtual environment (disarankan)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install semua library
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`

---

## 🧠 Arsitektur Model

### Temporal Fusion Transformer (TFT)

```
Input Layer
  ├── Static Covariates      → ticker_id (Bank BCA, BRI, Mandiri, BNI, BSI)
  ├── Known Future Inputs    → fitur kalender (hari, bulan, kuartal, sin/cos encoding)
  └── Observed Past Inputs   → harga OHLCV, RSI, MACD, MA, volatilitas, lag features
         ↓
  Variable Selection Networks (VSN)
         ↓
  LSTM Encoder (lookback 60 hari)
         ↓
  Multi-Head Self-Attention
         ↓
  MIMO Output Layer → Vektor 30 hari sekaligus (satu inferensi)
         ↓
  Quantile Predictions [10%, 25%, 50%, 75%, 90%]
```

### Strategi MIMO
Model menghasilkan **30 prediksi sekaligus** dalam satu forward pass,
bukan secara autoregressive (satu per satu). Ini menghindari error
propagation dan lebih efisien secara komputasi.

---

## 📊 Fitur Aplikasi

| Fitur | Keterangan |
|-------|-----------|
| **Dropdown Bank** | BBCA, BBRI, BMRI, BBNI, BRIS |
| **Horizon Prediksi** | 7–30 hari ke depan (adjustable) |
| **Chart Aktual vs Prediksi** | Plotly interactive dengan confidence band |
| **Attention Weights** | Variable importance + temporal heatmap |
| **Analisis Teknikal** | Candlestick, RSI, Volatilitas |
| **Model Cache** | Auto-cache 24 jam untuk kecepatan |

---

## 🔧 Konfigurasi

Edit `model.py` → `DEFAULT_CONFIG` untuk mengubah hyperparameter:

```python
DEFAULT_CONFIG = {
    "encoder_length":      60,   # lookback window (hari trading)
    "hidden_size":         64,   # ukuran hidden layer
    "attention_head_size": 4,    # jumlah attention head
    "dropout":             0.15,
    "hidden_continuous_size": 32,
    "batch_size":          64,
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
}
```

---

## 📦 Dependensi Utama

| Library | Versi | Fungsi |
|---------|-------|--------|
| `pytorch-forecasting` | ≥1.0 | Model TFT |
| `pytorch-lightning` | ≥2.1 | Training framework |
| `torch` | ≥2.1 | Deep learning backend |
| `yfinance` | ≥0.2.36 | Data saham |
| `streamlit` | ≥1.32 | Web UI |
| `plotly` | ≥5.18 | Visualisasi interaktif |
| `scikit-learn` | ≥1.3 | RobustScaler |

---

## ⚠️ Mode Fallback

Jika `pytorch-forecasting` **tidak tersedia**, aplikasi otomatis
menggunakan **FallbackTFTModel** (LSTM sederhana + simulasi realistis)
sehingga UI tetap berjalan penuh tanpa error.

---

## ⚠️ Disclaimer

> Prediksi ini dibuat untuk **tujuan edukasi dan riset**.
> Bukan merupakan saran investasi.
> Selalu lakukan riset mandiri sebelum berinvestasi.
