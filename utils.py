"""
utils.py
════════
Utilitas pendukung aplikasi TFT Stock Analytics:
  · Caching model dan prediksi ke disk (pickle + joblib)
  · Logging helper
  · Konversi tipe data
  · Environment check
"""

import os
import pickle
import hashlib
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Setup logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("tft_utils")

# ── Direktori cache ────────────────────────────────────────────────────────
CACHE_DIR = Path(".tft_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CACHE
# ══════════════════════════════════════════════════════════════════════════════
def save_model_cache(key: str, data: dict) -> bool:
    """
    Simpan model dan hasil prediksi ke disk.

    Args:
        key  : Identifier unik (e.g. 'BBCA.JK_30_50_0.001')
        data : Dict berisi model, predictions, actuals, attention, future

    Returns:
        True jika berhasil disimpan
    """
    cache_path = _get_cache_path(key)
    try:
        # Simpan hanya data numerik (bukan model PyTorch) untuk keamanan
        serializable = {
            "predictions": np.array(data["predictions"]).tolist(),
            "actuals":     np.array(data["actuals"]).tolist(),
            "attention":   _serialize_attention(data.get("attention", {})),
            "future":      np.array(data["future"]).tolist(),
            "timestamp":   datetime.now().isoformat(),
            "key":         key,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cache disimpan: {cache_path}")
        return True
    except Exception as e:
        logger.warning(f"Gagal menyimpan cache: {e}")
        return False


def load_model_cache(key: str) -> dict | None:
    """
    Muat cache model dari disk.

    Returns:
        Dict berisi data cache, atau None jika tidak ditemukan / kedaluwarsa.
    """
    cache_path = _get_cache_path(key)
    if not cache_path.exists():
        return None

    try:
        # Cek umur cache (max 24 jam)
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours > 24:
            logger.info(f"Cache kedaluwarsa ({age_hours:.1f} jam). Menghapus.")
            cache_path.unlink()
            return None

        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        # Konversi kembali ke numpy
        result = {
            "model":       None,  # model tidak disimpan di cache ringan
            "predictions": np.array(data["predictions"]),
            "actuals":     np.array(data["actuals"]),
            "attention":   _deserialize_attention(data.get("attention", {})),
            "future":      np.array(data["future"]),
        }
        logger.info(f"Cache dimuat: {cache_path} (dibuat {mtime.strftime('%H:%M:%S')})")
        return result

    except Exception as e:
        logger.warning(f"Gagal memuat cache: {e}")
        return None


def clear_all_cache() -> int:
    """Hapus semua file cache. Returns jumlah file yang dihapus."""
    count = 0
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
        count += 1
    logger.info(f"Menghapus {count} file cache.")
    return count


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _get_cache_path(key: str) -> Path:
    """Buat path cache berdasarkan hash dari key."""
    safe_key = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"tft_{safe_key}.pkl"


def _serialize_attention(attn: dict) -> dict:
    """Konversi numpy arrays ke list agar bisa di-pickle."""
    result = {}
    for k, v in attn.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = {dk: float(dv) if isinstance(dv, (np.floating, float)) else dv
                         for dk, dv in v.items()}
        else:
            result[k] = v
    return result


def _deserialize_attention(attn: dict) -> dict:
    """Konversi list kembali ke numpy arrays."""
    result = {}
    for k, v in attn.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            result[k] = np.array(v)
        else:
            result[k] = v
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_dependencies() -> dict:
    """
    Periksa ketersediaan semua library yang dibutuhkan.

    Returns:
        Dict {library_name: bool}
    """
    deps = {
        "torch":                _import_check("torch"),
        "pytorch_forecasting":  _import_check("pytorch_forecasting"),
        "pytorch_lightning":    _import_check("pytorch_lightning"),
        "yfinance":             _import_check("yfinance"),
        "streamlit":            _import_check("streamlit"),
        "plotly":               _import_check("plotly"),
        "sklearn":              _import_check("sklearn"),
        "pandas":               _import_check("pandas"),
        "numpy":                _import_check("numpy"),
    }
    return deps


def _import_check(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def format_idr(value: float) -> str:
    """Format angka ke format Rupiah Indonesia."""
    if value >= 1_000_000:
        return f"Rp {value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"Rp {value/1_000:.1f}K"
    return f"Rp {value:,.0f}"


def safe_pct_change(new_val: float, old_val: float) -> float:
    """Hitung perubahan persentase dengan aman (hindari div by zero)."""
    if old_val == 0:
        return 0.0
    return (new_val - old_val) / abs(old_val) * 100
