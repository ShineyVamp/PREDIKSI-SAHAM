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

# Versi skema cache. Naikkan angka ini kalau struktur data cache berubah,
# supaya cache lama yang formatnya beda otomatis dianggap tidak valid.
CACHE_SCHEMA_VERSION = 5

# Berapa lama cache dianggap masih segar (jam).
CACHE_MAX_AGE_HOURS = 24

# Kunci yang wajib ada di dict "future" agar hasil cache bisa dipakai UI.
_REQUIRED_FUTURE_KEYS = ("close", "close_lower", "close_upper")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CACHE
# ══════════════════════════════════════════════════════════════════════════════
def _serialize_future(future) -> dict:
    """
    Ubah dict prediksi masa depan menjadi dict berisi list murni.

    'future' selalu berupa dict seperti:
        {"close": np.array([...]), "close_lower": ..., "close_upper": ..., ...}

    Versi lama memakai np.array(future).tolist() yang justru merusak struktur
    dict dan membuat cache tidak bisa dibaca lagi. Fungsi ini memperbaikinya:
    setiap value dikonversi ke list satu per satu sehingga aman untuk pickle.
    """
    if not isinstance(future, dict):
        # Bentuk lama / tidak terduga. Bungkus seadanya agar tidak crash.
        return {"close": np.asarray(future, dtype=float).ravel().tolist()}

    out = {}
    for k, v in future.items():
        out[k] = np.asarray(v, dtype=float).ravel().tolist()
    return out


def _deserialize_future(future) -> dict:
    """Kembalikan dict future dari list menjadi dict numpy array."""
    if not isinstance(future, dict):
        return {}
    return {k: np.asarray(v, dtype=float) for k, v in future.items()}


def _serialize_backtest(bt) -> dict:
    """Ubah dict backtest (matriks numpy) menjadi struktur aman untuk pickle."""
    if not isinstance(bt, dict):
        return {}
    out = {}
    for k, v in bt.items():
        out[k] = np.asarray(v, dtype=float)
    return out


def _deserialize_backtest(bt) -> dict:
    if not isinstance(bt, dict):
        return {}
    return {k: np.asarray(v, dtype=float) for k, v in bt.items()}


def save_model_cache(key: str, data: dict) -> bool:
    """
    Simpan hasil (bukan bobot model) ke disk.

    Args:
        key  : Identifier unik (mis. 'BBCA.JK_30_balanced_s5')
        data : Dict berisi backtest (matriks), attention, future

    Returns:
        True jika berhasil disimpan.
    """
    cache_path = _get_cache_path(key)
    try:
        serializable = {
            "schema":    CACHE_SCHEMA_VERSION,
            "backtest":  _serialize_backtest(data.get("backtest", {})),
            "attention": _serialize_attention(data.get("attention", {})),
            "future":    _serialize_future(data.get("future", {})),
            "timestamp": datetime.now().isoformat(),
            "key":       key,
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
    Muat cache hasil prediksi dari disk.

    Mengembalikan dict siap pakai, atau None bila cache tidak ada, kedaluwarsa,
    formatnya lama, atau rusak. Pemanggil cukup memeriksa `if cached is None`.
    """
    cache_path = _get_cache_path(key)
    if not cache_path.exists():
        return None

    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours > CACHE_MAX_AGE_HOURS:
            logger.info(f"Cache kedaluwarsa ({age_hours:.1f} jam). Menghapus.")
            cache_path.unlink()
            return None

        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        # Tolak cache dengan skema lama (format datanya tidak kompatibel).
        if data.get("schema") != CACHE_SCHEMA_VERSION:
            logger.info("Cache versi lama terdeteksi. Mengabaikan.")
            cache_path.unlink()
            return None

        future = _deserialize_future(data.get("future", {}))

        # Validasi isi: kalau kunci penting hilang, anggap cache tidak valid.
        if not all(k in future and len(future[k]) > 0 for k in _REQUIRED_FUTURE_KEYS):
            logger.info("Cache tidak lengkap. Mengabaikan.")
            return None

        result = {
            "backtest":  _deserialize_backtest(data.get("backtest", {})),
            "attention": _deserialize_attention(data.get("attention", {})),
            "future":    future,
            "timestamp": data.get("timestamp"),
        }
        logger.info(f"Cache dimuat: {cache_path} (dibuat {mtime.strftime('%H:%M:%S')})")
        return result

    except Exception as e:
        # Cache rusak: hapus supaya run berikutnya bersih, lalu anggap miss.
        logger.warning(f"Gagal memuat cache: {e}")
        try:
            cache_path.unlink()
        except OSError:
            pass
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
