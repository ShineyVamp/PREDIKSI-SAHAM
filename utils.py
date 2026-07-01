import os
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tft_utils")

CACHE_DIR = Path(".tft_cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_SCHEMA_VERSION = 10
CACHE_MAX_AGE_HOURS = 24
_REQUIRED_FUTURE_KEYS = ("close", "close_lower", "close_upper")


def _serialize_future(future) -> dict:
    if not isinstance(future, dict):
        return {"close": np.asarray(future, dtype=float).ravel().tolist()}
    return {k: np.asarray(v, dtype=float).ravel().tolist() for k, v in future.items()}


def _deserialize_future(future) -> dict:
    if not isinstance(future, dict):
        return {}
    return {k: np.asarray(v, dtype=float) for k, v in future.items()}


def _serialize_backtest(bt) -> dict:
    if not isinstance(bt, dict):
        return {}
    return {k: np.asarray(v, dtype=float) for k, v in bt.items()}


def _deserialize_backtest(bt) -> dict:
    if not isinstance(bt, dict):
        return {}
    return {k: np.asarray(v, dtype=float) for k, v in bt.items()}


def save_model_cache(key: str, data: dict) -> bool:
    cache_path = _get_cache_path(key)
    try:
        serializable = {
            "schema": CACHE_SCHEMA_VERSION,
            "backtest": _serialize_backtest(data.get("backtest", {})),
            "attention": _serialize_attention(data.get("attention", {})),
            "future": _serialize_future(data.get("future", {})),
            "timestamp": datetime.now().isoformat(),
            "key": key,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cache disimpan: {cache_path}")
        return True
    except Exception as e:
        logger.warning(f"Gagal menyimpan cache: {e}")
        return False


def load_model_cache(key: str):
    cache_path = _get_cache_path(key)
    if not cache_path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime).total_seconds() / 3600 > CACHE_MAX_AGE_HOURS:
            cache_path.unlink()
            return None
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("schema") != CACHE_SCHEMA_VERSION:
            cache_path.unlink()
            return None
        future = _deserialize_future(data.get("future", {}))
        if not all(k in future and len(future[k]) > 0 for k in _REQUIRED_FUTURE_KEYS):
            return None
        return {
            "backtest": _deserialize_backtest(data.get("backtest", {})),
            "attention": _deserialize_attention(data.get("attention", {})),
            "future": future,
            "timestamp": data.get("timestamp"),
        }
    except Exception as e:
        logger.warning(f"Gagal memuat cache: {e}")
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def save_panel_cache(key: str, attention: dict, per_ticker: dict, n_banks: int) -> bool:
    cache_path = _get_cache_path(key)
    try:
        ser_per = {}
        for tk, v in (per_ticker or {}).items():
            ser_per[str(tk)] = {
                "backtest": _serialize_backtest(v.get("backtest", {})),
                "future": _serialize_future(v.get("future", {})),
            }
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "panel": True,
            "attention": _serialize_attention(attention or {}),
            "per_ticker": ser_per,
            "n_banks": int(n_banks),
            "timestamp": datetime.now().isoformat(),
            "key": key,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cache panel disimpan: {cache_path} ({len(ser_per)} bank)")
        return True
    except Exception as e:
        logger.warning(f"Gagal menyimpan cache panel: {e}")
        return False


def load_panel_cache(key: str):
    cache_path = _get_cache_path(key)
    if not cache_path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime).total_seconds() / 3600 > CACHE_MAX_AGE_HOURS:
            cache_path.unlink()
            return None
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("schema") != CACHE_SCHEMA_VERSION or not data.get("panel"):
            return None
        per = {}
        for tk, v in data.get("per_ticker", {}).items():
            future = _deserialize_future(v.get("future", {}))
            if not all(k in future and len(future[k]) > 0 for k in _REQUIRED_FUTURE_KEYS):
                continue       
            per[str(tk)] = {
                "backtest": _deserialize_backtest(v.get("backtest", {})),
                "future": future,
            }
        if not per:
            return None
        return {
            "attention": _deserialize_attention(data.get("attention", {})),
            "per_ticker": per,
            "n_banks": int(data.get("n_banks", len(per))),
        }
    except Exception as e:
        logger.warning(f"Gagal memuat cache panel: {e}")
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def clear_all_cache() -> int:
    count = 0
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
        count += 1
    logger.info(f"Menghapus {count} file cache.")
    return count


def _get_cache_path(key: str) -> Path:
    safe = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"tft_{safe}.pkl"


def _serialize_attention(attn: dict) -> dict:
    result = {}
    for k, v in attn.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = {dk: (float(dv) if isinstance(dv, (np.floating, float)) else dv)
                         for dk, dv in v.items()}
        else:
            result[k] = v
    return result


def _deserialize_attention(attn: dict) -> dict:
    result = {}
    for k, v in attn.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            result[k] = np.array(v)
        else:
            result[k] = v
    return result


def check_dependencies() -> dict:
    return {m: _import_check(m) for m in (
        "torch", "pytorch_forecasting", "lightning", "yfinance",
        "streamlit", "plotly", "sklearn", "pandas", "numpy")}


def _import_check(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def format_idr(value: float) -> str:
    if value >= 1_000_000:
        return f"Rp {value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"Rp {value/1_000:.1f}K"
    return f"Rp {value:,.0f}"


def safe_pct_change(new_val: float, old_val: float) -> float:
    if old_val == 0:
        return 0.0
    return (new_val - old_val) / abs(old_val) * 100
